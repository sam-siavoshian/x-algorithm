"""The surrogate: one transformer standing in for Phoenix (MODEL.md).

Inputs per post are built the way Phoenix builds a candidate token: every
feature is projected to the model width and the projections are summed into
one token (phoenix/xrex/models/recsys_feature_prep.py::build_feature_prep_streams).
Outputs are Phoenix's 64 action heads by ActionName index, plus distributional
heads over reach, which is what the product reports as odds.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from xviral.constants import ENGAGEMENT_COUNT_LOG2_SCALE, POST_AGE_NUM_BUCKETS
from xviral.model.config import HistHead, ModelCfg, TrainCfg
from xviral.model.episodes import I_LIKES, I_VIEWS, Episode
from xviral.model.features import FeatureTable
from xviral.model.layers import RMSNorm, Trunk

TIME_GAP_CARDINALITY = 19  # recsys_feature_prep.py:52
EMB_INIT_STD = 0.1         # lookup tables start well below the O(1) scale of the MLP feature groups


def post_age_bucket(age_s: torch.Tensor) -> torch.Tensor:
    """Hourly bucket, 0 = unknown, 81 = overflow (recsys_feature_prep.py::_compute_post_age_bucket_linear)."""
    b = (age_s // 60) // 60 + 1
    return torch.where(age_s < 0, torch.zeros_like(b), b.clamp(0, POST_AGE_NUM_BUCKETS - 1))


def time_gap_bucket(gap_s: torch.Tensor) -> torch.Tensor:
    """Log-spaced bucket with edges 60 * 2^k seconds (recsys_feature_prep.py::_compute_time_gap_bucket)."""
    edges = 60 * 2 ** torch.arange(TIME_GAP_CARDINALITY - 2, device=gap_s.device)
    b = 1 + torch.bucketize(gap_s, edges, right=True)
    return torch.where(gap_s < 0, torch.zeros_like(b), b)


# --- distributional heads ---------------------------------------------------------


def hist_edges(spec: HistHead, device) -> torch.Tensor:
    return torch.linspace(spec.lo, spec.hi, spec.bins + 1, device=device)


def hist_targets(y: torch.Tensor, spec: HistHead, sigma_bins: float) -> torch.Tensor:
    """HL-Gauss soft targets: the mass a Gaussian centred on y puts in each bin.

    Classification over bins with smoothed targets is more robust than regression for noisy,
    heavy-tailed outcomes, and yields a full distribution. X models its own heavy-tailed
    quantity (dwell time) the same way in cumulative form, as thresholds plus restoration
    (CREAD, phoenix/xrex/models/loss_recsys.py:110-187).
    """
    edges = hist_edges(spec, y.device)
    width = (spec.hi - spec.lo) / spec.bins
    yc = y.clamp(spec.lo + 1e-4, spec.hi - 1e-4)
    cdf = 0.5 * (1.0 + torch.erf((edges - yc[..., None]) / (sigma_bins * width * math.sqrt(2.0))))
    p = cdf[..., 1:] - cdf[..., :-1]
    return p / p.sum(-1, keepdim=True).clamp_min(1e-12)


def hist_mean(logits: torch.Tensor, spec: HistHead) -> torch.Tensor:
    e = hist_edges(spec, logits.device)
    return (logits.softmax(-1) * (0.5 * (e[:-1] + e[1:]))).sum(-1)


def hist_tail(logits: torch.Tensor, spec: HistHead, thr: torch.Tensor | float) -> torch.Tensor:
    """P(y >= thr): bins above the threshold, plus the covered fraction of the bin containing it."""
    e = hist_edges(spec, logits.device)
    thr = torch.as_tensor(thr, device=logits.device, dtype=torch.float32)
    width = (spec.hi - spec.lo) / spec.bins
    frac = ((e[1:] - thr[..., None]) / width).clamp(0.0, 1.0)
    return (logits.softmax(-1) * frac).sum(-1)


def hist_quantile(logits: torch.Tensor, spec: HistHead, q: float) -> torch.Tensor:
    e = hist_edges(spec, logits.device)
    p = logits.softmax(-1)
    cdf = p.cumsum(-1)
    idx = (cdf < q).sum(-1).clamp(max=spec.bins - 1)
    below = torch.where(idx > 0, cdf.gather(-1, (idx - 1).clamp(min=0)[..., None]).squeeze(-1), torch.zeros_like(idx, dtype=cdf.dtype))
    mass = p.gather(-1, idx[..., None]).squeeze(-1).clamp_min(1e-9)
    width = (spec.hi - spec.lo) / spec.bins
    return e[idx] + ((q - below) / mass).clamp(0, 1) * width


# --- token encoders ----------------------------------------------------------------


class PostEncoder(nn.Module):
    """Content, author-at-post-time and context features of one post, summed into one vector."""

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        # Feature groups are summed into one token, so they must arrive at comparable scale. The post
        # embedding is a unit vector (entries ~ 1/sqrt(dim)); it is multiplied by sqrt(dim) before its
        # MLP, and lookup tables start small. Phoenix does the same job with explicit input scaling
        # (phoenix/xrex/models/scaling.py). Without this, content enters at about 1% of the token.
        mlp = lambda i: nn.Sequential(nn.Linear(i, d), nn.GELU(approximate="tanh"), nn.Linear(d, d))  # noqa: E731
        self.emb = mlp(cfg.emb_dim) if cfg.emb_dim else None
        self.emb_scale = math.sqrt(cfg.emb_dim) if cfg.emb_dim else 1.0
        self.no_emb = nn.Parameter(torch.zeros(d))
        self.sid = nn.Embedding(cfg.sid_levels * cfg.sid_vocab, d) if cfg.sid_levels else None
        self.num = mlp(cfg.n_num) if cfg.n_num else None
        self.cat = nn.ModuleList(nn.Embedding(c, d) for c in cfg.cat_cards)
        self.aux = nn.ModuleDict({k: mlp(v) for k, v in cfg.aux_dims.items()})
        self.aux_scale = {k: math.sqrt(v) for k, v in cfg.aux_dims.items()}
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=EMB_INIT_STD)

    def forward(self, t: dict, rows: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        x = 0.0
        if self.num is not None:
            x = x + self.num(t["num"][rows])
        for i, emb in enumerate(self.cat):
            x = x + emb(t["cat"][rows][..., i].long())
        if cfg.use_content:
            if self.emb is not None and "emb" in t:
                has = t["has_emb"][rows][..., None]
                x = x + torch.where(has, self.emb(t["emb"][rows].float() * self.emb_scale), self.no_emb.expand(*rows.shape, -1))
            if self.sid is not None and "sid" in t:
                codes = t["sid"][rows].long()                                    # [..., L], 0 = missing
                offs = torch.arange(codes.shape[-1], device=codes.device) * cfg.sid_vocab
                x = x + (self.sid(codes + offs) * (codes > 0)[..., None]).sum(-2)
            for name, proj in self.aux.items():
                if f"aux_{name}" in t:
                    x = x + proj(t[f"aux_{name}"][rows].float() * self.aux_scale[name]) * t[f"has_aux_{name}"][rows][..., None]
        return x


class OutcomeEncoder(nn.Module):
    """Engagement counts of a post, as Phoenix encodes them.

    Phoenix feeds log2(count + 1) / 32 for like, reply, repost, quote and view, plus a constant,
    through a 6 -> 64 -> D MLP (recsys_feature_prep.py:730-754). Our sources lack some counts, so
    presence bits are added, along with the two follower-normalised ratios. Ratios are computed
    here from counts and followers; no label column is ever read.
    """

    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(5 + 5 + 2 + 1, 64), nn.GELU(approximate="tanh"), nn.Linear(64, d))

    def forward(self, counts: torch.Tensor, visible: torch.Tensor, followers: torch.Tensor) -> torch.Tensor:
        present = (counts >= 0) & visible[..., None]
        c = counts.clamp(min=0).float()
        logc = torch.log2(c + 1.0) / ENGAGEMENT_COUNT_LOG2_SCALE * present
        f = torch.log10(followers.clamp(min=0) + 1.0)[..., None]
        known = (followers > 0)[..., None]     # 0 = follower count unknown: no ratio
        ratio = (torch.log10(c[..., [I_LIKES, I_VIEWS]] + 1.0) - f) * (present[..., [I_LIKES, I_VIEWS]] & known)
        x = torch.cat([logc, present.float(), ratio, visible[..., None].float()], dim=-1)
        return self.mlp(x) * visible[..., None]


class NeighborPool(nn.Module):
    """Attention pooling of a candidate's retrieved neighbours into the candidate token."""

    def __init__(self, d: int):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.feat = nn.Sequential(nn.Linear(2, 32), nn.GELU(approximate="tanh"), nn.Linear(32, d))
        self.scale = d ** -0.5

    def forward(self, cand: torch.Tensor, nbr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        att = (self.q(cand)[..., None, :] * self.k(nbr)).sum(-1) * self.scale
        att = att.masked_fill(~mask, float("-inf"))
        any_n = mask.any(-1, keepdim=True)
        w = torch.where(any_n, att.softmax(-1), torch.zeros_like(att))
        return (w[..., None] * self.v(nbr)).sum(-2)


class HistoryKNN(nn.Module):
    """Explicit in-context estimates from the account's own visible history.

    For each candidate, the weighted mean of earlier posts' outcomes (relative to the account's own
    median) under four kernels:
      content   cosine similarity on the raw post embedding, softmax at two temperatures:
                "how did this account's most similar posts do"
      hour      Gaussian in circular hour-of-day distance (time-zone free, since it is a difference):
                "how do this account's posts do at this hour"
      format    same media/link/video combination: "how do this account's posts in this format do"
    The trunk's attention can learn all of these; handing it the estimates makes the account-specific
    signal available from the first training step.
    """

    TEMPS = (0.05, 0.2)
    HOUR_SIGMA = 2.0
    N_FEATS = 14

    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(self.N_FEATS, 64), nn.GELU(approximate="tanh"), nn.Linear(64, d))

    def forward(self, e_c: torch.Tensor, e_h: torch.Tensor, counts_h: torch.Tensor, visible: torch.Tensor, has_emb_h: torch.Tensor,
                med_like: torch.Tensor, med_view: torch.Tensor, hour_c: torch.Tensor, hour_h: torch.Tensor,
                fmt_c: torch.Tensor, fmt_h: torch.Tensor) -> torch.Tensor:
        sim = F.normalize(e_c, dim=-1) @ F.normalize(e_h, dim=-1).transpose(1, 2)        # [B, C, H]
        dh = (hour_c[:, :, None] - hour_h[:, None, :]).abs()
        dh = torch.minimum(dh, 24.0 - dh)
        hour_logit = -(dh ** 2) / (2 * self.HOUR_SIGMA ** 2)
        same_fmt = fmt_c[:, :, None] == fmt_h[:, None, :]
        neg = float("-inf")

        def wmean(logits: torch.Tensor, ok: torch.Tensor, rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            any_ok = ok.any(-1, keepdim=True)
            w = logits.masked_fill(~ok, neg).softmax(-1)
            w = torch.where(any_ok, w, torch.zeros_like(w))
            return (w * rel[:, None, :]).sum(-1), any_ok.squeeze(-1).float()

        feats = []
        for idx, med in ((I_LIKES, med_like), (I_VIEWS, med_view)):
            ok = visible & (counts_h[..., idx] >= 0)                                      # [B, H]
            rel = (torch.log10(counts_h[..., idx].clamp(min=0).float() + 1.0) - med[:, None]) * ok
            okc = ok[:, None, :].expand_as(sim)
            for temp in self.TEMPS:
                feats.append(wmean(sim / temp, okc & has_emb_h[:, None, :], rel)[0])
            feats.append(wmean(hour_logit, okc, rel)[0])
            f_mean, f_any = wmean(torch.zeros_like(sim), okc & same_fmt, rel)
            feats += [f_mean, f_any, okc.any(-1).float()]
        vis = visible & has_emb_h
        feats.append(sim.masked_fill(~vis[:, None, :], -1.0).max(-1).values * vis.any(-1)[:, None])
        feats.append((sim * vis[:, None, :]).sum(-1) / vis.sum(-1).clamp(min=1)[:, None])
        return self.mlp(torch.stack(feats, dim=-1))


class Surrogate(nn.Module):
    def __init__(self, cfg: ModelCfg, action_index: list[int]):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.register_buffer("action_index", torch.tensor(action_index, dtype=torch.long), persistent=False)
        self.post = PostEncoder(cfg)
        self.outcome = OutcomeEncoder(d)
        self.nbr_pool = NeighborPool(d)
        self.hist_knn = HistoryKNN(d)
        self.ctx = nn.Sequential(nn.Linear(cfg.n_ctx, d), nn.GELU(approximate="tanh"), nn.Linear(d, d))
        self.age_emb = nn.Embedding(POST_AGE_NUM_BUCKETS, d)
        self.gap_emb = nn.Embedding(TIME_GAP_CARDINALITY, d)
        nn.init.normal_(self.age_emb.weight, std=EMB_INIT_STD)
        nn.init.normal_(self.gap_emb.weight, std=EMB_INIT_STD)
        self.role = nn.Parameter(torch.zeros(3, d))   # history, candidate, neighbour
        nn.init.normal_(self.role, std=0.02)
        self.in_norm = RMSNorm(d)
        self.drop = nn.Dropout(cfg.dropout)
        self.trunk = Trunk(d, cfg.n_layers, cfg.n_q_heads, cfg.n_kv_heads, cfg.ffn_mult, cfg.dropout)
        self.actions = nn.Linear(d, cfg.n_actions)
        self.hist = nn.ModuleDict({k: nn.Linear(d, h.bins) for k, h in cfg.hist_heads.items()})
        self.pair_log_scale = nn.Parameter(torch.tensor(math.log(4.0)))
        self.live_enc = None
        if cfg.live:
            from xviral.model.live import LiveEncoder

            self.live_enc = LiveEncoder(**cfg.live)
            self.live_proj = nn.Sequential(nn.Linear(self.live_enc.dim, d), nn.GELU(approximate="tanh"), nn.Linear(d, d))
            nn.init.zeros_(self.live_proj[-1].weight)   # starts as a no-op on a stage-A checkpoint
            nn.init.zeros_(self.live_proj[-1].bias)

    def live_tokens(self, table: FeatureTable, ep: Episode) -> torch.Tensor:
        """Stage B: encode the valid candidates' rendered posts with the adapter-tuned encoder."""
        idx = ep.cand_mask.nonzero(as_tuple=False)
        rows = ep.cand_rows[ep.cand_mask].tolist()
        vec = self.live_enc.encode_chunked([table.texts[r] for r in rows])
        out = vec.new_zeros(*ep.cand_rows.shape, self.live_enc.dim)
        out[idx[:, 0], idx[:, 1]] = vec
        return self.live_proj(out)

    def forward(self, table: FeatureTable, ep: Episode) -> dict:
        t, cfg = table.t, self.cfg
        created, counts, followers = t["created_s"], t["counts"], t["followers"]
        H = ep.hist_rows.shape[1]

        hist_age = ep.cutoff_s[:, None] - created[ep.hist_rows]
        vis = ep.hist_visible if cfg.use_history_outcomes else torch.zeros_like(ep.hist_visible)
        hist = (self.post(t, ep.hist_rows) + self.outcome(counts[ep.hist_rows], vis, followers[ep.hist_rows])
                + self.age_emb(post_age_bucket(hist_age)) + self.gap_emb(time_gap_bucket(hist_age)) + self.role[0])

        cand_content = self.post(t, ep.cand_rows)
        observed = (ep.cand_obs_counts >= 0).any(-1)
        cand = (cand_content + self.outcome(ep.cand_obs_counts, observed, followers[ep.cand_rows])
                + self.age_emb(post_age_bucket(ep.cand_obs_age_s))
                + self.gap_emb(time_gap_bucket(created[ep.cand_rows] - ep.cutoff_s[:, None])) + self.role[1])
        if cfg.use_neighbors and ep.nbr_rows.shape[-1] > 0:
            age_h = (created[ep.cand_rows][..., None] - created[ep.nbr_rows]).clamp(min=0).float() / 3600.0
            nfeat = torch.stack([ep.nbr_cos, torch.log1p(age_h) / 6.0], dim=-1)
            nbr = (self.post(t, ep.nbr_rows) + self.outcome(counts[ep.nbr_rows], ep.nbr_visible, followers[ep.nbr_rows])
                   + self.nbr_pool.feat(nfeat) + self.role[2])
            cand = cand + self.nbr_pool(cand_content, nbr, ep.nbr_mask)

        if self.live_enc is not None and cfg.use_content:
            cand = cand + self.live_tokens(table, ep)
        if cfg.use_hist_knn and cfg.use_history_outcomes and "emb" in t:
            hour = lambda rows: (created[rows] % 86400).float() / 3600.0  # noqa: E731
            e_c, e_h = t["emb"][ep.cand_rows].float(), t["emb"][ep.hist_rows].float()
            has_h = t["has_emb"][ep.hist_rows] if cfg.use_content else torch.zeros_like(ep.hist_visible)
            cand = cand + self.hist_knn(e_c, e_h, counts[ep.hist_rows], ep.hist_visible, has_h, ep.ctx[:, 3], ep.ctx[:, 5],
                                        hour(ep.cand_rows), hour(ep.hist_rows), t["fmt"][ep.cand_rows], t["fmt"][ep.hist_rows])

        ctx = ep.ctx if cfg.use_history_outcomes else torch.cat([ep.ctx[:, :1], torch.zeros_like(ep.ctx[:, 1:])], dim=1)
        x = torch.cat([self.ctx(ctx)[:, None], hist, cand], dim=1)
        valid = torch.cat([torch.ones_like(ep.hist_mask[:, :1]), ep.hist_mask, ep.cand_mask], dim=1)
        x = self.trunk(self.drop(self.in_norm(x)), valid, 1, H)[:, 1 + H:]
        return {"actions": self.actions(x), "hist": {k: head(x) for k, head in self.hist.items()}}

    # --- loss ---

    def loss(self, ep: Episode, out: dict, tcfg: TrainCfg) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        parts: dict[str, torch.Tensor] = {}
        total = out["actions"].new_zeros(())
        for name, spec in cfg.hist_heads.items():
            y = ep.y[name]
            m = ~torch.isnan(y)
            if m.any():
                tgt = hist_targets(y[m], spec, cfg.hl_gauss_sigma_bins)
                ce = -(tgt * F.log_softmax(out["hist"][name][m].float(), dim=-1)).sum(-1)
                w = ep.w[m]
                parts[name] = (ce * w).sum() / w.sum().clamp_min(1e-8)
                total = total + spec.weight * parts[name]
        r = ep.rates
        m = ~torch.isnan(r)
        if m.any():
            logits = out["actions"][..., self.action_index].float()
            bce = F.binary_cross_entropy_with_logits(logits[m], r[m], reduction="none")
            w = ep.w[..., None].expand_as(r)[m]
            parts["rates"] = (bce * w).sum() / w.sum().clamp_min(1e-8)
            total = total + tcfg.w_rates * parts["rates"]
        pair = self.pair_loss(ep, out, tcfg)
        if pair is not None:
            parts["pair"] = pair
            total = total + tcfg.w_pair * pair
        return total, {k: float(v.detach()) for k, v in parts.items()}

    def pair_loss(self, ep: Episode, out: dict, tcfg: TrainCfg) -> torch.Tensor | None:
        """RankNet over same-author candidates in an episode: the confound control for content effects.

        A pair uses views when both posts have them, otherwise likes, on the absolute heads, so it
        needs no follower count.
        """
        cfg = self.cfg
        C = ep.cand_rows.shape[1]
        if C < 2:
            return None
        upper = torch.triu(torch.ones(C, C, dtype=torch.bool, device=ep.w.device), diagonal=1)[None]
        losses, used = [], None
        for name in ("views", "likes"):
            y = ep.y[name]
            ok = ~torch.isnan(y)
            both = ok[:, :, None] & ok[:, None, :] & upper
            if used is not None:
                both &= ~used
            used = both if used is None else (used | both)
            dy = torch.nan_to_num(y)[:, :, None] - torch.nan_to_num(y)[:, None, :]
            m = both & (dy.abs() >= tcfg.pair_margin)
            if m.any():
                s = hist_mean(out["hist"][name].float(), cfg.hist_heads[name])
                ds = (s[:, :, None] - s[:, None, :])[m] * self.pair_log_scale.exp()
                losses.append(F.binary_cross_entropy_with_logits(ds, (dy[m] > 0).float(), reduction="none"))
        return torch.cat(losses).mean() if losses else None

    # --- predictions ---

    @torch.no_grad()
    def summarize(self, ep: Episode, out: dict, temperature: dict | None = None) -> dict:
        """Per-candidate predictions the product reports. `temperature` holds per-head calibration."""
        cfg = self.cfg
        T = temperature or {}
        res: dict[str, torch.Tensor] = {}
        for name, spec in cfg.hist_heads.items():
            lg = out["hist"][name].float() / T.get(name, 1.0)
            res[f"{name}_mean"] = hist_mean(lg, spec)
            for q in (0.1, 0.5, 0.9):
                res[f"{name}_q{int(q * 100)}"] = hist_quantile(lg, spec, q)
        reach = out["hist"]["reach"].float() / T.get("reach", 1.0)
        for mult, thr in (("1x", 0.0), ("10x", 1.0), ("100x", 2.0)):
            res[f"p_reach_{mult}"] = hist_tail(reach, cfg.hist_heads["reach"], thr)
        for name in ("rel_like", "rel_view"):
            thr = torch.nan_to_num(ep.own_p90_thr[name], nan=1e6)[:, None]
            res[f"p_own_p90_{name}"] = hist_tail(out["hist"][name].float() / T.get(name, 1.0), cfg.hist_heads[name], thr)
        res["actions"] = out["actions"].float().sigmoid()
        return res
