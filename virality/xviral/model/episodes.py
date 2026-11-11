"""Episodes: the training and scoring unit.

One episode is an author at a cutoff time:
  context     one token of author-baseline statistics, computed from visible history only
  history     the author's previous posts, oldest to newest, right-aligned (padding on the left)
  candidates  the post at the cutoff and, in training, the author's next posts within the pair window
  neighbours  per candidate, similar recent posts by other authors

This is Phoenix's layout ([user | history | candidates], phoenix/README.md) with the author's own
past posts and their outcomes standing in for a viewer's engagement history: the outcomes of an
author's earlier posts are the audience's aggregated history with that author.

Leakage rules, enforced here by construction:
  - history is strictly earlier in the author's timeline than the first candidate;
  - a history post's outcome is hidden if the post is younger than `outcome_mask_hours` at the cutoff,
    because at serving time its counts would still be immature;
  - a neighbour's outcome is hidden under the same rule relative to its candidate, and neighbours
    not strictly earlier than the candidate are removed;
  - candidates never share a split with rows of another split, and they do not see each other
    (candidate isolation in the trunk).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from xviral.model.config import ModelCfg, TrainCfg
from xviral.model.features import FeatureTable

I_LIKES, I_VIEWS = 0, 4


@dataclass
class Episode:
    anchor: torch.Tensor          # [B]
    cutoff_s: torch.Tensor        # [B]
    hist_rows: torch.Tensor       # [B, H]
    hist_mask: torch.Tensor       # [B, H] bool
    hist_visible: torch.Tensor    # [B, H] bool, outcome visible
    cand_rows: torch.Tensor       # [B, C]
    cand_mask: torch.Tensor       # [B, C] bool
    cand_obs_counts: torch.Tensor  # [B, C, 5], -1 = not observed (cold start)
    cand_obs_age_s: torch.Tensor  # [B, C]
    nbr_rows: torch.Tensor        # [B, C, K]
    nbr_mask: torch.Tensor        # [B, C, K] bool
    nbr_visible: torch.Tensor     # [B, C, K] bool
    nbr_cos: torch.Tensor         # [B, C, K]
    ctx: torch.Tensor             # [B, n_ctx]
    y: dict                       # name -> [B, C] float, NaN where undefined
    rates: torch.Tensor           # [B, C, A], NaN where undefined
    w: torch.Tensor               # [B, C]
    own_p90_thr: dict             # 'rel_like' / 'rel_view' -> [B], log10(p90) - log10(median); NaN if undefined


def _nan_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row count, median and 90th percentile over non-NaN entries; NaN where a row has none."""
    n = (~torch.isnan(x)).sum(dim=1)
    safe = torch.where(n[:, None] > 0, x, torch.zeros_like(x))
    med = torch.nanquantile(safe, 0.5, dim=1)
    p90 = torch.nanquantile(safe, 0.9, dim=1)
    nan = torch.full_like(med, float("nan"))
    return n, torch.where(n > 0, med, nan), torch.where(n > 0, p90, nan)


def build(table: FeatureTable, anchors: torch.Tensor, mcfg: ModelCfg, tcfg: TrainCfg, *, train: bool,
          n_candidates: int | None = None, generator: torch.Generator | None = None) -> Episode:
    t = table.t
    dev = anchors.device
    B = anchors.shape[0]
    H = mcfg.max_history
    C = n_candidates or (mcfg.max_candidates if train else 1)
    K = mcfg.max_neighbors
    created, split = t["created_s"], t["split"]
    a_start, a_end = t["a_start"][anchors], t["a_end"][anchors]
    cutoff = created[anchors]
    mask_s = int(tcfg.outcome_mask_hours * 3600)

    hist_idx = anchors[:, None] - H + torch.arange(H, device=dev)[None]
    hist_mask = hist_idx >= a_start[:, None]
    hist_rows = hist_idx.clamp(min=0)
    hist_visible = hist_mask & ((cutoff[:, None] - created[hist_rows]) >= mask_s)

    cand_idx = anchors[:, None] + torch.arange(C, device=dev)[None]
    cand_rows = cand_idx.clamp(max=table.n - 1)
    cand_mask = (cand_idx < a_end[:, None]) & ((created[cand_rows] - cutoff[:, None]) <= int(tcfg.pair_window_days * 86400))
    cand_mask &= split[cand_rows] == split[anchors][:, None]
    if train and C > 1:
        n_c = torch.randint(1, C + 1, (B,), device=dev, generator=generator)
        cand_mask &= torch.arange(C, device=dev)[None] < n_c[:, None]
    cand_mask[:, 0] = True

    # Author baseline from visible history only.
    counts_h = t["counts"][hist_rows]
    nan = float("nan")

    def vis_log(i: int) -> torch.Tensor:
        c = counts_h[..., i]
        return torch.where(hist_visible & (c >= 0), torch.log10(c.clamp(min=0).float() + 1.0), torch.full(c.shape, nan, device=dev))

    n_like, med_like, p90_like = _nan_stats(vis_log(I_LIKES))
    n_view, med_view, p90_view = _nan_stats(vis_log(I_VIEWS))
    has_like = n_like >= tcfg.min_history_for_rel
    has_view = n_view >= tcfg.min_history_for_rel

    def vis(name: str) -> torch.Tensor:
        return torch.where(hist_visible, t[name][hist_rows], torch.full(hist_rows.shape, nan, device=dev))

    _, med_ylike, _ = _nan_stats(vis("y_like"))
    _, med_yreach, _ = _nan_stats(vis("y_reach"))
    z = lambda x: torch.nan_to_num(x, nan=0.0)  # noqa: E731
    ctx = torch.stack([
        hist_mask.sum(1).float() / H, n_like.float() / H, n_view.float() / H,
        z(med_like), z(p90_like), z(med_view), z(p90_view), has_like.float(), has_view.float(),
        torch.log10(t["followers"][anchors].clamp(min=0) + 1.0), z(med_ylike), z(med_yreach),
    ], dim=1)

    # Labels.
    cand_counts = t["counts"][cand_rows]
    nanC = torch.full(cand_rows.shape, nan, device=dev)

    def lab(v: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
        return torch.where(cand_mask & ok, v, nanC)

    log_likes = torch.log10(cand_counts[..., I_LIKES].clamp(min=0).float() + 1.0)
    log_views = torch.log10(cand_counts[..., I_VIEWS].clamp(min=0).float() + 1.0)
    y = {
        "views": lab(log_views, cand_counts[..., I_VIEWS] >= 0),
        "likes": lab(log_likes, cand_counts[..., I_LIKES] >= 0),
        "reach": lab(t["y_reach"][cand_rows], torch.ones_like(cand_mask)),
        "like": lab(t["y_like"][cand_rows], torch.ones_like(cand_mask)),
        "rel_like": lab(log_likes - z(med_like)[:, None], (cand_counts[..., I_LIKES] >= 0) & has_like[:, None]),
        "rel_view": lab(log_views - z(med_view)[:, None], (cand_counts[..., I_VIEWS] >= 0) & has_view[:, None]),
    }
    rates = torch.where(cand_mask[..., None], t["rates"][cand_rows], torch.full((B, C, t["rates"].shape[1]), nan, device=dev))
    w = t["w_row"][cand_rows] * cand_mask.float()
    own_p90_thr = {
        "rel_like": torch.where(has_like, p90_like - med_like, torch.full_like(med_like, nan)),
        "rel_view": torch.where(has_view, p90_view - med_view, torch.full_like(med_view, nan)),
    }

    # Neighbours.
    if "nbr_idx" in t and K > 0:
        nb = t["nbr_idx"][cand_rows][..., :K].long()
        nbr_cos = t["nbr_cos"][cand_rows][..., :K]
        nbr_rows = nb.clamp(min=0)
        age = created[cand_rows][..., None] - created[nbr_rows]
        nbr_mask = (nb >= 0) & cand_mask[..., None] & (age > 0)
        nbr_visible = nbr_mask & (age >= mask_s)
    else:
        nbr_rows = torch.zeros((B, C, 0), dtype=torch.long, device=dev)
        nbr_mask = torch.zeros((B, C, 0), dtype=torch.bool, device=dev)
        nbr_visible, nbr_cos = nbr_mask, torch.zeros((B, C, 0), device=dev)

    # Cold start by default: no observed counts on the candidate.
    obs = torch.full((B, C, 5), -1, dtype=torch.long, device=dev)
    obs_age = torch.zeros((B, C), dtype=torch.long, device=dev)
    if train and tcfg.p_live > 0:
        obs, obs_age = thin_counts(cand_counts, tcfg, generator)

    return Episode(anchor=anchors, cutoff_s=cutoff, hist_rows=hist_rows, hist_mask=hist_mask, hist_visible=hist_visible,
                   cand_rows=cand_rows, cand_mask=cand_mask, cand_obs_counts=obs, cand_obs_age_s=obs_age,
                   nbr_rows=nbr_rows, nbr_mask=nbr_mask, nbr_visible=nbr_visible, nbr_cos=nbr_cos, ctx=ctx, y=y,
                   rates=rates, w=w, own_p90_thr=own_p90_thr)


def thin_counts(final_counts: torch.Tensor, tcfg: TrainCfg, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulated early observation of a post: counts at a random age under an exponential-saturation curve.

    X projects a post's 24 h views from its views at age t with exactly this curve, half-life 8 h
    (home-mixer/util/popular_posts.rs::quality). The half-life here is drawn per example from a wide
    range, because real posts differ and the model must stay uncertain about the curve. Each final
    event is kept with probability g(age), i.e. binomial thinning. This is an assumption standing in
    for real trajectories; it is off unless p_live > 0 and is never used to report accuracy.
    """
    dev = final_counts.device
    shape = final_counts.shape[:-1]
    u = lambda: torch.rand(shape, device=dev, generator=generator)  # noqa: E731
    live = u() < tcfg.p_live
    age = torch.exp(math.log(300.0) + u() * (math.log(86400.0) - math.log(300.0)))
    lo, hi = tcfg.live_half_life_hours
    half = torch.exp(math.log(lo) + u() * (math.log(hi) - math.log(lo))) * 3600.0
    g = (1.0 - torch.exp(-math.log(2.0) * age / half)).clamp(1e-6, 1.0)
    present = final_counts >= 0
    kept = torch.binomial(final_counts.clamp(min=0).float(), g[..., None].expand_as(final_counts).contiguous(), generator=generator).long()
    obs = torch.where(present & live[..., None], kept, torch.full_like(final_counts, -1))
    return obs, torch.where(live, age.long(), torch.zeros_like(age, dtype=torch.long))
