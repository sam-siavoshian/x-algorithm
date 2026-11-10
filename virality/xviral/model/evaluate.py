"""Scoring rows and measuring what matters (DESIGN.md section 7).

Every row is scored alone, with its own full history, exactly as a draft
would be at serving time. Metrics:
  pairwise      same-author pairs within the pair window: which post did better.
                The author baseline alone scores 50% here, so this isolates content skill.
  within_rho    Spearman between prediction and outcome inside each author, averaged.
  global_rho    Spearman across all rows (dominated by author size; reported for completeness).
  own_p90_auc   AUROC for "this post lands in the author's own top 10%", from visible history.
  tier AUROCs   reach >= 1x / 10x / 100x followers, where views exist.
  nll, ece      quality of the predicted distributions and tier probabilities.
"""

from __future__ import annotations

import numpy as np
import torch

from xviral.model import episodes
from xviral.model.config import ModelCfg, TrainCfg
from xviral.model.features import FeatureTable
from xviral.model.model import Surrogate, hist_targets


@torch.no_grad()
def predict_rows(model: Surrogate, table: FeatureTable, rows: torch.Tensor, tcfg: TrainCfg, batch_size: int = 512,
                 temperature: dict | None = None, with_nll: bool = True) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, list] = {}
    for i in range(0, len(rows), batch_size):
        ep = episodes.build(table, rows[i:i + batch_size], model.cfg, tcfg, train=False)
        o = model(table, ep)
        s = model.summarize(ep, o, temperature)
        rec = {k: v[:, 0] for k, v in s.items() if k != "actions"}
        rec["rates"] = s["actions"][:, 0][:, model.action_index]
        for name in model.cfg.hist_heads:
            rec[f"y_{name}"] = ep.y[name][:, 0]
            if with_nll:
                y = ep.y[name][:, 0]
                m = ~torch.isnan(y)
                T = (temperature or {}).get(name, 1.0)
                nll = torch.full_like(y, float("nan"))
                if m.any():
                    tgt = hist_targets(y[m], model.cfg.hist_heads[name], model.cfg.hl_gauss_sigma_bins)
                    nll[m] = -(tgt * torch.log_softmax(o["hist"][name][:, 0][m].float() / T, -1)).sum(-1)
                rec[f"nll_{name}"] = nll
        for name in ("rel_like", "rel_view"):
            rec[f"thr_{name}"] = ep.own_p90_thr[name]
        for k, v in rec.items():
            out.setdefault(k, []).append(v.float().cpu().numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["row"] = rows.cpu().numpy()
    return res


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(a[ok]), _rank(b[ok]))[0, 1])


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    ok = ~(np.isnan(y) | np.isnan(s))
    y, s = y[ok], s[ok]
    n1, n0 = int((y > 0.5).sum()), int((y <= 0.5).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s))
    ss = s[order]
    # average ranks for ties
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y > 0.5].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def same_author_pairs(author: np.ndarray, created_s: np.ndarray, y: np.ndarray, window_s: float, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (i, j) into the given arrays: same author, within the window, |y_i - y_j| >= margin."""
    order = np.lexsort((created_s, author))
    a, t, yy = author[order], created_s[order], y[order]
    I, J = [], []
    n = len(order)
    start = 0
    for i in range(n):
        if i > 0 and a[i] != a[i - 1]:
            start = i
        while t[i] - t[start] > window_s:
            start += 1
        if i > start and not np.isnan(yy[i]):
            js = np.arange(start, i)
            ok = ~np.isnan(yy[js]) & (np.abs(yy[js] - yy[i]) >= margin)
            js = js[ok]
            I.append(np.full(len(js), i))
            J.append(js)
    if not I:
        return np.zeros(0, int), np.zeros(0, int)
    return order[np.concatenate(I)], order[np.concatenate(J)]


def metrics(pred: dict[str, np.ndarray], table: FeatureTable, tcfg: TrainCfg, score_key: dict | None = None) -> dict:
    """score_key: which prediction ranks posts for each target; defaults to the head means."""
    rows = pred["row"]
    author = table.t["author_idx"][rows].cpu().numpy()
    created = table.t["created_s"][rows].cpu().numpy()
    res: dict[str, float] = {"n": int(len(rows))}
    window = tcfg.pair_window_days * 86400
    for target, head in (("like", "like"), ("reach", "reach")):
        y = pred[f"y_{head}"]
        s = pred[(score_key or {}).get(target, f"{head}_mean")]
        if np.isnan(y).all():
            continue
        i, j = same_author_pairs(author, created, y, window, tcfg.pair_margin)
        if len(i):
            agree = (s[i] > s[j]) == (y[i] > y[j])
            ties = s[i] == s[j]
            res[f"pairwise_{target}"] = float((agree & ~ties).mean() + 0.5 * ties.mean())
            res[f"pairs_{target}"] = int(len(i))
        rhos = []
        for a in np.unique(author):
            m = (author == a) & ~np.isnan(y)
            if m.sum() >= 8:
                r = spearman(s[m], y[m])
                if not np.isnan(r):
                    rhos.append(r)
        res[f"within_rho_{target}"] = float(np.mean(rhos)) if rhos else float("nan")
        res[f"global_rho_{target}"] = spearman(s, y)
        res[f"nll_{head}"] = float(np.nanmean(pred[f"nll_{head}"])) if f"nll_{head}" in pred else float("nan")
    for name in ("rel_like", "rel_view"):
        y, thr = pred[f"y_{name}"], pred[f"thr_{name}"]
        ok = ~(np.isnan(y) | np.isnan(thr))
        if ok.sum() > 50:
            lab = np.where(ok, (y >= thr).astype(float), np.nan)
            p = np.where(ok, pred[f"p_own_p90_{name}"], np.nan)
            res[f"own_p90_auc_{name}"] = auroc(lab, p)
            res[f"own_p90_ece_{name}"] = ece(lab, p)
            res[f"own_p90_rate_{name}"] = float(np.nanmean(lab))
            res[f"nll_{name}"] = float(np.nanmean(pred[f"nll_{name}"]))
    y = pred["y_reach"]
    if not np.isnan(y).all():
        for mult, thr in (("1x", 0.0), ("10x", 1.0), ("100x", 2.0)):
            lab = np.where(np.isnan(y), np.nan, (y >= thr).astype(float))
            res[f"auc_reach_{mult}"] = auroc(lab, pred[f"p_reach_{mult}"])
            res[f"ece_reach_{mult}"] = ece(lab, pred[f"p_reach_{mult}"])
    return res


def fit_temperature(model: Surrogate, table: FeatureTable, rows: torch.Tensor, tcfg: TrainCfg, batch_size: int = 512) -> dict[str, float]:
    """Per-head temperature minimising validation NLL of the smoothed targets (author-held-out rows)."""
    model.eval()
    logits: dict[str, list] = {k: [] for k in model.cfg.hist_heads}
    ys: dict[str, list] = {k: [] for k in model.cfg.hist_heads}
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            ep = episodes.build(table, rows[i:i + batch_size], model.cfg, tcfg, train=False)
            o = model(table, ep)
            for k in model.cfg.hist_heads:
                logits[k].append(o["hist"][k][:, 0].float())
                ys[k].append(ep.y[k][:, 0])
    temps: dict[str, float] = {}
    grid = torch.exp(torch.linspace(np.log(0.5), np.log(3.0), 41))
    for k, spec in model.cfg.hist_heads.items():
        lg, y = torch.cat(logits[k]), torch.cat(ys[k])
        m = ~torch.isnan(y)
        if m.sum() < 100:
            temps[k] = 1.0
            continue
        tgt = hist_targets(y[m], spec, model.cfg.hl_gauss_sigma_bins)
        nll = [float(-(tgt * torch.log_softmax(lg[m] / T, -1)).sum(-1).mean()) for T in grid]
        temps[k] = float(grid[int(np.argmin(nll))])
    return temps


def baseline_author_median(table: FeatureTable, rows: torch.Tensor, mcfg: ModelCfg, tcfg: TrainCfg) -> dict[str, np.ndarray]:
    """Predict every post at its author's visible-history median. Pairwise accuracy is 50% by construction;
    listed so the content skill of other models reads against it."""
    out = {"like_mean": [], "reach_mean": []}
    for i in range(0, len(rows), 2048):
        ep = episodes.build(table, rows[i:i + 2048], mcfg, tcfg, train=False)
        out["like_mean"].append(ep.ctx[:, 10].cpu().numpy())
        out["reach_mean"].append(ep.ctx[:, 11].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}
