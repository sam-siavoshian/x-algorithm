"""Retrieval features over the post embedding (DATA_PIPELINE.md 7.5, 7.3).

Phoenix sees what each viewer recently engaged with. We cannot, but we can see
what the corpus recently did with similar content. Leakage rule: a post's
neighbours come only from whole days strictly before the post's own day, so
no neighbour can be later than, or simultaneous with, the target.

Also here: the author centroid (mean embedding of the author's previous
originals) and self-similarity to the author's own posts in the prior 48 h,
which matters because X's DPP reranker suppresses near-duplicates of a
higher-scoring post in the same feed (vm-ranker/scoring/dpp.rs, theta 0.65).
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

DAY_MS = 86_400_000


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def global_neighbors(meta: pl.DataFrame, emb: np.ndarray, k: int = 16, window_days: int = 7,
                     density_cos: float = 0.85) -> pl.DataFrame:
    """meta rows align with emb rows. Needs post_id, author_id, created_at_ms, and outcome columns
    log_reach_ratio, rate_fav, rate_reply, follower_tier_idx (nullable).

    Returns per post: nbr_post_id, nbr_cos, nbr_log_reach, nbr_rate_fav, nbr_rate_reply,
    nbr_tier, nbr_age_h (lists of length <= k), nbr_density_7d, nbr_best_reach_7d.
    """
    e = _unit(np.nan_to_num(emb.astype(np.float32)))
    day = (meta["created_at_ms"].to_numpy() // DAY_MS).astype(np.int64)
    authors = meta["author_id"].to_numpy()
    ts = meta["created_at_ms"].to_numpy()
    outcome = meta["log_reach_ratio"].to_numpy(allow_copy=True).astype(np.float64)
    has_outcome = ~np.isnan(outcome)
    cols = {c: meta[c].to_numpy(allow_copy=True).astype(np.float64) for c in ("rate_fav", "rate_reply", "follower_tier_idx")}
    order = np.argsort(day, kind="stable")
    days = np.unique(day)
    n = len(meta)
    res = {name: [None] * n for name in ("nbr_post_id", "nbr_cos", "nbr_log_reach", "nbr_rate_fav", "nbr_rate_reply", "nbr_tier", "nbr_age_h")}
    density = np.zeros(n, dtype=np.int32)
    best = np.full(n, np.nan)
    ids = meta["post_id"].to_numpy()
    day_sorted = day[order]
    for d in days:
        q_idx = order[np.searchsorted(day_sorted, d, "left"):np.searchsorted(day_sorted, d, "right")]
        lo, hi = np.searchsorted(day_sorted, d - window_days, "left"), np.searchsorted(day_sorted, d, "left")
        cand = order[lo:hi]
        cand = cand[has_outcome[cand]]
        if cand.size == 0:
            continue
        sims = e[q_idx] @ e[cand].T
        same = authors[q_idx][:, None] == authors[cand][None, :]
        sims[same] = -np.inf
        kk = min(k, cand.size)
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        for row, qi in enumerate(q_idx):
            t = top[row][np.argsort(-sims[row, top[row]])]
            t = t[np.isfinite(sims[row, t])]
            ci = cand[t]
            res["nbr_post_id"][qi] = ids[ci].tolist()
            res["nbr_cos"][qi] = sims[row, t].astype(np.float32).tolist()
            res["nbr_log_reach"][qi] = outcome[ci].tolist()
            res["nbr_rate_fav"][qi] = cols["rate_fav"][ci].tolist()
            res["nbr_rate_reply"][qi] = cols["rate_reply"][ci].tolist()
            res["nbr_tier"][qi] = cols["follower_tier_idx"][ci].tolist()
            res["nbr_age_h"][qi] = ((ts[qi] - ts[ci]) / 3.6e6).tolist()
            close = sims[row] >= density_cos
            density[qi] = int(close.sum())
            if close.any():
                best[qi] = float(np.nanmax(outcome[cand[close]]))
    out = pl.DataFrame({"post_id": ids})
    for name, vals in res.items():
        dtype = pl.List(pl.Int64) if name == "nbr_post_id" else pl.List(pl.Float32)
        out = out.with_columns(pl.Series(name, vals, dtype=dtype))
    return out.with_columns(pl.Series("nbr_density_7d", density), pl.Series("nbr_best_reach_7d", best).fill_nan(None))


def author_centroid_and_selfsim(meta: pl.DataFrame, emb: np.ndarray, window: int = 50, selfsim_hours: float = 48.0) -> tuple[np.ndarray, pl.DataFrame]:
    """Per post: mean embedding of the author's previous `window` posts (strictly earlier), and the
    max cosine to the author's own posts within the previous `selfsim_hours`.
    Returns (centroids [N, d] with NaN rows where no history, frame of post_id, self_sim_48h, centroid_n).
    """
    e = _unit(np.nan_to_num(emb.astype(np.float32)))
    n, d = e.shape
    cent = np.full((n, d), np.nan, dtype=np.float32)
    selfsim = np.full(n, np.nan)
    cnt = np.zeros(n, dtype=np.int32)
    authors = meta["author_id"].to_numpy()
    ts = meta["created_at_ms"].to_numpy()
    order = np.lexsort((ts, authors))
    a_sorted = authors[order]
    bounds = np.flatnonzero(np.r_[True, a_sorted[1:] != a_sorted[:-1], True])
    win_ms = selfsim_hours * 3.6e6
    for s, t in zip(bounds[:-1], bounds[1:]):
        idx = order[s:t]
        if len(idx) < 2:
            continue
        ae = e[idx]
        cs = np.vstack([np.zeros((1, d), dtype=np.float64), np.cumsum(ae, axis=0, dtype=np.float64)])
        at = ts[idx]
        for j in range(1, len(idx)):
            lo = max(0, j - window)
            cent[idx[j]] = ((cs[j] - cs[lo]) / (j - lo)).astype(np.float32)
            cnt[idx[j]] = j - lo
            recent = np.flatnonzero((at[:j] >= at[j] - win_ms) & (at[:j] < at[j]))
            if recent.size:
                selfsim[idx[j]] = float((ae[recent] @ ae[j]).max())
    frame = pl.DataFrame({"post_id": meta["post_id"], "self_sim_48h": selfsim, "centroid_n": cnt}).with_columns(pl.col("self_sim_48h").fill_nan(None))
    return cent, frame
