"""The feature table: every post as a row of model-ready arrays, sorted by (author, time).

Built once from the dataset stage's rows plus the embedding stores; loaded as
tensors on the training device so episodes are assembled by index gathers.
Normalisation statistics come from training rows only and are stored with the
table, so serving applies the same transforms.

Layout on disk: <dir>/meta.json and one .npy per array.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from xviral.features.content import ASPECT_CLASSES
from xviral.labels import HEAD_INDEX
from xviral.text.urls import DOMAIN_CLASS_NAMES

log = logging.getLogger(__name__)

# (column, transform). Columns absent from the dataset are skipped, so the table adapts to which
# pipeline stages have run (neighbour features exist only after the embedding stages).
NUM_SPEC: list[tuple[str, str]] = [
    # content scalars home-mixer computes (DATA_PIPELINE.md 7.2)
    ("weighted_text_len", "log1p"), ("newline_count", "log1p"), ("external_url_count", "log1p"), ("media_count", "log1p"),
    ("video_duration_ms", "log1p"), ("mentions_n", "log1p"), ("hashtags_n", "log1p"),
    ("has_url", "bool"), ("has_media", "bool"), ("has_photo", "bool"), ("has_video", "bool"), ("has_gif", "bool"),
    ("video_gt_10s", "bool"), ("video_index_eligible", "bool"), ("thunder_has_video", "bool"), ("is_quote", "bool"),
    ("is_long_post", "bool"),
    # author state at post time (7.3)
    ("followers_at_post", "log1p"), ("author_following", "log1p"), ("account_age_days", "log1p"), ("verified", "bool"),
    ("cold_start_eligible_author", "bool"), ("mutual_share_est", "id"), ("reply_share", "id"), ("graph_pagerank", "log1p"),
    ("cadence_24h", "log1p"), ("cadence_7d", "log1p"), ("hours_since_prev_original", "log1p"),
    ("history_n", "log1p"), ("history_likes_n", "log1p"),
    ("trailing_likes_p50", "log1p"), ("trailing_likes_p90", "log1p"), ("trailing_like_ratio_p50", "log1p"),
    ("trailing_reach_p50", "log1p"), ("trailing_reach_p90", "log1p"), ("trailing_rate_fav_p50", "id"),
    ("trailing_rate_reply_p50", "id"), ("trailing_media_share", "id"), ("trailing_video_share", "id"), ("trailing_link_share", "id"),
    # duplicates, suppression proxies, retrieval summaries (3, 7.5, 7.6)
    ("near_dup_rank", "log1p"), ("near_dup_cluster_size", "log1p"), ("is_self_repeat", "bool"), ("copypasta_events", "log1p"),
    ("url_blocked", "bool"), ("self_sim_48h", "id"), ("nbr_density_7d", "log1p"), ("nbr_best_reach_7d", "id"),
    ("audience_zone_known", "bool"),
    # media and speech scalars (present after the media stages)
    ("frame_motion", "id"), ("speech_ratio", "id"), ("speech_in_first_3s", "bool"), ("transcript_len", "log1p"),
]

CAT_SPEC: list[tuple[str, list[str] | int]] = [
    ("domain_class", list(DOMAIN_CLASS_NAMES)),
    ("media_aspect", list(ASPECT_CLASSES)),
    ("video_duration_bucket", ["none", "le_10s", "10_60s", "gt_60s"]),
    ("kind", ["original", "quote"]),
    ("local_hour_idx", 25),   # Phoenix's hour-of-day cardinality
    ("local_dow", 7),
    ("follower_tier_idx", 6),
]

COUNT_COLS = ("likes", "replies", "reposts", "quotes", "views")  # Phoenix's ENGAGEMENT_COUNT_ORDER
ACTIONS = list(HEAD_INDEX)                                       # rate_fav, rate_reply, ...
ACTION_INDEX = [HEAD_INDEX[a] for a in ACTIONS]
SPLITS = {"train": 0, "val": 1, "test": 2}


def _num_expr(col: str, how: str) -> pl.Expr:
    c = pl.col(col).cast(pl.Float64, strict=False)
    if how == "log1p":
        return c.clip(lower_bound=0).log1p()
    return c


def build(rows: pl.DataFrame, out_dir: str | Path, emb=None, sid: pl.DataFrame | None = None,
          neighbors: pl.DataFrame | None = None, aux: dict | None = None, extra: pl.DataFrame | None = None,
          max_neighbors: int = 16) -> dict:
    """rows: output of the dataset stage. emb: an EmbeddingStore of the post embedding (or None).
    aux: name -> EmbeddingStore keyed by post_id (pooled image or frame embeddings).
    extra: post_id-keyed frame of additional scalar columns (media and speech scalars)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if extra is not None:
        rows = rows.join(extra, on="post_id", how="left")
    if neighbors is not None:
        keep = [c for c in ("post_id", "nbr_post_id", "nbr_cos", "nbr_density_7d", "nbr_best_reach_7d", "self_sim_48h") if c in neighbors.columns]
        rows = rows.join(neighbors.select(keep), on="post_id", how="left")
    rows = rows.sort("author_id", "created_at_ms", "post_id")
    n = rows.height
    split = rows["split"].replace_strict(SPLITS, return_dtype=pl.Int8).to_numpy()
    train = split == 0

    # numeric
    names = [c for c, _ in NUM_SPEC if c in rows.columns]
    X = rows.select([_num_expr(c, h).alias(c) for c, h in NUM_SPEC if c in rows.columns]).to_numpy().astype(np.float32)
    miss = np.isnan(X)
    mean = np.nanmean(np.where(train[:, None], X, np.nan), axis=0)
    std = np.nanstd(np.where(train[:, None], X, np.nan), axis=0)
    mean = np.nan_to_num(mean)
    std = np.where(np.nan_to_num(std) < 1e-6, 1.0, np.nan_to_num(std))
    Xs = np.nan_to_num((X - mean) / std).clip(-8, 8)
    miss_cols = [i for i in range(X.shape[1]) if 0.001 < miss[train, i].mean()]
    num = np.concatenate([Xs, miss[:, miss_cols].astype(np.float32)], axis=1)
    num_names = names + [f"{names[i]}__missing" for i in miss_cols]

    # categorical
    cats, cards, cat_names = [], [], []
    for col, spec in CAT_SPEC:
        if col not in rows.columns:
            continue
        if isinstance(spec, list):
            v = rows[col].cast(pl.String).replace_strict({s: i for i, s in enumerate(spec)}, default=0, return_dtype=pl.Int32).to_numpy()
            card = len(spec)
        else:
            v = rows[col].cast(pl.Int32, strict=False).fill_null(0).clip(0, spec - 1).to_numpy()
            card = spec
        cats.append(v)
        cards.append(card)
        cat_names.append(col)
    cat = np.stack(cats, axis=1).astype(np.int32) if cats else np.zeros((n, 0), dtype=np.int32)

    counts = rows.select([pl.col(c).cast(pl.Int64, strict=False).fill_null(-1) if c in rows.columns else pl.lit(-1, dtype=pl.Int64).alias(c)
                          for c in COUNT_COLS]).to_numpy().astype(np.int64)
    followers = rows["followers_at_post"].cast(pl.Float64, strict=False).fill_null(0).to_numpy()
    likes, views = counts[:, 0].astype(np.float64), counts[:, 4].astype(np.float64)
    y_like = np.where(counts[:, 0] >= 0, np.log10((likes + 1) / (followers + 1)), np.nan).astype(np.float32)
    y_reach = np.where(counts[:, 4] >= 0, np.log10((views + 1) / (followers + 1)), np.nan).astype(np.float32)
    rates = rows.select([pl.col(a).cast(pl.Float32, strict=False) if a in rows.columns else pl.lit(None, dtype=pl.Float32).alias(a)
                         for a in ACTIONS]).to_numpy().astype(np.float32)

    author = rows["author_id"].to_numpy()
    change = np.r_[True, author[1:] != author[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n]
    a_idx = np.cumsum(change) - 1
    arrays = {
        "post_id": rows["post_id"].to_numpy().astype(np.int64),
        "author_idx": a_idx.astype(np.int32),
        "a_start": starts[a_idx].astype(np.int64),
        "a_end": ends[a_idx].astype(np.int64),
        "created_s": (rows["created_at_ms"].to_numpy() // 1000).astype(np.int64),
        "split": split.astype(np.int8),
        "w_row": rows["w_row"].cast(pl.Float32, strict=False).fill_null(1.0).to_numpy() if "w_row" in rows.columns else np.ones(n, np.float32),
        "num": num, "cat": cat, "counts": counts, "followers": followers.astype(np.float32),
        "y_like": y_like, "y_reach": y_reach, "rates": rates,
    }

    ids = arrays["post_id"]
    emb_dim = 0
    if emb is not None:
        probe, _ = emb.get(ids[:1].tolist(), fill_nan=False)
        emb_dim = probe.shape[1]
        mm = np.lib.format.open_memmap(out / "emb.npy", mode="w+", dtype=np.float16, shape=(n, emb_dim))
        has = np.zeros(n, dtype=bool)
        for i in range(0, n, 100_000):
            v, f = emb.get(ids[i:i + 100_000].tolist(), fill_nan=False)
            mm[i:i + len(v)] = v.astype(np.float16)
            has[i:i + len(v)] = f
        mm.flush()
        del mm
        arrays["has_emb"] = has
    aux_dims = {}
    for name, store in (aux or {}).items():
        v, f = store.get(ids.tolist(), fill_nan=False)
        arrays[f"aux_{name}"] = v.astype(np.float16)
        arrays[f"has_aux_{name}"] = f
        aux_dims[name] = int(v.shape[1])
    if sid is not None:
        levels = sorted(c for c in sid.columns if c.startswith("sid_"))
        j = rows.select("post_id").join(sid, on="post_id", how="left")
        arrays["sid"] = j.select([pl.col(c).cast(pl.Int32).fill_null(0) for c in levels]).to_numpy().astype(np.int16)
    if "nbr_post_id" in rows.columns:
        pos = {int(p): i for i, p in enumerate(ids)}
        nbr = np.full((n, max_neighbors), -1, dtype=np.int32)
        cos = np.zeros((n, max_neighbors), dtype=np.float32)
        for i, (pl_ids, pl_cos) in enumerate(zip(rows["nbr_post_id"].to_list(), rows["nbr_cos"].to_list())):
            if not pl_ids:
                continue
            for k, (q, c) in enumerate(zip(pl_ids[:max_neighbors], pl_cos[:max_neighbors])):
                r = pos.get(int(q))
                if r is not None:
                    nbr[i, k], cos[i, k] = r, c
        arrays["nbr_idx"], arrays["nbr_cos"] = nbr, cos

    for k, v in arrays.items():
        np.save(out / f"{k}.npy", v)
    meta = {
        "n": n, "n_authors": int(len(starts)), "num_names": num_names, "num_mean": mean.tolist(), "num_std": std.tolist(),
        "num_base": names, "num_missing_cols": miss_cols, "cat_names": cat_names, "cat_cards": cards, "emb_dim": emb_dim,
        "aux_dims": aux_dims, "sid_levels": int(arrays["sid"].shape[1]) if "sid" in arrays else 0,
        "actions": ACTIONS, "action_index": ACTION_INDEX,
        "splits": {k: int((split == v).sum()) for k, v in SPLITS.items()},
        "has_views": int((counts[:, 4] >= 0).sum()), "has_likes": int((counts[:, 0] >= 0).sum()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("feature table: %d rows, %d authors, %d numeric, %d categorical, emb %d", n, len(starts), num.shape[1], cat.shape[1], emb_dim)
    return meta


@dataclass
class FeatureTable:
    """Arrays as tensors on one device. Embeddings stay fp16 and are cast on gather."""

    meta: dict
    t: dict

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> "FeatureTable":
        import torch

        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        t = {}
        for p in sorted(d.glob("*.npy")):
            t[p.stem] = torch.from_numpy(np.load(p)).to(device)
        return cls(meta=meta, t=t)

    @property
    def n(self) -> int:
        return self.meta["n"]

    @property
    def device(self):
        return self.t["post_id"].device

    def rows_of_split(self, name: str):
        import torch

        return torch.nonzero(self.t["split"] == SPLITS[name]).squeeze(1)
