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
import warnings
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
    ("external_urls_text", "log1p"),   # links counted in the text itself; present even when a source has no URL table
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
    ("url_blocked", "bool"), ("self_sim_48h", "id"), ("nbr_density_7d", "log1p"), ("nbr_best_reach_7d", "id"), ("nbr_best_like_7d", "id"),
    ("audience_zone_known", "bool"),
    # how the post is written: the vetted groups of features/text_structure.py (opening line, layout, link placement)
    ("ts_first_line_chars", "log1p"), ("ts_first_line_words", "log1p"), ("ts_first_line_share", "id"), ("ts_open_question", "bool"),
    ("ts_open_colon", "bool"), ("ts_open_number", "bool"), ("ts_open_first_person", "bool"), ("ts_open_announce", "bool"),
    ("ts_open_mention", "bool"), ("ts_words", "log1p"), ("ts_lines", "log1p"), ("ts_blank_lines", "log1p"), ("ts_bullet_lines", "log1p"),
    ("ts_mean_line_chars", "log1p"), ("ts_max_line_chars", "log1p"), ("ts_sentences", "log1p"), ("ts_words_per_sentence", "log1p"),
    ("ts_url_first_line", "bool"), ("ts_url_at_end", "bool"), ("ts_chars_before_url_share", "id"),
    ("ts_caps_words", "log1p"), ("ts_exclaims", "log1p"), ("ts_questions", "log1p"), ("ts_emoji", "log1p"), ("ts_number_tokens", "log1p"),
    ("ts_money", "bool"), ("ts_percent", "bool"), ("ts_ellipsis", "bool"), ("ts_quotes", "log1p"), ("ts_parens", "log1p"),
    ("ts_arrows", "log1p"), ("ts_code", "bool"),
    ("ts_i", "log1p"), ("ts_we", "log1p"), ("ts_you", "log1p"), ("ts_negation", "log1p"), ("ts_superlative", "log1p"),
    ("ts_comparison", "log1p"), ("ts_how_why", "bool"), ("ts_thread_marker", "bool"), ("ts_call_to_action", "bool"),
    ("ts_len_vs_own", "id"), ("ts_media_vs_own", "id"), ("ts_link_vs_own", "id"), ("ts_gap_vs_own", "id"), ("ts_lines_vs_own", "id"),
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


def _arrays(rows: pl.DataFrame, meta: dict | None = None) -> tuple[dict, dict, pl.DataFrame]:
    """Model-ready arrays for `rows`, sorted by (author, time).

    With `meta` (a trained model's table metadata) the stored feature list, normalisation statistics,
    missing-indicator columns and categorical layout are applied as they are, so new posts are
    transformed exactly like the training data. Without it they are computed from training rows.
    """
    rows = rows.sort("author_id", "created_at_ms", "post_id")
    n = rows.height
    if "split" not in rows.columns:
        rows = rows.with_columns(pl.lit("test").alias("split"))
    split = rows["split"].replace_strict(SPLITS, return_dtype=pl.Int8).to_numpy()
    train = split == 0

    # numeric
    how = dict(NUM_SPEC)
    names = list(meta["num_base"]) if meta else [c for c, _ in NUM_SPEC if c in rows.columns]
    rows = rows.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in names if c not in rows.columns])
    X = rows.select([_num_expr(c, how[c]).alias(c) for c in names]).to_numpy().astype(np.float32)
    miss = np.isnan(X)
    if meta:
        mean, std = np.asarray(meta["num_mean"], np.float32), np.asarray(meta["num_std"], np.float32)
        miss_cols = list(meta["num_missing_cols"])
    else:
        with warnings.catch_warnings():            # columns that are entirely missing give an empty slice
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nan_to_num(np.nanmean(np.where(train[:, None], X, np.nan), axis=0))
            std = np.nan_to_num(np.nanstd(np.where(train[:, None], X, np.nan), axis=0))
        std = np.where(std < 1e-6, 1.0, std)
        miss_cols = [i for i in range(X.shape[1]) if 0.001 < miss[train, i].mean()]
    Xs = np.nan_to_num((X - mean) / std).clip(-8, 8)
    num = np.concatenate([Xs, miss[:, miss_cols].astype(np.float32)], axis=1)
    num_names = names + [f"{names[i]}__missing" for i in miss_cols]

    # categorical
    cats, cards, cat_names = [], [], []
    spec_of = dict(CAT_SPEC)
    for col in (meta["cat_names"] if meta else [c for c, _ in CAT_SPEC if c in rows.columns]):
        spec = spec_of[col]
        if col not in rows.columns:
            rows = rows.with_columns(pl.lit(None, dtype=pl.String if isinstance(spec, list) else pl.Int32).alias(col))
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

    # Absent columns are added to the frame first: a select of literals alone would collapse to one row.
    rows = rows.with_columns([pl.lit(None, dtype=pl.Int64).alias(c) for c in COUNT_COLS if c not in rows.columns]
                             + [pl.lit(None, dtype=pl.Float32).alias(a) for a in ACTIONS if a not in rows.columns])
    counts = rows.select([pl.col(c).cast(pl.Int64, strict=False).fill_null(-1) for c in COUNT_COLS]).to_numpy().astype(np.int64)
    # Follower count 0 means unknown (some sources carry none); follower-normalised targets are then undefined.
    followers = rows["followers_at_post"].cast(pl.Float64, strict=False).fill_null(0).clip(lower_bound=0).to_numpy()
    known = followers > 0
    likes, views = counts[:, 0].astype(np.float64), counts[:, 4].astype(np.float64)
    y_likes = np.where(counts[:, 0] >= 0, np.log10(np.maximum(likes, 0) + 1), np.nan).astype(np.float32)
    y_views = np.where(counts[:, 4] >= 0, np.log10(np.maximum(views, 0) + 1), np.nan).astype(np.float32)
    with np.errstate(divide="ignore"):             # absent counts are -1 and masked by the where
        y_like = np.where((counts[:, 0] >= 0) & known, np.log10((likes + 1) / (followers + 1)), np.nan).astype(np.float32)
        y_reach = np.where((counts[:, 4] >= 0) & known, np.log10((views + 1) / (followers + 1)), np.nan).astype(np.float32)
    rates = rows.select([pl.col(a).cast(pl.Float32, strict=False) for a in ACTIONS]).to_numpy().astype(np.float32)
    assert counts.shape[0] == n and rates.shape[0] == n

    # Post format code for the account-history profile: media | link | video.
    flag = lambda c: (rows[c].cast(pl.Int8, strict=False).fill_null(0).to_numpy() if c in rows.columns else np.zeros(n, np.int8))  # noqa: E731
    has_link = np.maximum(flag("has_url"), (rows["external_urls_text"].fill_null(0).to_numpy() > 0).astype(np.int8)
                          if "external_urls_text" in rows.columns else 0)
    fmt = (flag("has_media") + 2 * has_link + 4 * flag("has_video")).astype(np.int8)

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
        "audience": (rows["in_audience"].fill_null(True).to_numpy() if "in_audience" in rows.columns else np.ones(n, bool)),
        "num": num, "cat": cat, "fmt": fmt, "counts": counts, "followers": followers.astype(np.float32),
        "y_like": y_like, "y_reach": y_reach, "y_likes": y_likes, "y_views": y_views, "rates": rates,
    }

    stats = {"num_names": num_names, "num_mean": np.asarray(mean).tolist(), "num_std": np.asarray(std).tolist(), "num_base": names,
             "num_missing_cols": miss_cols, "cat_names": cat_names, "cat_cards": cards, "n_authors": int(len(starts))}
    return arrays, stats, rows


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
        keep = [c for c in ("post_id", "nbr_post_id", "nbr_cos", "nbr_density_7d", "nbr_best_reach_7d", "nbr_best_like_7d", "self_sim_48h") if c in neighbors.columns]
        rows = rows.join(neighbors.select(keep), on="post_id", how="left")
    arrays, stats, rows = _arrays(rows)
    n = rows.height
    split, counts = arrays["split"], arrays["counts"]

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

    if "text" in rows.columns:
        # Rendered post text in table order, for stage B's live encoder.
        pl.DataFrame({"post_id": ids, "text": render_texts(rows)}).write_parquet(out / "texts.parquet", compression="zstd")

    for k, v in arrays.items():
        np.save(out / f"{k}.npy", v)
    meta = {
        "n": n, **stats, "emb_dim": emb_dim,
        "aux_dims": aux_dims, "sid_levels": int(arrays["sid"].shape[1]) if "sid" in arrays else 0,
        "actions": ACTIONS, "action_index": ACTION_INDEX,
        "labeled_actions": [idx for idx, ok in zip(ACTION_INDEX, (~np.isnan(arrays["rates"])).any(0)) if ok],
        "splits": {k: int((split == v).sum()) for k, v in SPLITS.items()},
        "has_views": int((counts[:, 4] >= 0).sum()), "has_likes": int((counts[:, 0] >= 0).sum()),
        "audience_rows": int(arrays["audience"].sum()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("feature table: %d rows, %d authors, %d numeric, %d categorical, emb %d", n, stats["n_authors"],
             arrays["num"].shape[1], arrays["cat"].shape[1], emb_dim)
    return meta


def render_texts(rows: pl.DataFrame) -> list[str]:
    """Each post as phoenix/reference/mm_encoder.py renders a text post: author line, text, t.co links removed,
    and the quoted post's text when the row carries it."""
    from xviral.embed import mm

    cols = [c for c in ("post_id", "text", "text_note", "author_handle", "author_display_name", "quoted_text") if c in rows.columns]
    return [mm.render(mm.build_post(r, quoted={"text": r.get("quoted_text")}))[0] for r in rows.select(cols).iter_rows(named=True)]


def transform(rows: pl.DataFrame, meta: dict, emb: np.ndarray | None = None, sid: np.ndarray | None = None,
              device: str = "cpu") -> tuple["FeatureTable", pl.DataFrame]:
    """An in-memory table for new posts, using a trained model's stored transforms.

    `emb` and `sid` are aligned with `rows` as passed in. Returns the table and the rows in table order.
    """
    import torch

    rows = rows.with_row_index("_in")
    arrays, _, srt = _arrays(rows, meta)
    order = srt["_in"].to_numpy()
    n = srt.height
    if meta["emb_dim"]:
        e = np.zeros((n, meta["emb_dim"]), np.float16) if emb is None else np.asarray(emb)[order].astype(np.float16)
        arrays["emb"], arrays["has_emb"] = e, np.full(n, emb is not None)
    if meta["sid_levels"]:
        arrays["sid"] = (np.zeros((n, meta["sid_levels"]), np.int16) if sid is None else np.asarray(sid)[order].astype(np.int16))
    t = {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}
    return FeatureTable(meta={**meta, "n": n}, t=t, texts=render_texts(srt) if "text" in srt.columns else None), srt.drop("_in")


@dataclass
class FeatureTable:
    """Arrays as tensors on one device. Embeddings stay fp16 and are cast on gather."""

    meta: dict
    t: dict
    texts: list | None = None

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> "FeatureTable":
        import torch

        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        t = {}
        for p in sorted(d.glob("*.npy")):
            t[p.stem] = torch.from_numpy(np.load(p)).to(device)
        texts = pl.read_parquet(d / "texts.parquet")["text"].to_list() if (d / "texts.parquet").exists() else None
        return cls(meta=meta, t=t, texts=texts)

    @property
    def n(self) -> int:
        return self.meta["n"]

    @property
    def device(self):
        return self.t["post_id"].device

    def rows_of_split(self, name: str, audience_only: bool = False):
        """Row indices of a split; with audience_only, only posts by audience accounts (see filters.select_accounts)."""
        import torch

        keep = self.t["split"] == SPLITS[name]
        if audience_only and "audience" in self.t:
            keep = keep & self.t["audience"]
        return torch.nonzero(keep).squeeze(1)
