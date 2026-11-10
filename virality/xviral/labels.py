"""Labels (DATA_PIPELINE.md section 6). Everything is derived; nothing is copied
from a source's own notion of 'viral'.

Phoenix predicts per-viewer action probabilities; public counts are the
audience average of those, so per-impression rates are the observable target.
Reach is normalised by followers, the scale X's own exploration label uses.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import polars as pl

from xviral.config import LabelCfg
from xviral.constants import EXPLORATION_REACH_TARGET_FRACTION

log = logging.getLogger(__name__)

# Phoenix ActionName indices (phoenix/crates/serving/xai-recsys-proto/proto/recsys.proto:456-697),
# used so the surrogate's heads line up with the ported value model.
HEAD_INDEX = {
    "rate_fav": 1, "rate_reply": 4, "rate_quote": 5, "rate_repost": 6, "rate_bookmark": 36,
    "rate_follow": 21, "rate_click": 29, "rate_profile_click": 30, "rate_share": 37, "rate_link_open": 39,
}


def recency_weight_expr(cfg: LabelCfg) -> pl.Expr:
    year = pl.from_epoch("created_at_ms", time_unit="ms").dt.year()
    keys = sorted(cfg.recency_weights)
    e = pl.when(year <= keys[0]).then(pl.lit(cfg.recency_weights[keys[0]]))
    for k in keys[1:]:
        e = e.when(year <= k).then(pl.lit(cfg.recency_weights[k]))
    return e.otherwise(pl.lit(cfg.recency_weights[keys[-1]]))


def rates(df: pl.DataFrame) -> pl.DataFrame:
    v = pl.col("views").cast(pl.Float64)
    ok = pl.col("views").is_not_null() & (pl.col("views") > 0)

    def r(num: str) -> pl.Expr:
        return pl.when(ok & pl.col(num).is_not_null()).then((pl.col(num).cast(pl.Float64) / v).clip(0.0, 1.0)).otherwise(None)

    return df.with_columns(
        r("likes").alias("rate_fav"),
        r("replies").alias("rate_reply"),
        r("reposts").alias("rate_repost"),
        r("quotes").alias("rate_quote"),
        r("bookmarks").alias("rate_bookmark"),
        r("detail_expands").alias("rate_click"),
        r("profile_clicks").alias("rate_profile_click"),
        r("link_clicks").alias("rate_link_open"),
        r("new_follows").alias("rate_follow"),
        r("shares").alias("rate_share"),
    )


def reach(df: pl.DataFrame) -> pl.DataFrame:
    f = pl.col("followers_at_post").cast(pl.Float64).clip(lower_bound=1.0)
    rr = pl.col("views").cast(pl.Float64) / f
    lr = pl.col("likes").cast(pl.Float64) / f
    own_like = lambda c: (  # noqa: E731
        pl.when(pl.col(c).is_not_null() & pl.col("likes").is_not_null()).then(pl.col("likes") >= pl.col(c)).otherwise(None)
        if c in df.columns else pl.lit(None, dtype=pl.Boolean))
    return df.with_columns(
        rr.alias("reach_ratio"),
        rr.log1p().alias("log_reach_ratio"),
        lr.alias("like_ratio"),
        lr.log1p().alias("log_like_ratio"),
        own_like("trailing_likes_p90").alias("y_likes_beat_own_p90"),
        (rr >= 1).alias("y_reach_1x"),
        (rr >= 10).alias("y_reach_10x"),
        (rr >= 100).alias("y_reach_100x"),
        # Author-relative tiers use only strictly earlier posts (trailing_* from xviral.authors).
        pl.when(pl.col("trailing_views_p90").is_not_null()).then(pl.col("views") >= pl.col("trailing_views_p90")).otherwise(None).alias("y_beat_own_p90"),
        pl.when(pl.col("trailing_views_p99").is_not_null()).then(pl.col("views") >= pl.col("trailing_views_p99")).otherwise(None).alias("y_beat_own_p99"),
        # X's exploration label at the 24 h point reduces to views < 3% of followers
        # (recsys_batch.py:105-135). Computed on final views; originals only.
        pl.when(pl.col("views").is_not_null() & (pl.col("kind") == "original"))
        .then(pl.col("views").cast(pl.Float64) < EXPLORATION_REACH_TARGET_FRACTION * pl.col("followers_at_post").cast(pl.Float64))
        .otherwise(None).alias("y_post_unexplored"),
    ).with_columns(
        # Views can be absent; keep tier labels null rather than false in that case.
        *[pl.when(pl.col("views").is_null()).then(None).otherwise(pl.col(c)).alias(c) for c in ("y_reach_1x", "y_reach_10x", "y_reach_100x")]
    )


def row_weights(df: pl.DataFrame, cfg: LabelCfg) -> pl.DataFrame:
    full = math.log1p(cfg.full_weight_views)
    w_views = (pl.col("views").fill_null(0).cast(pl.Float64).log1p() / full).clip(upper_bound=1.0)
    w = recency_weight_expr(cfg) * w_views
    w = pl.when(pl.col("source") == "analytics").then(w * cfg.analytics_weight_multiplier).otherwise(w)
    return df.with_columns(w.alias("w_row"))


def build(df: pl.DataFrame, cfg: LabelCfg) -> pl.DataFrame:
    return row_weights(reach(rates(df)), cfg)


def pairs(df: pl.DataFrame, cfg: LabelCfg, seed: int = 0) -> pl.DataFrame:
    """Same-author pairs within `pair_window_days`, the confound control for content effects.

    Comparison quantity: log reach ratio where views exist, else log1p(likes / followers).
    Pairs whose difference is under the tie margin are dropped. At most
    `max_pairs_per_author` pairs per author, sampled uniformly.
    """
    q = df.select(
        "post_id", "author_id", "created_at_ms", "split" if "split" in df.columns else pl.lit("train").alias("split"),
        pl.when(pl.col("views").is_not_null() & (pl.col("views") > 0)).then(pl.col("log_reach_ratio"))
        .otherwise((pl.col("likes").cast(pl.Float64) / pl.col("followers_at_post").cast(pl.Float64).clip(lower_bound=1)).log1p()).alias("_y"),
        pl.col("views").is_not_null().alias("_has_views"),
    ).filter(pl.col("_y").is_not_null())
    win = int(cfg.pair_window_days * 86_400_000)
    # Bucket by window-length periods: two posts within `win` are in the same or adjacent bucket,
    # so two equi-joins replace an all-pairs self-join per author.
    q = q.with_columns((pl.col("created_at_ms") // win).alias("_wk"))
    a = q.rename({c: f"{c}_a" for c in q.columns if c not in ("author_id",)})
    b = q.rename({c: f"{c}_b" for c in q.columns if c not in ("author_id",)})
    same = a.join(b, left_on=["author_id", "_wk_a"], right_on=["author_id", "_wk_b"], how="inner")
    nxt = a.with_columns((pl.col("_wk_a") + 1).alias("_wk_next")).join(
        b, left_on=["author_id", "_wk_next"], right_on=["author_id", "_wk_b"], how="inner").drop("_wk_next")
    cols = sorted(c for c in same.columns if not c.startswith("_wk"))
    cand = pl.concat([same.select(cols), nxt.select(cols)], how="vertical_relaxed")
    # Orient every pair as (smaller post_id, larger post_id), then drop duplicate orientations.
    swap = {c: (c[:-2] + "_b" if c.endswith("_a") else c[:-2] + "_a" if c.endswith("_b") else c) for c in cols}
    fwd = cand.filter(pl.col("post_id_a") < pl.col("post_id_b"))
    rev = cand.filter(pl.col("post_id_a") > pl.col("post_id_b")).rename(swap).select(cols)
    p = pl.concat([fwd, rev], how="vertical_relaxed").unique(["post_id_a", "post_id_b"]).filter(
        ((pl.col("created_at_ms_a") - pl.col("created_at_ms_b")).abs() <= win)
        & (pl.col("split_a") == pl.col("split_b"))
        & (pl.col("_has_views_a") == pl.col("_has_views_b"))
        & ((pl.col("_y_a") - pl.col("_y_b")).abs() >= cfg.pair_tie_margin)
    )
    p = p.with_columns(pl.int_range(pl.len()).shuffle(seed=seed).over("author_id").alias("_r")).filter(pl.col("_r") < cfg.max_pairs_per_author)
    out = p.select(
        "author_id", pl.col("post_id_a"), pl.col("post_id_b"),
        (pl.col("_y_a") > pl.col("_y_b")).cast(pl.Int8).alias("a_wins"),
        (pl.col("_y_a") - pl.col("_y_b")).alias("margin"),
        pl.col("_has_views_a").alias("has_views"),
        pl.col("split_a").alias("split"),
    )
    log.info("pairs: %d pairs over %d authors", out.height, out["author_id"].n_unique() if out.height else 0)
    return out


def label_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Base rates per follower tier, for the phase-0 report."""
    return (
        df.filter(pl.col("views").is_not_null())
        .group_by("follower_tier")
        .agg(
            pl.len().alias("n"),
            pl.col("log_reach_ratio").median().alias("median_log_reach"),
            pl.col("y_reach_1x").mean().alias("p_1x"),
            pl.col("y_reach_10x").mean().alias("p_10x"),
            pl.col("y_reach_100x").mean().alias("p_100x"),
            pl.col("y_post_unexplored").mean().alias("p_unexplored"),
            pl.col("rate_fav").median().alias("median_rate_fav"),
        )
        .sort("follower_tier")
    )
