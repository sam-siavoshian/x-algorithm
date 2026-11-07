"""Author features: the stand-in for Phoenix's learned author embedding and the
viewer-author follow relationship (DATA_PIPELINE.md section 7.3).

Leakage rule, enforced by construction: every trailing statistic is computed
as the author's state *after* each post, then attached to each target post
with an as-of join that excludes exact matches. A post therefore only sees
posts created strictly before it.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from xviral.constants import COLD_START_FOLLOWER_CAP, FOLLOWER_TIER_EDGES, FOLLOWER_TIER_NAMES

log = logging.getLogger(__name__)

DAY_MS = 86_400_000


def follower_tier_expr(col: str = "followers_at_post") -> pl.Expr:
    c = pl.col(col)
    e = pl.when(c < FOLLOWER_TIER_EDGES[0]).then(pl.lit(FOLLOWER_TIER_NAMES[0]))
    for edge, name in zip(FOLLOWER_TIER_EDGES[1:], FOLLOWER_TIER_NAMES[1:-1]):
        e = e.when(c < edge).then(pl.lit(name))
    return e.otherwise(pl.lit(FOLLOWER_TIER_NAMES[-1]))


def followers_at_post(posts: pl.DataFrame, accounts: pl.DataFrame) -> pl.DataFrame:
    """Linear interpolation of follower count at post time between the nearest snapshots."""
    snaps = (
        accounts.select("author_id", pl.col("snapshot_at_ms").alias("t"), pl.col("followers").cast(pl.Float64).alias("f"))
        .filter(pl.col("t").is_not_null() & pl.col("f").is_not_null())
        .group_by("author_id", "t").agg(pl.col("f").max())
        .sort("t")
    )
    left = posts.select("post_id", "author_id", pl.col("created_at_ms").alias("t")).sort("t")
    back = left.join_asof(snaps.rename({"f": "f0"}).with_columns(pl.col("t").alias("t0")), on="t", by="author_id", strategy="backward", check_sortedness=False)
    fwd = left.join_asof(snaps.rename({"f": "f1"}).with_columns(pl.col("t").alias("t1")), on="t", by="author_id", strategy="forward", check_sortedness=False)
    j = back.join(fwd.select("post_id", "f1", "t1"), on="post_id", how="left")
    w = ((pl.col("t") - pl.col("t0")) / (pl.col("t1") - pl.col("t0"))).clip(0, 1)
    interp = (
        pl.when(pl.col("f0").is_not_null() & pl.col("f1").is_not_null() & (pl.col("t1") > pl.col("t0")))
        .then(pl.col("f0") + w * (pl.col("f1") - pl.col("f0")))
        .otherwise(pl.coalesce("f0", "f1"))
    )
    j = j.with_columns(interp.round(0).cast(pl.Int64).alias("followers_at_post"))
    out = posts.join(j.select("post_id", "followers_at_post"), on="post_id", how="left")
    return out.with_columns(pl.coalesce("followers_at_post", "author_followers").alias("followers_at_post"))


def _post_quantities(posts: pl.DataFrame) -> pl.DataFrame:
    v = pl.col("views").cast(pl.Float64)
    ext = pl.col("urls").list.eval(
        (~pl.element().struct.field("is_media")) & (pl.element().struct.field("final_domain").fill_null("") != "x.com")
    ).list.any()
    return posts.select(
        "post_id", "author_id", "created_at_ms", "views",
        (pl.col("likes") / v).alias("q_fav"),
        (pl.col("replies") / v).alias("q_reply"),
        (pl.col("reposts") / v).alias("q_repost"),
        (pl.col("quotes") / v).alias("q_quote"),
        (v / pl.col("followers_at_post").cast(pl.Float64).clip(lower_bound=1)).alias("q_reach"),
        (pl.col("replies") / pl.col("likes").clip(lower_bound=1)).alias("q_reply_to_fav"),
        (pl.col("media").list.len() > 0).cast(pl.Float64).alias("q_media"),
        pl.col("media").list.eval(pl.element().struct.field("type") == "video").list.any().cast(pl.Float64).alias("q_video"),
        ext.fill_null(False).cast(pl.Float64).alias("q_link"),
    )


def trailing_features(posts: pl.DataFrame, window: int = 50) -> pl.DataFrame:
    """Per-post author statistics over the author's previous `window` originals, strictly earlier."""
    q = _post_quantities(posts).sort("author_id", "created_at_ms")
    viewed = q.filter(pl.col("views").is_not_null() & (pl.col("views") > 0))

    def rq(col: str, p: float) -> pl.Expr:
        return pl.col(col).rolling_quantile(p, window_size=window, min_samples=1).over("author_id", order_by="created_at_ms")

    state_viewed = viewed.select(
        "author_id", "created_at_ms",
        *[rq(f"q_{k}", 0.5).alias(f"trailing_rate_{k}_p50") for k in ("fav", "reply", "repost", "quote")],
        *[rq(f"q_{k}", 0.9).alias(f"trailing_rate_{k}_p90") for k in ("fav", "reply", "repost", "quote")],
        rq("q_reach", 0.5).alias("trailing_reach_p50"),
        rq("q_reach", 0.9).alias("trailing_reach_p90"),
        rq("q_reach", 0.99).alias("trailing_reach_p99"),
        rq("q_reply_to_fav", 0.5).alias("trailing_reply_to_fav"),
        pl.col("views").cast(pl.Float64).rolling_quantile(0.9, window_size=window, min_samples=1).over("author_id", order_by="created_at_ms").alias("trailing_views_p90"),
        pl.col("views").cast(pl.Float64).rolling_quantile(0.99, window_size=window, min_samples=1).over("author_id", order_by="created_at_ms").alias("trailing_views_p99"),
        pl.int_range(1, pl.len() + 1).over("author_id", order_by="created_at_ms").clip(upper_bound=window).alias("history_n"),
    ).group_by("author_id", "created_at_ms").agg(pl.all().last()).sort("created_at_ms")

    state_all = q.select(
        "author_id", "created_at_ms",
        *[pl.col(f"q_{k}").rolling_mean(window_size=window, min_samples=1).over("author_id", order_by="created_at_ms").alias(f"trailing_{k}_share")
          for k in ("media", "video", "link")],
    ).group_by("author_id", "created_at_ms").agg(pl.all().last()).sort("created_at_ms")

    target = q.select("post_id", "author_id", "created_at_ms").sort("created_at_ms")
    out = target.join_asof(state_viewed, on="created_at_ms", by="author_id", strategy="backward", allow_exact_matches=False, check_sortedness=False)
    out = out.join_asof(state_all, on="created_at_ms", by="author_id", strategy="backward", allow_exact_matches=False, check_sortedness=False)
    out = out.with_columns(pl.col("history_n").fill_null(0))

    # Cadence: originals in the prior 24 h and 7 d, and the gap to the previous original.
    t = posts.select("post_id", "author_id", pl.col("created_at_ms").cast(pl.Datetime("ms")).alias("_t")).sort("author_id", "_t")
    # Rolling output rows align positionally with the sorted input; with closed='left' the window
    # excludes the current post, so its own post_id is not in the window and must not be used as a key.
    c24 = t.rolling(index_column="_t", period="24h", group_by="author_id", closed="left").agg(pl.len().alias("cadence_24h"))
    c7 = t.rolling(index_column="_t", period="7d", group_by="author_id", closed="left").agg(pl.len().alias("cadence_7d"))
    assert c24["_t"].equals(t["_t"]) and c7["_t"].equals(t["_t"])
    cad = t.select("post_id", "_t", "author_id").with_columns(c24["cadence_24h"], c7["cadence_7d"])
    cad = cad.with_columns(((pl.col("_t") - pl.col("_t").shift(1).over("author_id")).dt.total_milliseconds() / 3_600_000).alias("hours_since_prev_original"))
    out = out.join(cad.select("post_id", "cadence_24h", "cadence_7d", "hours_since_prev_original"), on="post_id", how="left")
    return out.drop("author_id", "created_at_ms")


def author_table(posts: pl.DataFrame, accounts: pl.DataFrame, activity: pl.DataFrame | None,
                 copypasta: pl.DataFrame | None, edges: pl.DataFrame | None, donors: pl.DataFrame | None) -> pl.DataFrame:
    """One row per author: descriptors, hygiene signals, mutual-follow share."""
    latest = (
        accounts.sort("snapshot_at_ms", nulls_last=False)
        .group_by("author_id")
        .agg(
            pl.col("followers").drop_nulls().last().alias("followers_latest"),
            pl.col("following").drop_nulls().last().alias("following_latest"),
            pl.col("created_at_ms").drop_nulls().first().alias("account_created_ms"),
            pl.col("verified").drop_nulls().last().alias("verified"),
            pl.col("author_handle").drop_nulls().last().alias("handle"),
        )
    )
    ts = posts.select("author_id", pl.col("created_at_ms").cast(pl.Datetime("ms")).alias("_t")).sort("author_id", "_t")
    burst = ts.rolling(index_column="_t", period="24h", group_by="author_id").agg(pl.len().alias("_n")).group_by("author_id").agg(pl.col("_n").max().alias("max_originals_24h"))
    ext_domains = (
        posts.select("author_id", "post_id", pl.col("urls"))
        .explode("urls")
        .filter(pl.col("urls").is_not_null())
        .with_columns(pl.col("urls").struct.field("final_domain").alias("d"), pl.col("urls").struct.field("is_media").alias("m"))
        .filter(~pl.col("m").fill_null(False) & pl.col("d").is_not_null() & (pl.col("d") != "x.com"))
        .unique(["author_id", "post_id", "d"])
        .group_by("author_id", "d").len()
        .group_by("author_id").agg(pl.col("len").max().alias("_top"))
    )
    n_orig = posts.group_by("author_id").len().rename({"len": "originals_n"})
    a = (
        latest.join(burst, on="author_id", how="left")
        .join(n_orig, on="author_id", how="left")
        .join(ext_domains, on="author_id", how="left")
        .with_columns(
            (pl.col("_top").fill_null(0) / pl.col("originals_n").clip(lower_bound=1)).alias("top_domain_share"),
            (pl.col("followers_latest").cast(pl.Float64) / pl.col("following_latest").cast(pl.Float64).clip(lower_bound=1)).alias("follow_ratio"),
            (pl.col("followers_latest") <= COLD_START_FOLLOWER_CAP).alias("cold_start_eligible_author"),
        )
        .drop("_top")
    )
    if copypasta is not None:
        a = a.join(copypasta.group_by("author_id").agg(pl.col("is_self_repeat").sum().alias("copypasta_events")), on="author_id", how="left")
    if activity is not None:
        a = a.join(
            activity.select("author_id", (pl.col("replies") / pl.col("rows").clip(lower_bound=1)).alias("reply_share"),
                            (pl.col("replies") / pl.col("originals").clip(lower_bound=1)).alias("replies_per_original")),
            on="author_id", how="left",
        )
    a = a.with_columns(pl.col("copypasta_events").fill_null(0) if "copypasta_events" in a.columns else pl.lit(0).alias("copypasta_events"))
    if edges is not None and donors is not None and donors.height:
        a = a.join(mutual_share_exact(edges, donors), on="author_id", how="left")
    else:
        a = a.with_columns(pl.lit(None, dtype=pl.Float64).alias("mutual_share_exact"))
    return fit_mutual_share(a)


def mutual_share_exact(edges: pl.DataFrame, donors: pl.DataFrame) -> pl.DataFrame:
    """Fraction of a donor's followers the donor follows back. Exact, from their own archive.

    This matters because the reply weight is 5 for non-mutuals and 5 + 15 = 20 for mutuals
    (home-mixer/params/param.rs:410-416; xai-value-model weights.rs:86-93).
    """
    d = donors.select(pl.col("author_id"))
    inc = edges.join(d, left_on="dst_author_id", right_on="author_id", how="semi").select(pl.col("dst_author_id").alias("author_id"), pl.col("src_author_id").alias("x"))
    out = edges.join(d, left_on="src_author_id", right_on="author_id", how="semi").select(pl.col("src_author_id").alias("author_id"), pl.col("dst_author_id").alias("x"))
    mutual = inc.join(out, on=["author_id", "x"], how="semi").group_by("author_id").len().rename({"len": "_m"})
    tot = inc.group_by("author_id").len().rename({"len": "_f"})
    return tot.join(mutual, on="author_id", how="left").select(
        "author_id", (pl.col("_m").fill_null(0) / pl.col("_f").clip(lower_bound=1)).alias("mutual_share_exact")
    )


def fit_mutual_share(a: pl.DataFrame, min_donors: int = 30) -> pl.DataFrame:
    """Estimate mutual share for every author from donors whose share is known exactly.

    Mutuals cannot exceed min(followers, following), so the model predicts the fraction of that
    bound which is realised, from log followers, log following and account age. With too few
    donors the estimate is left null rather than invented.
    """
    f = a["followers_latest"].fill_null(0).cast(pl.Float64).to_numpy()
    g = a["following_latest"].fill_null(0).cast(pl.Float64).to_numpy()
    bound = np.where(f > 0, np.minimum(f, g) / np.maximum(f, 1), 0.0)
    known = a["mutual_share_exact"].to_numpy(allow_copy=True) if "mutual_share_exact" in a.columns else np.full(a.height, np.nan)
    known = np.asarray(known, dtype=np.float64)
    mask = ~np.isnan(known) & (bound > 0)
    est = np.full(a.height, np.nan)
    if mask.sum() >= min_donors:
        from sklearn.linear_model import Ridge

        X = np.column_stack([np.log1p(f), np.log1p(g), np.log1p(f) - np.log1p(g)])
        y = np.clip(known[mask] / bound[mask], 1e-3, 1 - 1e-3)
        y = np.log(y / (1 - y))
        model = Ridge(alpha=1.0).fit(X[mask], y)
        frac = 1 / (1 + np.exp(-model.predict(X)))
        est = frac * bound
        log.info("mutual share: fit on %d donors", int(mask.sum()))
    else:
        log.warning("mutual share: only %d donors with exact share; estimate left null", int(mask.sum()))
    est = np.where(~np.isnan(known), known, est)
    return a.with_columns(pl.Series("mutual_share_est", est, dtype=pl.Float64).fill_nan(None))
