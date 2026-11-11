"""Row-level columns derived from the author snapshot; shared by the feature stage and serving."""

from __future__ import annotations

import polars as pl

from xviral.authors import follower_tier_expr
from xviral.constants import COLD_START_FOLLOWER_CAP, FOLLOWER_TIER_NAMES


def author_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        follower_tier_expr().alias("follower_tier"),
        pl.col("followers_at_post").cast(pl.Float64).log1p().alias("log_followers_at_post"),
        pl.col("author_following").cast(pl.Float64).log1p().alias("log_following"),
        (pl.col("followers_at_post").cast(pl.Float64) / pl.col("author_following").cast(pl.Float64).clip(lower_bound=1)).alias("follow_ratio"),
        ((pl.col("created_at_ms") - pl.col("author_created_at_ms")) / 86_400_000).alias("account_age_days"),
        (pl.col("followers_at_post") <= COLD_START_FOLLOWER_CAP).alias("cold_start_eligible_author"),
    ).with_columns(
        pl.col("follower_tier").replace_strict({n: i for i, n in enumerate(FOLLOWER_TIER_NAMES)}, return_dtype=pl.Int32).alias("follower_tier_idx"),
    )
