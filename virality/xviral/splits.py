"""Train / validation / test splits (DATA_PIPELINE.md section 9).

test        the newest `test_time_fraction` of rows by creation time, all authors.
            This is the number reported to anyone.
validation  a random `val_author_fraction` of authors among the remaining rows,
            held out entirely: model selection and calibration.
train       the rest.
Analytics rows are split by account, never by row, so a connected account's
posts all land on one side of the train/validation boundary.
"""

from __future__ import annotations

import polars as pl

from xviral.config import SplitCfg


def assign(df: pl.DataFrame, cfg: SplitCfg) -> pl.DataFrame:
    cut = df["created_at_ms"].quantile(1.0 - cfg.test_time_fraction, interpolation="lower")
    authors = df.filter(pl.col("created_at_ms") < cut).select("author_id").unique().sort("author_id")
    val = authors.sample(fraction=cfg.val_author_fraction, seed=cfg.seed, shuffle=True)
    return df.with_columns(
        pl.when(pl.col("created_at_ms") >= cut).then(pl.lit("test"))
        .when(pl.col("author_id").is_in(val["author_id"].implode())).then(pl.lit("val"))
        .otherwise(pl.lit("train")).alias("split")
    )


def summary(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by("split").agg(
        pl.len().alias("rows"),
        pl.col("author_id").n_unique().alias("authors"),
        pl.col("views").is_not_null().sum().alias("rows_with_views"),
        pl.from_epoch("created_at_ms", time_unit="ms").min().alias("first"),
        pl.from_epoch("created_at_ms", time_unit="ms").max().alias("last"),
    ).sort("split")
