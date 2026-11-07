"""Phase-0 reporting: the survivor table the go/no-go gate reads (DATA_PIPELINE.md 4.8, 11)."""

from __future__ import annotations

import polars as pl


def count_table(df: pl.DataFrame) -> pl.DataFrame:
    """Counts by source x year x kind x media x follower tier x views availability."""
    media = (
        pl.when(pl.col("media").list.eval(pl.element().struct.field("type") == "video").list.any()).then(pl.lit("video"))
        .when(pl.col("media").list.len() > 0).then(pl.lit("image_or_gif"))
        .otherwise(pl.lit("none"))
    )
    return (
        df.with_columns(
            pl.from_epoch("created_at_ms", time_unit="ms").dt.year().alias("year"),
            media.alias("media_kind"),
            pl.col("views").is_not_null().alias("has_views"),
        )
        .group_by("source", "year", "kind", "media_kind", "follower_tier", "has_views")
        .agg(pl.len().alias("rows"), pl.col("author_id").n_unique().alias("authors"))
        .sort("source", "year", "kind", "media_kind", "follower_tier", "has_views")
    )


def headline(df: pl.DataFrame) -> dict:
    v = df.filter(pl.col("views").is_not_null())
    return {
        "rows": df.height,
        "authors": df["author_id"].n_unique(),
        "rows_with_views": v.height,
        "authors_with_views": v["author_id"].n_unique() if v.height else 0,
        "video_rows": df.filter(pl.col("media").list.eval(pl.element().struct.field("type") == "video").list.any()).height,
        "by_source": dict(df.group_by("source").len().iter_rows()),
    }
