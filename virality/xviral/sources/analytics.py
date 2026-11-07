"""X analytics content exports (per-post CSV), from your account and donors.

These are the only source of labels for Phoenix's click, profile-click,
link-open, follow and share heads, so every row is kept regardless of other
filters. X has shipped more than one export layout; headers are matched by
normalized synonyms and an unknown layout fails loudly instead of guessing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import polars as pl

from xviral.schema import conform
from xviral.util.core import iso_to_ms, snowflake_ms_expr

log = logging.getLogger(__name__)

SYNONYMS: dict[str, list[str]] = {
    "post_id": ["post id", "tweet id", "post_id", "tweet_id", "id"],
    "link": ["post link", "tweet permalink", "permalink", "link"],
    "text": ["post text", "tweet text", "text"],
    "date": ["date", "time", "created at", "posted at"],
    "views": ["impressions", "views"],
    "likes": ["likes"],
    "engagements": ["engagements"],
    "bookmarks": ["bookmarks"],
    "shares": ["shares"],
    "new_follows": ["new follows", "follows"],
    "replies": ["replies"],
    "reposts": ["reposts", "retweets"],
    "profile_clicks": ["profile visits", "user profile clicks", "profile clicks"],
    "detail_expands": ["detail expands"],
    "link_clicks": ["url clicks", "link clicks"],
}
REQUIRED = ("views",)


def _norm(h: str) -> str:
    return re.sub(r"\s+", " ", h.strip().lower().replace("_", " "))


def map_headers(headers: list[str]) -> dict[str, str]:
    normed = {_norm(h): h for h in headers}
    out: dict[str, str] = {}
    for field, syns in SYNONYMS.items():
        for s in syns:
            if _norm(s) in normed:
                out[field] = normed[_norm(s)]
                break
    missing = [f for f in REQUIRED if f not in out]
    if missing or ("post_id" not in out and "link" not in out):
        raise ValueError(f"analytics export: unrecognized layout, headers={headers}")
    return out


def read_export(path: str | Path, handle: str | None = None, followers: int | None = None) -> pl.DataFrame:
    path = Path(path)
    raw = pl.read_csv(path, infer_schema=False)
    mp = map_headers(raw.columns)
    cols = {f: pl.col(c) for f, c in mp.items()}
    num = lambda f: cols[f].str.replace_all(",", "").cast(pl.Int64, strict=False) if f in cols else pl.lit(None, dtype=pl.Int64)  # noqa: E731
    pid = cols["post_id"].cast(pl.Int64, strict=False) if "post_id" in cols else cols["link"].str.extract(r"/status/(\d+)", 1).cast(pl.Int64)
    df = raw.select(
        pid.alias("post_id"),
        (cols["text"] if "text" in cols else pl.lit(None, dtype=pl.String)).alias("text"),
        (cols["link"].str.extract(r"(?:x|twitter)\.com/([^/]+)/status/", 1) if "link" in cols else pl.lit(handle)).alias("author_handle"),
        (cols["date"].map_elements(iso_to_ms, return_dtype=pl.Int64) if "date" in cols else pl.lit(None, dtype=pl.Int64)).alias("created_at_src_ms"),
        *[num(f).alias(f) for f in ("views", "likes", "engagements", "bookmarks", "shares", "new_follows", "replies",
                                    "reposts", "profile_clicks", "detail_expands", "link_clicks")],
    ).filter(pl.col("post_id").is_not_null())
    if handle:
        df = df.with_columns(pl.col("author_handle").fill_null(handle))
    df = df.with_columns(
        snowflake_ms_expr("post_id").alias("created_at_ms"),
        pl.lit(int(path.stat().st_mtime * 1000)).alias("snapshot_at_ms"),
        pl.lit("file_date").alias("snapshot_kind"),
        pl.lit("count").alias("views_state"),
        pl.lit(followers, dtype=pl.Int64).alias("author_followers"),
        pl.lit("analytics").alias("source"),
        pl.lit(path.name).alias("source_file"),
        pl.lit(False).alias("synthetic_id"),
        # Kind is unknown in the export; replies are detected by a leading mention and
        # corrected later by joining to other sources on post_id.
        pl.when(pl.col("text").fill_null("").str.starts_with("@")).then(pl.lit("reply")).otherwise(pl.lit("original")).alias("kind"),
    )
    return conform(df)
