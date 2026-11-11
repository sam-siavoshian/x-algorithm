"""SinclairSchneider/tweets_sample_2026 (Hugging Face): 75.8M X posts, January to July 2026, with views.

Collected through Nitter search over a newspaper vocabulary, so it is a general sample: about 1.6%
tech by keyword. 73 Parquet shards, 25.7 GB. No license is stated on the card.

Format facts, checked on real shards (research/08_public_datasets.md and our own check):
- **The engagement columns are shifted.** `comments` = replies, `retweets` = reposts,
  `quotes` = LIKES, `likes` = VIEWS. Evidence: `quotes` <= `likes` in 99.99% of original posts
  and their median ratio is 1.3%, the usual likes-per-view rate. There is no quote or bookmark count.
- No follower count, no video information (`videos` is always empty), no scrape time.
- Counts are frozen at an unrecorded scrape time. Median views fall steeply for posts from May
  onward, consistent with later months being captured younger. Posts before `mature_before` are
  treated as mature; later posts get no snapshot time and the maturity filter drops them.
- `id` is the post ID as a string; `profile_id` the author ID; `username` carries a leading '@'.

Ingest is two passes so author timelines survive: pass 1 finds authors who post about tech;
pass 2 keeps every original post by those authors plus every keyword-matching post.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from xviral.schema import MEDIA_STRUCT, URL_STRUCT, conform
from xviral.text.tech import is_tech_post
from xviral.text.urls import registrable_domain
from xviral.util.core import date_to_ms, snowflake_ms_expr, write_parquet

log = logging.getLogger(__name__)

SOURCE = "nitter_2026"


def _roots(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(~pl.col("is-retweet").fill_null(False) & (pl.col("replying-to").list.len().fill_null(0) == 0))


def author_stats(path: str | Path) -> pl.DataFrame:
    """Per author in one shard: original posts and how many match the tech regex."""
    return (
        _roots(pl.scan_parquet(path)).select(pl.col("profile_id").cast(pl.Int64, strict=False).alias("author_id"), is_tech_post().alias("t"))
        .group_by("author_id").agg(pl.len().alias("posts"), pl.col("t").sum().alias("tech_posts")).collect()
    )


def read_shard(path: str | Path, mature_before: str = "2026-05-01", keep_authors: pl.Series | None = None) -> pl.DataFrame:
    """Original posts from one shard in the unified schema: tech-keyword matches plus `keep_authors`' posts."""
    path = Path(path)
    lf = _roots(pl.scan_parquet(path)).with_columns(pl.col("profile_id").cast(pl.Int64, strict=False).alias("author_id"))
    keep = is_tech_post()
    if keep_authors is not None:
        keep = keep | pl.col("author_id").is_in(keep_authors.implode())
    df = lf.filter(keep).collect()
    cutoff = date_to_ms(mature_before)
    q = pl.col("quoted-post").struct
    df = df.with_columns(
        pl.col("id").cast(pl.Int64, strict=False).alias("post_id"),
        pl.col("username").str.strip_prefix("@").alias("author_handle"),
        pl.col("name").alias("author_display_name"),
        pl.col("date").str.to_datetime(strict=False, time_zone="UTC").dt.epoch("ms").alias("created_at_src_ms"),
        # The shift: `quotes` holds likes and `likes` holds views.
        pl.col("quotes").alias("likes_true"), pl.col("likes").alias("views_true"),
        pl.col("comments").alias("replies"), pl.col("retweets").alias("reposts"),
        pl.when(q.field("link").is_not_null()).then(pl.lit("quote")).otherwise(pl.lit("original")).alias("kind"),
        q.field("text").alias("quoted_text"),
        pl.lit(False).alias("is_reply"),
    ).filter(pl.col("post_id").is_not_null()).with_columns(snowflake_ms_expr("post_id").alias("created_at_ms"))
    df = df.drop("likes", "quotes").rename({"likes_true": "likes", "views_true": "views"}).with_columns(
        pl.lit(None, dtype=pl.Int64).alias("quotes"),
        # Mature only before the cutoff; later posts keep a null snapshot and are dropped by the maturity filter.
        pl.when(pl.col("created_at_ms") < cutoff).then(pl.lit("assumed_mature")).otherwise(pl.lit("unknown")).alias("snapshot_kind"),
        pl.lit("count").alias("views_state"),
        pl.lit(SOURCE).alias("source"), pl.lit(path.name).alias("source_file"), pl.lit(False).alias("synthetic_id"),
    )
    media, urls = [], []
    for pics, gifs, ext in zip(df["pictures"].to_list(), df["gifs"].to_list(), df["external-link"].to_list()):
        m = [{"type": "photo", "url": u, "video_url": None, "width": None, "height": None, "duration_ms": None, "alt": None} for u in (pics or [])]
        m += [{"type": "gif", "url": u, "video_url": None, "width": None, "height": None, "duration_ms": None, "alt": None} for u in (gifs or [])]
        media.append(m)
        urls.append([{"tco": None, "expanded": ext, "final_domain": registrable_domain(ext), "is_media": False}] if ext else [])
    df = df.with_columns(pl.Series("media", media, dtype=pl.List(MEDIA_STRUCT)), pl.Series("urls", urls, dtype=pl.List(URL_STRUCT)),
                         pl.col("text").fill_null("").str.count_matches(r"(?:^|[^\w])@\w{1,15}").cast(pl.Int32).alias("mentions_n"),
                         pl.col("text").fill_null("").str.count_matches(r"(?:^|[^\w&])#\w+").cast(pl.Int32).alias("hashtags_n"))
    return conform(df)


def ingest(shard_dir: str | Path, out_dir: str | Path, mature_before: str = "2026-05-01", min_tech_posts: int = 3,
           min_tech_share: float = 0.3) -> dict:
    files = sorted(Path(shard_dir).glob("*.parquet"))
    out = Path(out_dir)
    stats = pl.concat([author_stats(f) for f in files]).group_by("author_id").agg(pl.col("posts", "tech_posts").sum())
    authors = stats.filter((pl.col("tech_posts") >= min_tech_posts) & (pl.col("tech_posts") / pl.col("posts") >= min_tech_share))
    log.info("nitter_2026: %d authors, %d tech authors", stats.height, authors.height)
    n = 0
    for f in files:
        target = out / "posts" / f"{f.stem}.parquet"
        if target.exists():
            continue
        df = read_shard(f, mature_before, authors["author_id"])
        write_parquet(df, target)
        n += df.height
        log.info("nitter_2026 %s: kept %d", f.name, df.height)
    return {"shards": len(files), "authors": stats.height, "tech_authors": authors.height, "kept": n}
