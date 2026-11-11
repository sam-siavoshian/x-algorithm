"""enryu43/twitter100m_tweets + twitter100m_users (Hugging Face; Zenodo 10.5281/zenodo.15086029).

88M posts from 145.8K accounts, full timelines up to 2023-04-20, with likes, replies, reposts and
quotes; a users table with followers and bios. No views, no media, no reply or quote flags.
It contains the actual tech accounts (paulg, sama, naval, levelsio, ylecun, dhh, patio11, pmarca,
balajis, swyx), which makes it the deepest like-labeled source of tech author timelines.

Tweets:  user, id, tweet, replies, retweets, likes, quotes, date
Users:   user, id, verified, followers, description, location

Counts are one snapshot taken around the last post date, so the newest weeks are immature; the
snapshot time is set to the corpus's last post date and the maturity filter handles the rest.
Accounts are kept by bio (tech regex) or by an explicit handle list; all their originals are kept.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from xviral.schema import conform
from xviral.text.tech import is_tech_bio
from xviral.util.core import date_to_ms, snowflake_ms_expr, write_parquet

log = logging.getLogger(__name__)

SOURCE = "twitter100m"


def tech_users(users_path: str | Path, seed_handles: list[str] | None = None) -> pl.DataFrame:
    u = pl.read_parquet(users_path).rename({"id": "author_id", "user": "author_handle", "description": "author_bio"})
    keep = is_tech_bio()
    if seed_handles:
        keep = keep | pl.col("author_handle").str.to_lowercase().is_in([h.lower().lstrip("@") for h in seed_handles])
    return u.filter(keep).unique("author_handle")


def read_shard(path: str | Path, users: pl.DataFrame, snapshot: str = "2023-04-21", min_created: str = "2020-01-01") -> pl.DataFrame:
    path = Path(path)
    df = (
        pl.scan_parquet(path).rename({"user": "author_handle"})
        .join(users.lazy().select("author_handle", "author_id", "followers", "verified", "author_bio"), on="author_handle", how="inner")
        .filter(~pl.col("tweet").fill_null("").str.contains(r"^(@|RT @)"))   # no reply or repost flags: use the text
        .collect()
    )
    df = df.with_columns(
        pl.col("id").alias("post_id"), pl.col("tweet").alias("text"), pl.col("retweets").alias("reposts"),
        pl.col("date").str.to_datetime(strict=False, time_zone="UTC").dt.epoch("ms").alias("created_at_src_ms"),
        pl.col("followers").alias("author_followers"), pl.col("verified").alias("author_verified"),
        pl.lit("original").alias("kind"), pl.lit(False).alias("is_reply"),
        pl.lit(date_to_ms(snapshot)).alias("snapshot_at_ms"), pl.lit("file_date").alias("snapshot_kind"),
        pl.lit("absent").alias("views_state"), pl.lit(SOURCE).alias("source"), pl.lit(path.name).alias("source_file"),
        pl.lit(False).alias("synthetic_id"),
        pl.col("tweet").fill_null("").str.count_matches(r"(?:^|[^\w])@\w{1,15}").cast(pl.Int32).alias("mentions_n"),
        pl.col("tweet").fill_null("").str.count_matches(r"(?:^|[^\w&])#\w+").cast(pl.Int32).alias("hashtags_n"),
    ).with_columns(snowflake_ms_expr("post_id").alias("created_at_ms")).filter(pl.col("created_at_ms") >= date_to_ms(min_created))
    return conform(df)


def ingest(tweets_dir: str | Path, users_path: str | Path, out_dir: str | Path, seed_handles: list[str] | None = None,
           snapshot: str = "2023-04-21", min_created: str = "2020-01-01") -> dict:
    users = tech_users(users_path, seed_handles)
    out = Path(out_dir)
    n = 0
    for f in sorted(Path(tweets_dir).glob("*.parquet")):
        target = out / "posts" / f"{f.stem}.parquet"
        if target.exists():
            continue
        df = read_shard(f, users, snapshot, min_created)
        write_parquet(df, target)
        n += df.height
    log.info("twitter100m: %d tech accounts, %d posts kept", users.height, n)
    return {"tech_accounts": users.height, "kept": n}
