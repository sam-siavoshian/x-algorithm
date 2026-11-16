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


# ---------------------------------------------------------------------------------------------------------------
# Whole-timeline reader (2026-11-16, data track). The reader above keeps tech-bio accounts' originals only. For the
# startup use the bios, the small accounts and the replies matter, so this one keeps every post of a given account
# list, marks replies and reposts from the text (the source has no flags), and checks each shard before trusting it.
#
# Checked on all 41 shards (scripts/data_track/enryu_run.py):
# - `users.id` is the real X user id (paulg 183749519, sama 1605), handles and ids are unique in the users table.
# - `date` agrees with the time in the post id to the second in 99.6% of rows; the rest are dropped, as are ids from
#   before snowflake ids (2010).
# - likes >= quotes in 99.9% and likes >= reposts in 98% of rows with 20 or more interactions; a shard where either
#   falls under 95% is refused (a shifted or re-ordered count column would break both).
# - Self-replies inside a thread cannot be told from originals: the text carries no handle and there is no
#   conversation id. Replies to other accounts start with "@handle".

LAUNCH_HINT = (r"(?i)(introduc|announc|unveil|launch|releas|now live|it's live|is live|it's here|say hello|say hi|meet |presenting|shipped|shipping|rolling out|rolled out|"
               r"now support|you can now|now you can|changelog|new feature|open[- ]?sourc|raised|closed|funding|seed|series [a-e]|hiring|join our team|join my team|"
               r"available|beta|waitlist|early access|out now|hit |crossed|reached|passed|surpassed|demo|in action|walkthrough|sneak peek|here's)")


def all_users(users_path: str | Path) -> pl.DataFrame:
    u = pl.read_parquet(users_path).rename({"id": "author_id", "user": "author_handle", "description": "author_bio"})
    if u["author_handle"].str.to_lowercase().n_unique() != u.height or u["author_id"].n_unique() != u.height:
        raise ValueError("twitter100m users: handles or ids are not unique")
    return u.with_columns(pl.col("author_handle").str.to_lowercase().alias("h"))


def _base(path: Path, min_created: str) -> pl.LazyFrame:
    """Rows of one shard with ids, times and the kind read from the text; corrupt-time rows removed."""
    txt = pl.col("tweet").fill_null("")
    return (
        pl.scan_parquet(path)
        .with_columns(snowflake_ms_expr("id").alias("created_at_ms"), pl.col("date").str.to_datetime(strict=False, time_zone="UTC").dt.epoch("ms").alias("created_at_src_ms"))
        .filter(pl.col("created_at_ms").is_not_null() & ((pl.col("created_at_ms") - pl.col("created_at_src_ms")).abs() <= 1000) & (pl.col("created_at_ms") >= date_to_ms(min_created)))
        .with_columns(pl.when(txt.str.starts_with("RT @")).then(pl.lit("repost")).when(txt.str.contains(r"^\s*@\w")).then(pl.lit("reply")).otherwise(pl.lit("original")).alias("kind"),
                      pl.col("user").str.to_lowercase().alias("h"))
    )


def check_shard(path: str | Path) -> dict:
    """The guards. Raises when the count columns do not look like likes / reposts / quotes in that order."""
    d = pl.scan_parquet(path).with_columns(snowflake_ms_expr("id").alias("sf"), pl.col("date").str.to_datetime(strict=False, time_zone="UTC").dt.epoch("ms").alias("src"))
    busy = (pl.col("likes") + pl.col("retweets") + pl.col("replies") + pl.col("quotes")) >= 20
    r = d.select(pl.len().alias("rows"), pl.col("sf").is_null().sum().alias("pre_snowflake"), ((pl.col("sf") - pl.col("src")).abs() > 1000).sum().alias("time_mismatch"),
                 busy.sum().alias("busy"), (busy & (pl.col("likes") >= pl.col("quotes"))).sum().alias("likes_ge_quotes"), (busy & (pl.col("likes") >= pl.col("retweets"))).sum().alias("likes_ge_reposts"),
                 pl.col("id").n_unique().alias("unique_ids"), pl.col("sf").max().alias("last_ms")).collect().row(0, named=True)
    if r["busy"] >= 1000 and (r["likes_ge_quotes"] / r["busy"] < 0.95 or r["likes_ge_reposts"] / r["busy"] < 0.95):
        raise ValueError(f"{Path(path).name}: count columns do not look like likes/reposts/quotes (likes>=quotes {r['likes_ge_quotes'] / r['busy']:.1%}, likes>=reposts {r['likes_ge_reposts'] / r['busy']:.1%})")
    return r


def read_shard_accounts(path: str | Path, users: pl.DataFrame, keep: pl.Series, snapshot: str = "2023-04-21", min_created: str = "2020-01-01", kinds: tuple[str, ...] = ("original", "reply")) -> pl.DataFrame:
    """Every post (of the given kinds) by the accounts in `keep` (lower-case handles), in the unified schema, with bio and follower count."""
    path = Path(path)
    df = (_base(path, min_created).filter(pl.col("h").is_in(keep.implode()) & pl.col("kind").is_in(list(kinds)))
          .join(users.lazy().select("h", "author_id", "author_handle", "followers", "verified", "author_bio"), on="h", how="inner").collect())
    txt = pl.col("tweet").fill_null("")
    df = df.with_columns(
        pl.col("id").alias("post_id"), pl.col("tweet").alias("text"), pl.col("retweets").alias("reposts"),
        pl.col("followers").alias("author_followers"), pl.col("verified").alias("author_verified"), (pl.col("kind") == "reply").alias("is_reply"),
        pl.lit(date_to_ms(snapshot)).alias("snapshot_at_ms"), pl.lit("file_date").alias("snapshot_kind"),
        pl.lit("absent").alias("views_state"), pl.lit(SOURCE).alias("source"), pl.lit(path.name).alias("source_file"), pl.lit(False).alias("synthetic_id"),
        txt.str.count_matches(r"(?:^|[^\w])@\w{1,15}").cast(pl.Int32).alias("mentions_n"), txt.str.count_matches(r"(?:^|[^\w&])#\w+").cast(pl.Int32).alias("hashtags_n"),
    )
    return conform(df)
