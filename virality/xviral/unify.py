"""Merge every source into one posts table, one accounts table and one edges table.

Deduplication order (DATA_PIPELINE.md section 3):
1. Rows without a post ID (synthetic IDs from the scrape) are matched to real
   rows by (handle, creation second, text hash) and dropped if matched.
2. Exact post ID across sources: one row per post. For every column the first
   non-null value wins, in priority order: rows with views, then the latest
   snapshot, then source priority analytics > user_scrape > usc > community_archive.
   This keeps the most mature counts and carries the union of fields.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from xviral.schema import ACCOUNTS_SCHEMA, POSTS_SCHEMA, conform
from xviral.util.core import scan_dir, write_parquet

log = logging.getLogger(__name__)

SOURCE_PRIORITY = {"analytics": 0, "user_scrape": 1, "nitter_2026": 2, "usc": 3, "community_archive": 4, "twitter100m": 5}


def _norm_text_hash() -> pl.Expr:
    return (
        pl.col("text").fill_null("").str.to_lowercase()
        .str.replace_all(r"https?://\S+", "")
        .str.replace_all(r"\s+", " ").str.strip_chars()
        .hash(seed=7)
    )


def reconcile_synthetic(posts: pl.DataFrame) -> pl.DataFrame:
    """Drop synthetic-ID rows that match a real-ID row from any source."""
    synth = posts.filter(pl.col("synthetic_id"))
    if synth.is_empty():
        return posts
    key = [
        pl.col("author_handle").str.to_lowercase().alias("_h"),
        (pl.col("created_at_src_ms").fill_null(pl.col("created_at_ms")) // 1000).alias("_s"),
        _norm_text_hash().alias("_t"),
    ]
    real_keys = posts.filter(~pl.col("synthetic_id")).select(key).unique()
    matched = synth.with_columns(key).join(real_keys, on=["_h", "_s", "_t"], how="semi").select("post_id")
    log.info("reconcile: %d of %d synthetic rows matched a real post", matched.height, synth.height)
    return posts.join(matched, on="post_id", how="anti")


def _merge_duplicates(posts: pl.DataFrame) -> pl.DataFrame:
    """Field-level merge of rows sharing a post ID (see module docstring for the priority)."""
    posts = posts.with_columns(
        pl.col("views").is_null().alias("_no_views"),
        pl.col("snapshot_at_ms").fill_null(0).alias("_snap"),
        pl.col("source").replace_strict(SOURCE_PRIORITY, default=9, return_dtype=pl.Int32).alias("_prio"),
    ).sort(["post_id", "_no_views", "_snap", "_prio"], descending=[False, False, True, False])
    cols = [c for c in POSTS_SCHEMA if c != "post_id"]
    merged = posts.group_by("post_id", maintain_order=True).agg(
        *[pl.col(c).drop_nulls().first().alias(c) for c in cols],
        pl.col("source").unique().sort().str.join("+").alias("sources_all"),
    )
    # Lists (media, urls) are never null after conform; prefer the longest non-empty one.
    lists = posts.group_by("post_id").agg(
        pl.col("media").sort_by(pl.col("media").list.len(), descending=True).first().alias("media"),
        pl.col("urls").sort_by(pl.col("urls").list.len(), descending=True).first().alias("urls"),
    )
    return merged.drop("media", "urls").join(lists, on="post_id", how="left").select([*POSTS_SCHEMA.keys(), "sources_all"])


def merge_posts(frames: list[pl.DataFrame]) -> pl.DataFrame:
    posts = pl.concat([conform(f) for f in frames], how="vertical_relaxed")
    n0 = posts.height
    posts = reconcile_synthetic(posts)
    # Almost every post ID appears once. Only the repeated ones go through the grouped merge, which
    # keeps memory flat on corpora of tens of millions of rows.
    dup = posts["post_id"].is_duplicated()
    single = posts.filter(~dup).with_columns(pl.col("source").alias("sources_all"))
    merged = pl.concat([single.select([*POSTS_SCHEMA.keys(), "sources_all"]), _merge_duplicates(posts.filter(dup))], how="vertical_relaxed") \
        if dup.any() else single.select([*POSTS_SCHEMA.keys(), "sources_all"])
    log.info("merge: %d rows -> %d unique posts (%d rows shared an ID)", n0, merged.height, int(dup.sum()))
    return merged


def merge_accounts(frames: list[pl.DataFrame], posts: pl.DataFrame) -> pl.DataFrame:
    """All follower snapshots per author, for interpolation in xviral.authors."""
    from_posts = posts.select(
        "author_id", "author_handle",
        pl.col("snapshot_at_ms").fill_null(pl.col("created_at_ms")).alias("snapshot_at_ms"),
        pl.col("author_followers").alias("followers"), pl.col("author_following").alias("following"),
        pl.col("author_posts_count").alias("posts_count"), pl.col("author_created_at_ms").alias("created_at_ms"),
        pl.col("author_verified").alias("verified"), "source",
    )
    allf = [f.select(list(ACCOUNTS_SCHEMA)).cast(ACCOUNTS_SCHEMA, strict=False) for f in frames] + [from_posts.cast(ACCOUNTS_SCHEMA, strict=False)]
    acc = pl.concat(allf, how="vertical_relaxed").filter(pl.col("author_id").is_not_null() & pl.col("followers").is_not_null())
    return acc.unique(subset=["author_id", "snapshot_at_ms", "followers"]).sort("author_id", "snapshot_at_ms")


def resolve_handles(posts: pl.DataFrame, accounts: pl.DataFrame) -> pl.DataFrame:
    """Map synthetic (negative) author IDs to real ones by handle, using the most recent mapping."""
    real = (
        pl.concat([
            posts.filter(pl.col("author_id") > 0).select("author_id", "author_handle", pl.col("created_at_ms").alias("t")),
            accounts.filter(pl.col("author_id") > 0).select("author_id", "author_handle", pl.col("snapshot_at_ms").alias("t")),
        ])
        .filter(pl.col("author_handle").is_not_null())
        .with_columns(pl.col("author_handle").str.to_lowercase().alias("_h"))
        .sort("t", descending=True, nulls_last=True)
        .unique("_h", keep="first")
        .select("_h", pl.col("author_id").alias("_real_id"))
    )
    out = posts.with_columns(pl.col("author_handle").str.to_lowercase().alias("_h")).join(real, on="_h", how="left")
    n_fixed = out.filter((pl.col("author_id") < 0) & pl.col("_real_id").is_not_null()).height
    log.info("resolve_handles: %d rows moved from synthetic to real author IDs", n_fixed)
    return out.with_columns(
        pl.when(pl.col("author_id") < 0).then(pl.col("_real_id").fill_null(pl.col("author_id"))).otherwise(pl.col("author_id")).alias("author_id")
    ).drop("_h", "_real_id")


def run(stage_dirs: dict[str, Path], out_dir: Path) -> dict[str, pl.DataFrame]:
    """stage_dirs maps source name -> directory holding posts/, accounts/, edges/, activity/ subdirs."""
    posts_frames, acct_frames, edge_frames, act_frames, like_frames = [], [], [], [], []
    for name, d in stage_dirs.items():
        if (d / "posts").exists():
            posts_frames.append(scan_dir(d / "posts").collect())
        for sub, bucket in (("accounts", acct_frames), ("edges", edge_frames), ("activity", act_frames), ("viewer_likes", like_frames)):
            if (d / sub).exists() and any((d / sub).glob("*.parquet")):
                bucket.append(scan_dir(d / sub).collect())
        log.info("unify: loaded %s", name)
    posts = merge_posts(posts_frames)
    accounts = merge_accounts(acct_frames, posts)
    posts = resolve_handles(posts, accounts)
    out = {"posts": posts, "accounts": accounts}
    if edge_frames:
        out["edges"] = pl.concat(edge_frames, how="vertical_relaxed").unique()
    if act_frames:
        out["activity"] = (
            pl.concat(act_frames, how="vertical_relaxed")
            .group_by("author_id")
            .agg(pl.col("rows", "replies", "reposts", "originals").sum(), pl.col("first_ms").min(), pl.col("last_ms").max(),
                 pl.col("source").unique().sort().str.join("+").alias("source"))
        )
    if like_frames:
        out["viewer_likes"] = pl.concat(like_frames).unique()
    for k, v in out.items():
        write_parquet(v, out_dir / f"{k}.parquet")
    return out
