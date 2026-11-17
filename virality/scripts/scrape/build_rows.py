#!/usr/bin/env python3
"""Training rows from the home-timeline scrape, through the stages of xviral/pipeline.py (the functions of
scripts/expand/build_rows.py), appended to work/dataset/rows.parquet by scripts/expand/build_table.py.

    python scripts/scrape/build_rows.py --out data/scrape_track

What differs from the pipeline, and why:
  accounts   every scrape account is its own account (id = hash of "home_scrape/<handle>"), also when the handle is in
             the corpus. Its history then holds only posts sampled and read the same way; a history that steps from
             Community Archive rows (mature likes, no views) into scrape rows (counts read after 4 hours) would carry
             that step as if it were the account changing.
  maturity   NOT applied. 97.7% of the rows were read under 72 hours after posting, so the pipeline's rule leaves 6,351
             posts. Rows keep the counts as read; the capture age goes to the model as `src_regime` (home_0_1h ...
             home_24h_plus), and `age_h` is kept as a column.
  volume     no minimum: accounts with one post are kept (flag `acct_n`); the training variants choose by `acct_n`.
  kept out   posts that are training rows already (any split), accounts with a benchmark calibration pair, reposts.
  split      by account hash: a tenth validation, a tenth held out ("test": never trained on, never selected on), the
             rest train. Every row is before the 2026-04-23 cutoff.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from xviral import dedup, filters, labels
from xviral.config import Config
from xviral.schema import POSTS_SCHEMA
from xviral.text.tech import is_off_audience_post, is_tech_post
from xviral.util.core import date_to_ms, stable_hash64

spec = importlib.util.spec_from_file_location("expand_build", Path(__file__).resolve().parents[1] / "expand" / "build_rows.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
log = logging.getLogger("scrape")
D = Path("data")
CUT = date_to_ms("2026-04-23")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scrape_track")
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--limit-accounts", type=int, default=0, help="a test run on this many accounts")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out)
    cfg = Config.load(D / "config.yaml")
    cfg.filters.timestamp_tolerance_ms_by_source["home_scrape"] = 1000
    t0 = time.time()
    rows_schema = pl.scan_parquet(D / "work/dataset/rows.parquet").collect_schema()
    rows = pl.read_parquet(D / "work/dataset/rows.parquet", columns=["post_id", "author_id", "author_handle", "mutual_share_est", "followers_at_post", "author_following", "created_at_ms"])
    beta = ex.fit_mutual(rows)
    filt_n = pl.scan_parquet(D / "work/filtered/posts.parquet").select("author_id", pl.col("author_handle").str.to_lowercase().alias("handle")).unique().collect()
    bench = pl.read_parquet(D / "tracks/pairs/benchmark_pairs.parquet")
    cal_handles = filt_n.filter(pl.col("author_id").is_in(bench.filter(pl.col("part") == "calibration")["author_id"].unique().implode()))["handle"].unique()
    S = pl.read_parquet(D / "tracks/data/startup_accounts.parquet").select("handle", pl.col("kind").alias("acct_kind"))

    posts = pl.read_parquet(out / "posts.parquet").with_columns(pl.col("author_handle").str.to_lowercase().alias("handle"))
    counts: dict = {"read": posts.height, "accounts_read": posts["handle"].n_unique()}
    posts = posts.with_columns(pl.col("handle").map_elements(lambda h: -stable_hash64("author", "home_scrape/" + h), return_dtype=pl.Int64).alias("author_id"))
    assert posts.join(rows.select("author_id").unique(), on="author_id", how="semi").height == 0
    n = posts.height
    posts = posts.join(rows.select("post_id"), on="post_id", how="anti")
    counts["already_training_rows"] = n - posts.height
    n = posts.height
    posts = posts.filter(~pl.col("handle").is_in(cal_handles.implode()))
    counts["rows_of_benchmark_calibration_accounts"] = n - posts.height
    if args.limit_accounts:
        keep = posts["author_id"].unique().sort().head(args.limit_accounts)
        posts = posts.filter(pl.col("author_id").is_in(keep.implode()))
    handle_of = posts.select("author_id", "handle").unique("author_id")
    age = posts.select("post_id", ((pl.col("snapshot_at_ms") - pl.col("created_at_ms")) / 3.6e6).alias("age_h"))
    posts = posts.select(list(POSTS_SCHEMA))

    flog = filters.FilterLog()
    posts = flog.apply("validity", posts, lambda d: filters.validity(d, cfg.filters), exempt_analytics=False)
    posts = flog.apply("kind", posts, lambda d: filters.kind(d, cfg.filters))
    posts = flog.apply("recency", posts, lambda d: filters.recency(d, cfg.filters))
    lc = out / "lang.parquet"
    posts = filters.resolve_language(posts, cfg.filters, pl.read_parquet(lc) if lc.exists() else None)
    posts = flog.apply("language", posts, lambda d: filters.keep_languages(d, cfg.filters))
    ex.rss("filtered")
    st = ex.author_stage(posts.select(list(POSTS_SCHEMA)), None, cfg)
    ex.rss("authors")
    bad = filters.hygiene_bad_accounts(st["atab"], cfg.hygiene)
    counts["hygiene_accounts_dropped"] = int(bad.len())
    st["atab"].join(handle_of, on="author_id", how="left").write_parquet(out / "author_table.parquet")
    posts = flog.apply("hygiene", posts, lambda d: d.filter(~pl.col("author_id").is_in(bad.implode())), exempt_analytics=False)
    posts = flog.apply("select_cap", posts, lambda d: d.with_columns(pl.col("created_at_ms").rank("ordinal", descending=True).over("author_id").alias("_r")).filter(pl.col("_r") <= args.cap).drop("_r"), exempt_analytics=False)
    flog.frame().write_csv(out / "filter_log.csv")
    print(flog.frame())

    ts_ = posts.group_by("author_id").agg(is_tech_post(strong=True).mean().alias("acct_tech_share"), is_off_audience_post().mean().alias("acct_off_share"), pl.len().alias("acct_n"))
    nd = dedup.near_duplicate_clusters(posts)
    ex.rss("neardup")
    if "sources_all" not in posts.columns:
        posts = posts.with_columns(pl.col("source").alias("sources_all"))
    # Chunks of whole accounts, about 6,000 posts each.
    sizes = posts.group_by("author_id").len().sort("author_id")
    chunk = (sizes["len"].cum_sum() // 6000).to_numpy()
    posts = posts.join(sizes.select("author_id", pl.Series("_chunk", chunk)), on="author_id")
    parts = []
    for c in range(int(chunk.max()) + 1):
        parts.append(ex.feature_stage(posts.filter(pl.col("_chunk") == c).drop("_chunk"), st, nd))
        if c % 5 == 0:
            ex.rss(f"features {c}/{int(chunk.max()) + 1}")
    df = pl.concat(parts, how="vertical_relaxed")
    assert df.height == posts.height and df["post_id"].n_unique() == df.height
    del parts, posts
    df = ex.apply_mutual(df, beta).join(ts_, on="author_id", how="left").join(handle_of, on="author_id", how="left").join(S, on="handle", how="left").join(age, on="post_id", how="left")
    # Audience: the pipeline's account rule (tech share 0.15, off-audience share not above it) for accounts with five or
    # more posts, where a share means something; smaller accounts are general accounts.
    df = df.with_columns(((pl.col("acct_n") >= 5) & (pl.col("acct_tech_share") >= 0.15) & (pl.col("acct_off_share") <= pl.col("acct_tech_share"))).alias("in_audience"),
                         pl.col("_mut").alias("mutual_share_est"))
    df = labels.build(df, cfg.labels)
    h = pl.col("author_id").hash(seed=cfg.splits.seed) % 10
    df = df.with_columns(pl.when(h == 0).then(pl.lit("val")).when(h == 1).then(pl.lit("test")).otherwise(pl.lit("train")).alias("split"), pl.lit("home_scrape").alias("expand_src")).drop("_mut")
    miss = [c for c in rows_schema.names() if c not in df.columns]
    log.info("columns of rows.parquet absent from the new rows (left null): %s", miss)
    df = df.with_columns([pl.lit(None, dtype=rows_schema[c]).alias(c) for c in miss])
    extra = ["expand_src", "acct_kind", "acct_n", "age_h", "handle", "snapshot_at_ms"]
    extra = [c for c in extra if c not in rows_schema.names()]
    df = df.select([pl.col(c).cast(rows_schema[c], strict=False) for c in rows_schema.names()] + extra)
    assert df.filter(pl.col("created_at_ms") >= CUT).height == 0
    df.write_parquet(out / "new_rows.parquet", compression="zstd")
    counts.update(new_rows=df.height, new_accounts=df["author_id"].n_unique(), seconds=round(time.time() - t0),
                  by_split={r["split"]: {"rows": r["len"], "accounts": r["a"]} for r in df.group_by("split").agg(pl.len(), pl.col("author_id").n_unique().alias("a")).iter_rows(named=True)},
                  audience_rows=int(df["in_audience"].sum()), audience_accounts=df.filter(pl.col("in_audience"))["author_id"].n_unique(),
                  rows_of_accounts_20plus=df.filter(pl.col("acct_n") >= 20).height, accounts_20plus=df.filter(pl.col("acct_n") >= 20)["author_id"].n_unique(),
                  rows_of_accounts_under_20=df.filter(pl.col("acct_n") < 20).height, single_post_accounts=df.filter(pl.col("acct_n") == 1).height,
                  with_views=int(df["views"].is_not_null().sum()), age_h_median=float(df["age_h"].median()),
                  startup_list_rows=int(df["acct_kind"].is_not_null().sum()))
    (out / "build_counts.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
