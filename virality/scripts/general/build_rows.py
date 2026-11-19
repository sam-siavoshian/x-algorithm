#!/usr/bin/env python3
"""Rows of the general track's accounts (data/general/accounts.parquet: test bed, validation, training), through the same
pipeline stages as scripts/expand/build_rows.py (its stage functions are imported, not copied).

    python scripts/general/build_rows.py --out data/general [--cap-train 64 --cap-val 100 --cap-bed 120 --ctx 250]

Differences from the expand build, all to fit a 15 GB machine and the time budget:
  - an account is read from the one source it was sampled from, and only its most recent --ctx originals and quotes are read at
    full width (trailing statistics need the 30 posts before a row, hygiene counts repeats; neither needs years of timeline);
  - the cap is per role: the most recent 64 posts of a training account (posts are subsampled, not accounts), 100 of a
    validation account, 120 of a test-bed account;
  - brand accounts that the expand track already built are not rebuilt (their rows are relabelled when the table is built);
  - training and validation accounts keep no post from 2026-04-23 on; test-bed accounts keep theirs (no model trains on them);
  - these are general accounts: in_audience is false for all of them.
"""
from __future__ import annotations

import argparse, importlib.util, json, logging, time
from pathlib import Path

import numpy as np
import polars as pl

from xviral import dedup, filters, labels, unify
from xviral.config import Config
from xviral.schema import POSTS_SCHEMA, conform
from xviral.util.core import date_to_ms, synthetic_author_id

spec = importlib.util.spec_from_file_location("expand_build", "scripts/expand/build_rows.py")
E_ = importlib.util.module_from_spec(spec); spec.loader.exec_module(E_)
log = logging.getLogger("general")
D, SRC, CUT, LOW, rss = E_.D, E_.SRC, E_.CUT, E_.LOW, E_.rss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/general"); ap.add_argument("--accounts", default="data/general/accounts.parquet")
    ap.add_argument("--cap-train", type=int, default=64); ap.add_argument("--cap-val", type=int, default=100); ap.add_argument("--cap-bed", type=int, default=120)
    ap.add_argument("--ctx", type=int, default=250); ap.add_argument("--min-posts", type=int, default=20)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out); cfg = Config.load(D / "config.yaml"); t0 = time.time()
    rows_schema = pl.scan_parquet(D / "work/dataset/rows.parquet").collect_schema()
    rows = pl.read_parquet(D / "work/dataset/rows.parquet", columns=["post_id", "author_id", "author_handle", "mutual_share_est", "followers_at_post", "author_following", "created_at_ms"])
    exp_n = pl.read_parquet(D / "tracks/expand/rows/new_rows.parquet", columns=["post_id", "author_id", "author_handle"])
    filt_n = pl.scan_parquet(D / "work/filtered/posts.parquet").select("post_id", "author_id", LOW.alias("handle")).collect()
    rows_n = rows.select("post_id", "author_id", LOW.alias("handle"))
    hid = E_.handle_ids(rows_n, filt_n)
    beta = E_.fit_mutual(rows)
    S = pl.read_parquet(args.accounts).filter(~pl.col("in_expand")).select("handle", "src", "role", "tier", "fol")
    counts: dict = {"accounts_selected": S.height, "caps": {"train": args.cap_train, "val": args.cap_val, "bed": args.cap_bed}, "ctx": args.ctx}

    frames = []
    for name in ("twitter100m", "zenodo_brands", "nitter_2026_replies"):
        hs = S.filter(pl.col("src") == name)["handle"]
        lf = pl.scan_parquet(SRC[name]).with_columns(LOW.alias("handle")).filter(pl.col("handle").is_in(hs.implode()) & pl.col("kind").is_in(cfg.filters.keep_kinds) & (pl.col("created_at_ms") >= date_to_ms("2020-01-01")))
        if name == "nitter_2026_replies":
            lf = lf.filter(pl.col("created_at_ms") < date_to_ms("2026-05-01"))        # later posts were read young (DATA_PIPELINE 15.1)
        nar = lf.select("handle", "created_at_ms").collect()
        thr = nar.with_columns(pl.col("created_at_ms").rank("ordinal", descending=True).over("handle").alias("_r")).filter(pl.col("_r") <= args.ctx).group_by("handle").agg(pl.col("created_at_ms").min().alias("_thr"))
        counts[f"read_{name}"] = {"eligible_rows": nar.height, "accounts": nar["handle"].n_unique()}
        d = lf.join(thr.lazy(), on="handle").filter(pl.col("created_at_ms") >= pl.col("_thr")).drop("_thr").collect()
        if name == "nitter_2026_replies":
            d = d.with_columns(pl.when(pl.col("snapshot_at_ms").is_null()).then(pl.lit("unknown")).otherwise(pl.col("snapshot_kind")).alias("snapshot_kind"))
        frames.append(d.with_columns(pl.lit(name).alias("expand_src")))
        log.info("%s: %d eligible rows of %d accounts, %d read at full width", name, nar.height, nar["handle"].n_unique(), d.height); rss(name)
    new = pl.concat(frames, how="diagonal_relaxed"); del frames
    real = new.filter(pl.col("author_id") > 0).group_by("handle").agg(pl.col("author_id").mode().first().alias("_rid"))
    idmap = new.select("handle").unique().join(hid, on="handle", how="left").join(real, on="handle", how="left")
    idmap = idmap.with_columns(pl.coalesce("_cid", "_rid", pl.col("handle").map_elements(synthetic_author_id, return_dtype=pl.Int64)).alias("_aid"))
    dup = idmap.filter(pl.col("_aid").is_duplicated())
    if dup.height:                      # two sampled handles resolving to one account: keep the first handle
        drop_h = dup.sort("handle").filter(pl.col("_aid").is_duplicated() & ~pl.col("_aid").is_first_distinct())["handle"]
        new, idmap = new.filter(~pl.col("handle").is_in(drop_h.implode())), idmap.filter(~pl.col("handle").is_in(drop_h.implode()))
        counts["handles_dropped_same_account"] = int(drop_h.len())
    seen_ids = pl.concat([rows_n["author_id"], exp_n["author_id"]]).unique()
    clash = idmap.filter(pl.col("_aid").is_in(seen_ids.implode()))["handle"]      # an account another handle of which is already trained on
    new, idmap = new.filter(~pl.col("handle").is_in(clash.implode())), idmap.filter(~pl.col("handle").is_in(clash.implode()))
    counts["handles_dropped_account_already_trained"] = int(clash.len())
    new = new.join(idmap.select("handle", "_aid"), on="handle").with_columns(pl.col("_aid").alias("author_id")).drop("_aid")
    src_of = new.select("post_id", "expand_src").unique("post_id", keep="first")
    handle_of = new.select("author_id", "handle").unique("author_id")
    posts = unify.merge_posts([new.select(list(POSTS_SCHEMA))]); del new
    n0 = posts.height
    posts = posts.join(pl.concat([rows_n.select("post_id"), exp_n.select("post_id")]), on="post_id", how="anti")
    counts.update(unique_posts=n0, already_rows=n0 - posts.height)
    posts = posts.with_columns(pl.when(pl.col("source") == "twitterapi_io").then(pl.lit("zenodo_brands")).otherwise(pl.col("source")).alias("source"))
    flog = filters.FilterLog()
    posts = flog.apply("validity", posts, lambda d: filters.validity(d, cfg.filters), exempt_analytics=False)
    posts = flog.apply("kind", posts, lambda d: filters.kind(d, cfg.filters))
    posts = flog.apply("maturity", posts, lambda d: filters.maturity(d, cfg.filters, None), exempt_analytics=False)
    posts = flog.apply("recency", posts, lambda d: filters.recency(d, cfg.filters))
    lc = out / "lang_cache.parquet"
    posts = filters.resolve_language(posts, cfg.filters, pl.read_parquet(lc) if lc.exists() else None)
    posts.select("post_id", "lang").write_parquet(lc)
    posts = flog.apply("language", posts, lambda d: filters.keep_languages(d, cfg.filters))
    posts = flog.apply("volume", posts, lambda d: d.filter(pl.len().over("author_id") >= args.min_posts), exempt_analytics=False)
    log.info("filters done: %d posts, %d accounts, %.0fs", posts.height, posts["author_id"].n_unique(), time.time() - t0); rss("filtered")

    aid = posts["author_id"].unique()
    old = pl.scan_parquet(D / "work/filtered/posts.parquet").filter(pl.col("author_id").is_in(aid.implode())).collect().join(posts.select("post_id"), on="post_id", how="anti")
    ctx = pl.concat([posts.select(list(POSTS_SCHEMA)), conform(old)], how="vertical_relaxed")
    counts["context_posts_from_existing_corpus"] = old.height; del old
    st = E_.author_stage(ctx, None, cfg); del ctx; rss("authors")
    bad = filters.hygiene_bad_accounts(st["atab"], cfg.hygiene)
    counts["hygiene_accounts_dropped"] = int(bad.len())
    posts = flog.apply("hygiene", posts, lambda d: d.filter(~pl.col("author_id").is_in(bad.implode())), exempt_analytics=False)
    meta = handle_of.join(S, on="handle", how="left")
    posts = posts.join(meta.select("author_id", "role"), on="author_id", how="left")
    posts = flog.apply("no_post_cutoff_rows_for_train_val", posts, lambda d: d.filter((pl.col("role") == "bed") | (pl.col("created_at_ms") < CUT)), exempt_analytics=False)
    cap = pl.when(pl.col("role") == "train").then(args.cap_train).when(pl.col("role") == "val").then(args.cap_val).otherwise(args.cap_bed)
    posts = flog.apply("select_cap", posts, lambda d: d.with_columns(pl.col("created_at_ms").rank("ordinal", descending=True).over("author_id").alias("_r")).filter(pl.col("_r") <= cap).drop("_r"), exempt_analytics=False)
    posts = flog.apply("volume_after_cap", posts, lambda d: d.filter(pl.len().over("author_id") >= args.min_posts), exempt_analytics=False).drop("role")
    flog.frame().write_csv(out / "filter_log.csv"); print(flog.frame())

    from xviral.text.tech import is_off_audience_post, is_tech_post
    ts_ = posts.select("author_id", "text").group_by("author_id").agg(is_tech_post(strong=True).mean().alias("acct_tech_share"), is_off_audience_post().mean().alias("acct_off_share"))
    nd = dedup.near_duplicate_clusters(posts); rss("neardup")
    if "sources_all" not in posts.columns:
        posts = posts.with_columns(pl.col("source").alias("sources_all"))
    ids_all = posts["author_id"].unique().sort()
    parts = []
    for k in range(0, ids_all.len(), 60):
        parts.append(E_.feature_stage(posts.filter(pl.col("author_id").is_in(ids_all.slice(k, 60).implode())), st, nd))
        if k % 1200 == 0:
            rss(f"features {k}/{ids_all.len()}")
    df = pl.concat(parts, how="vertical_relaxed")
    assert df.height == posts.height and df["post_id"].n_unique() == df.height
    del parts, posts
    df = E_.apply_mutual(df, beta).join(ts_, on="author_id", how="left").join(meta.select("author_id", pl.col("role").alias("g_role"), pl.col("tier").alias("g_tier"), pl.col("fol").alias("g_followers")), on="author_id", how="left").join(src_of, on="post_id", how="left")
    df = df.with_columns(pl.lit(False).alias("in_audience"), pl.col("_mut").alias("mutual_share_est"))
    df = labels.build(df, cfg.labels)
    df = df.with_columns(pl.when(pl.col("g_role") == "train").then(pl.lit("train")).when(pl.col("g_role") == "val").then(pl.lit("val")).otherwise(pl.lit("test")).alias("split"),
                         pl.lit(None, dtype=pl.String).alias("acct_kind"), pl.lit(None, dtype=pl.String).alias("kind_from"), (pl.col("expand_src") == "zenodo_brands").alias("in_brand_set"), pl.lit(False).alias("acct_in_original")).drop("_mut")
    miss = [c for c in rows_schema.names() if c not in df.columns]
    log.info("columns of rows.parquet absent from the new rows (left null): %s", miss)
    df = df.with_columns([pl.lit(None, dtype=rows_schema[c]).alias(c) for c in miss])
    extra = ["expand_src", "acct_kind", "kind_from", "in_brand_set", "acct_in_original", "g_role", "g_tier", "g_followers"]
    df = df.select([pl.col(c).cast(rows_schema[c], strict=False) for c in rows_schema.names()] + extra)
    assert df.filter((pl.col("split") != "test") & (pl.col("created_at_ms") >= CUT)).height == 0
    assert df.join(rows_n, on="author_id", how="semi").height == 0 and df.join(exp_n, on="author_id", how="semi").height == 0
    df.write_parquet(out / "new_rows.parquet", compression="zstd")
    counts.update(new_rows=df.height, new_accounts=df["author_id"].n_unique(), seconds=round(time.time() - t0))
    counts["by_role_source"] = df.group_by("g_role", "expand_src").agg(pl.len().alias("rows"), pl.col("author_id").n_unique().alias("accounts"), pl.col("views").is_not_null().sum().alias("with_views")).sort("g_role", "expand_src").to_dicts()
    (out / "build_counts.json").write_text(json.dumps(counts, indent=1)); print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
