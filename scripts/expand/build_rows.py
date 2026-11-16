#!/usr/bin/env python3
"""Expanded training rows: the posts of company and founder accounts (tracks/data/startup_accounts.parquet) from the three
sources the data track converted, taken through the stages of xviral/pipeline.py and appended to work/dataset/rows.parquet.

    python scripts/expand/build_rows.py --cap 400 --out data/expand            # build
    python scripts/expand/build_rows.py --check 150 --out data/expand          # replication check on existing accounts

Stages, in pipeline order, on the new posts only (the existing 465,675 rows are copied unchanged):
  select    accounts are chosen first, on a narrow scan (handle in the startup list, not a benchmark calibration account);
            only their rows are then read in full, so memory stays near 3 GB
  unify     one row per post id across the three sources (unify.merge_posts); posts already in rows.parquet are dropped
  filter    validity, kind (original, quote), maturity (72 h; the 2026 view corpus only before 2026-05-01), recency
            (2020 for twitter100m, 2023 otherwise), language (the pipeline's detector), volume (20 posts, or already a training account)
  authors   followers at post time, trailing statistics, self-repeats, the author table; computed over the account's new
            posts together with its posts in work/filtered/posts.parquet, so a history that spans sources is one history
  hygiene   the pipeline's automated-account rules
  select    the most recent --cap new posts per account
  dedup     near-duplicate clusters among the selected new posts
  features  the feature stage's joins and functions, then labels and the split (test from 2026-04-23, validation by account)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from xviral import authors, dedup, filters, labels, unify
from xviral.config import Config
from xviral.features import audience, content, derive, suppression, text_structure
from xviral.schema import POSTS_SCHEMA, conform
from xviral.util.core import date_to_ms, synthetic_author_id

log = logging.getLogger("expand")


def rss(tag: str) -> None:
    import resource
    cur = int(open("/proc/self/statm").read().split()[1]) * 4096 / 1e9
    log.info("mem %-22s now %.1f GB, peak %.1f GB", tag, cur, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
D = Path("data")
SRC = {"twitter100m": D / "tracks/data/sources/twitter100m/posts/*.parquet",
       "zenodo_brands": D / "tracks/data/sources/zenodo_brands/posts.parquet",
       "nitter_2026_replies": D / "tracks/data/sources/nitter_2026_replies/posts/*.parquet"}
CUT = date_to_ms("2026-04-23")
LOW = pl.col("author_handle").str.to_lowercase()


def handle_ids(rows: pl.DataFrame, filt: pl.DataFrame) -> pl.DataFrame:
    """handle -> the author id the existing corpus uses for it (training rows first, then the filtered corpus)."""
    a = rows.group_by("handle", "author_id").len().with_columns(pl.lit(0).alias("p"))
    b = filt.group_by("handle", "author_id").len().with_columns(pl.lit(1).alias("p"))
    return pl.concat([a, b]).sort(["handle", "p", "len"], descending=[False, False, True]).unique("handle", keep="first").select("handle", pl.col("author_id").alias("_cid"))


def author_stage(ctx: pl.DataFrame, activity: pl.DataFrame | None, cfg: Config) -> dict:
    """pipeline.author_stage on an in-memory frame (no follow graph, no donor table)."""
    accounts = unify.merge_accounts([], ctx)
    posts = authors.followers_at_post(ctx, accounts)
    cp = dedup.author_copypasta(posts)
    return {"posts": posts.select("post_id", "followers_at_post"), "trail": authors.trailing_features(posts, window=cfg.labels.trailing_window), "cp": cp,
            "atab": authors.author_table(posts, accounts, activity, cp, None, None), "zones": audience.estimate_audience_zone(posts)}


def feature_stage(sel: pl.DataFrame, st: dict, nd: pl.DataFrame | None) -> pl.DataFrame:
    """pipeline.feature_stage: the same joins and functions, on the selected posts."""
    at = st["atab"]
    at = at.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in ("reply_share", "mutual_share_est") if c not in at.columns])
    assert at["author_id"].n_unique() == at.height
    df = (sel.join(st["posts"].select("post_id", "followers_at_post"), on="post_id", how="left")
          .join(st["trail"], on="post_id", how="left")
          .join(st["cp"].select("post_id", "is_self_repeat"), on="post_id", how="left")
          .join(at.select("author_id", "verified", "mutual_share_est", "reply_share", "copypasta_events"), on="author_id", how="left"))
    if nd is not None:
        df = df.join(nd, on="post_id", how="left")
    dbg = bool(__import__("os").environ.get("EXPAND_DEBUG"))
    if dbg:
        for nm, fr, key in (("posts", st["posts"], "post_id"), ("trail", st["trail"], "post_id"), ("cp", st["cp"], "post_id"), ("atab", at, "author_id"), ("nd", nd, "post_id"), ("sel", sel, "post_id")):
            log.info("unique %s: %d rows, %d keys", nm, fr.height, fr[key].n_unique())
        log.info("chunk %d rows after joins (from %d), authors %s, max text %d", df.height, sel.height, sel["author_handle"].unique().to_list()[:50], sel["text"].str.len_chars().max())
    df = derive.author_columns(df)
    df = df.join(content.content_features(df), on="post_id", how="left")
    if dbg: rss("content")
    df = df.join(text_structure.admitted(df), on="post_id", how="left")
    if dbg: rss("text_structure")
    df = df.join(audience.audience_time_features(df, st["zones"]), on="post_id", how="left")
    if dbg: rss("audience")
    return df.join(suppression.url_features(df, suppression.load_blocklists(None)), on="post_id", how="left")


def fit_mutual(rows: pl.DataFrame):
    """The pipeline estimates the mutual-follow share from followers and following with a fit on archive donors. The donor
    table is not in the shared data, so the same functional form is refitted on the estimates stored in the training rows."""
    if rows["mutual_share_est"].null_count() == rows.height:      # the existing rows carry no estimate (no donor table in that run): keep it null
        return None
    a = rows.filter(pl.col("mutual_share_est").is_not_null() & (pl.col("followers_at_post") > 0) & (pl.col("author_following") > 0)).sort("created_at_ms").unique("author_id", keep="last")
    f, g, y = a["followers_at_post"].to_numpy().astype(float), a["author_following"].to_numpy().astype(float), a["mutual_share_est"].to_numpy()
    bound = np.minimum(f, g) / np.maximum(f, 1)
    fr = np.clip(y / np.maximum(bound, 1e-9), 1e-3, 1 - 1e-3)
    X = np.column_stack([np.ones(len(f)), np.log1p(f), np.log1p(g)])
    beta = np.linalg.lstsq(X, np.log(fr / (1 - fr)), rcond=None)[0]
    pred = 1 / (1 + np.exp(-X @ beta)) * bound
    log.info("mutual share refit on %d accounts: median abs error %.4f, correlation %.3f", len(f), float(np.median(np.abs(pred - y))), float(np.corrcoef(pred, y)[0, 1]))
    return beta


def apply_mutual(df: pl.DataFrame, beta) -> pl.DataFrame:
    if beta is None:
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("_mut"))
    f, g = pl.col("followers_at_post").cast(pl.Float64), pl.col("author_following").cast(pl.Float64)
    z = beta[0] + beta[1] * f.log1p() + beta[2] * g.log1p()
    est = (1 / (1 + (-z).exp())) * pl.min_horizontal(f, g) / f.clip(lower_bound=1)
    return df.with_columns(pl.when((f > 0) & (g > 0)).then(est).otherwise(None).alias("_mut"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--min-posts", type=int, default=20)
    ap.add_argument("--out", default="data/expand")
    ap.add_argument("--check", type=int, default=0, help="replication check: rebuild the features of this many existing accounts and compare with rows.parquet")
    ap.add_argument("--sources", default="twitter100m,zenodo_brands,nitter_2026_replies")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(D / "config.yaml")
    t0 = time.time()
    rows_schema = pl.scan_parquet(D / "work/dataset/rows.parquet").collect_schema()
    need = ["post_id", "author_id", "author_handle", "created_at_ms", "in_audience", "acct_tech_share", "acct_off_share", "verified", "reply_share",
            "mutual_share_est", "split", "followers_at_post", "author_following"]
    rows = pl.read_parquet(D / "work/dataset/rows.parquet", columns=None if args.check else need)     # the wide table only for the check
    filt_n = pl.scan_parquet(D / "work/filtered/posts.parquet").select("post_id", "author_id", LOW.alias("handle")).collect()
    rows_n = rows.select("post_id", "author_id", LOW.alias("handle"))
    hid = handle_ids(rows_n, filt_n)
    beta = fit_mutual(rows)

    if args.check:
        # The same code on accounts whose answer is known: their filtered posts in, the stored training rows out.
        aid = rows.filter(pl.col("author_id") > 0).select("author_id").unique().sort("author_id").sample(args.check, seed=3)["author_id"]
        ctx = pl.scan_parquet(D / "work/filtered/posts.parquet").filter(pl.col("author_id").is_in(aid.implode())).collect()
        st = author_stage(ctx, None, cfg)
        ref = rows.filter(pl.col("author_id").is_in(aid.implode()))
        mine = labels.build(feature_stage(ctx.join(ref.select("post_id"), on="post_id", how="semi"), st, None), cfg.labels)
        j = ref.join(mine, on="post_id", suffix="_m")
        rep = {}
        for c in ref.columns:
            if c + "_m" not in j.columns or c in ("text", "media", "urls", "author_id"):
                continue
            a, b = j[c], j[c + "_m"]
            if a.dtype.is_numeric() or a.dtype == pl.Boolean:
                x, y = a.cast(pl.Float64).to_numpy(), b.cast(pl.Float64).to_numpy()
                both = ~(np.isnan(x) | np.isnan(y))
                rep[c] = {"null_ref": float(np.isnan(x).mean()), "null_mine": float(np.isnan(y).mean()),
                          "equal": float(np.isclose(x[both], y[both], rtol=1e-4, atol=1e-6).mean()) if both.any() else None}
            else:
                rep[c] = {"equal": float((a.cast(pl.String).fill_null("") == b.cast(pl.String).fill_null("")).mean())}
        rep["_missing_in_mine"] = [c for c in ref.columns if c not in mine.columns]
        rep["_rows"] = j.height
        (out / "replication_check.json").write_text(json.dumps(rep, indent=1))
        bad = {k: v for k, v in rep.items() if isinstance(v, dict) and ((v.get("equal") or 1) < 0.995 or abs(v.get("null_ref", 0) - v.get("null_mine", 0)) > 0.005)}
        print(json.dumps(bad, indent=1)); print("missing in mine:", rep["_missing_in_mine"], "rows", j.height)
        return

    S = pl.read_parquet(D / "tracks/data/startup_accounts.parquet").select("handle", pl.col("kind").alias("acct_kind"), "kind_from", "in_brand_set", pl.col("author_id").alias("_sid"))
    bench = pl.read_parquet(D / "tracks/pairs/benchmark_pairs.parquet")
    cal_ids = bench.filter(pl.col("part") == "calibration")["author_id"].unique()
    cal_handles = filt_n.filter(pl.col("author_id").is_in(cal_ids.implode()))["handle"].unique()
    S = S.filter(~pl.col("handle").is_in(cal_handles.implode()))
    log.info("startup accounts %d after removing %d handles of benchmark calibration accounts", S.height, cal_handles.len())
    counts: dict = {"cap": args.cap, "startup_accounts_considered": S.height, "calibration_handles_excluded": int(cal_handles.len())}

    # Accounts first, on narrow columns; then only their rows in full.
    frames, act = [], []
    for name in args.sources.split(","):
        lf = pl.scan_parquet(SRC[name]).with_columns(LOW.alias("handle")).filter(pl.col("handle").is_in(S["handle"].implode()))
        nar = lf.select("handle", "kind", "created_at_ms").collect()
        act.append(nar.group_by("handle").agg(pl.len().alias("rows"), (pl.col("kind") == "reply").sum().alias("replies"), (pl.col("kind") == "repost").sum().alias("reposts"),
                                              pl.col("kind").is_in(["original", "quote"]).sum().alias("originals"), pl.col("created_at_ms").min().alias("first_ms"), pl.col("created_at_ms").max().alias("last_ms")))
        counts[f"read_{name}"] = {"rows": nar.height, "accounts": nar["handle"].n_unique()}
        log.info("%s: %d rows of %d accounts", name, nar.height, nar["handle"].n_unique())
        del nar
        # Full width only for rows that can pass the kind and recency filters (applied again below, with the log).
        d = lf.filter(pl.col("kind").is_in(cfg.filters.keep_kinds) & (pl.col("created_at_ms") >= date_to_ms("2020-01-01"))).collect()
        if name == "nitter_2026_replies":      # this conversion marks every row assumed_mature; only rows before 2026-05-01 carry a snapshot time
            d = d.with_columns(pl.when(pl.col("snapshot_at_ms").is_null()).then(pl.lit("unknown")).otherwise(pl.col("snapshot_kind")).alias("snapshot_kind"))
        d = d.with_columns(pl.lit(name).alias("expand_src"))
        frames.append(d)
    new = pl.concat(frames, how="diagonal_relaxed")
    del frames
    # One author id per handle: the id the existing corpus uses, else the source's real id, else a synthetic one.
    real = new.filter(pl.col("author_id") > 0).group_by("handle").agg(pl.col("author_id").mode().first().alias("_rid"))
    idmap = new.select("handle").unique().join(hid, on="handle", how="left").join(real, on="handle", how="left")
    idmap = idmap.with_columns(pl.coalesce("_cid", "_rid", pl.col("handle").map_elements(synthetic_author_id, return_dtype=pl.Int64)).alias("_aid"))
    new = new.join(idmap.select("handle", "_aid"), on="handle").with_columns(pl.col("_aid").alias("author_id")).drop("_aid")
    activity = pl.concat(act).group_by("handle").agg(pl.col("rows", "replies", "reposts", "originals").sum(), pl.col("first_ms").min(), pl.col("last_ms").max()) \
        .join(idmap.select("handle", pl.col("_aid").alias("author_id")), on="handle").drop("handle") \
        .group_by("author_id").agg(pl.col("rows", "replies", "reposts", "originals").sum(), pl.col("first_ms").min(), pl.col("last_ms").max())   # two handles can be one account
    src_of = new.select("post_id", "expand_src").unique("post_id", keep="first")
    handle_of = new.select("author_id", "handle").unique("author_id")

    rss("read")
    posts = unify.merge_posts([new.select(list(POSTS_SCHEMA))])
    del new
    n0 = posts.height
    posts = posts.join(rows_n.select("post_id"), on="post_id", how="anti")
    counts["unique_posts"] = n0
    counts["already_training_rows"] = n0 - posts.height
    posts = posts.with_columns(pl.when(pl.col("source") == "twitterapi_io").then(pl.lit("zenodo_brands")).otherwise(pl.col("source")).alias("source"))

    rss("merged")
    flog = filters.FilterLog()
    posts = flog.apply("validity", posts, lambda d: filters.validity(d, cfg.filters), exempt_analytics=False)
    posts = flog.apply("kind", posts, lambda d: filters.kind(d, cfg.filters))
    posts = flog.apply("maturity", posts, lambda d: filters.maturity(d, cfg.filters, None), exempt_analytics=False)
    posts = flog.apply("recency", posts, lambda d: filters.recency(d, cfg.filters))
    lc = out / "lang_cache.parquet"
    posts = filters.resolve_language(posts, cfg.filters, pl.read_parquet(lc) if lc.exists() else None)
    posts.select("post_id", "lang").write_parquet(lc)
    posts = flog.apply("language", posts, lambda d: filters.keep_languages(d, cfg.filters))
    in_rows = rows_n["author_id"].unique()
    posts = flog.apply("volume", posts, lambda d: d.filter((pl.len().over("author_id") >= args.min_posts) | pl.col("author_id").is_in(in_rows.implode())), exempt_analytics=False)
    log.info("filters done: %d posts, %d accounts, %.0fs", posts.height, posts["author_id"].n_unique(), time.time() - t0)

    # Author stage over the account's whole known history: the new posts plus its posts in the existing filtered corpus.
    rss("filtered")
    aid = posts["author_id"].unique()
    old = pl.scan_parquet(D / "work/filtered/posts.parquet").filter(pl.col("author_id").is_in(aid.implode())).collect()
    old = old.join(posts.select("post_id"), on="post_id", how="anti")
    ctx = pl.concat([posts.select(list(POSTS_SCHEMA)), conform(old)], how="vertical_relaxed")
    counts["context_posts_from_existing_corpus"] = old.height
    del old
    rss("ctx")
    st = author_stage(ctx, activity, cfg)
    del ctx
    rss("authors")
    bad = filters.hygiene_bad_accounts(st["atab"], cfg.hygiene)
    st["atab"].join(handle_of, on="author_id", how="left").write_parquet(out / "author_table.parquet")
    bad = bad.filter(~bad.is_in(in_rows.implode()))          # accounts already in training passed the pipeline's own hygiene
    counts["hygiene_accounts_dropped"] = int(bad.len())
    posts = flog.apply("hygiene", posts, lambda d: d.filter(~pl.col("author_id").is_in(bad.implode())), exempt_analytics=False)
    posts = flog.apply("select_cap", posts, lambda d: d.with_columns(pl.col("created_at_ms").rank("ordinal", descending=True).over("author_id").alias("_r")).filter(pl.col("_r") <= args.cap).drop("_r"), exempt_analytics=False)
    flog.frame().write_csv(out / "filter_log.csv")
    print(flog.frame())

    # Account-level flags. Existing accounts keep theirs; brand accounts are general accounts (a separate domain, as the
    # data track advises); every other startup account is audience by definition, as seed handles are in filters.account_selection.
    from xviral.text.tech import is_off_audience_post, is_tech_post
    ts_ = posts.select("author_id", "text").group_by("author_id").agg(is_tech_post(strong=True).mean().alias("acct_tech_share"), is_off_audience_post().mean().alias("acct_off_share"))
    ex = rows.group_by("author_id").agg(pl.col("in_audience").first().alias("_aud"), pl.col("acct_tech_share").first().alias("_ts"), pl.col("acct_off_share").first().alias("_os"),
                                        pl.col("verified").drop_nulls().last().alias("_ver"), pl.col("reply_share").drop_nulls().last().alias("_rs"),
                                        pl.col("mutual_share_est").drop_nulls().last().alias("_ms"), (pl.col("split") == "val").any().alias("_val"))
    meta = handle_of.join(S, on="handle", how="left")
    rss("pre-neardup")
    nd = dedup.near_duplicate_clusters(posts)
    rss("neardup")
    if "sources_all" not in posts.columns:
        posts = posts.with_columns(pl.col("source").alias("sources_all"))
    # In chunks of accounts: every function of the feature stage works per post or per account, and the wide frame of
    # 300k posts with their text-structure intermediates does not fit next to anything else on a 15 GB machine.
    ids_all = posts["author_id"].unique().sort()
    parts = []
    for k in range(int(__import__('os').environ.get('EXPAND_START', 0)), ids_all.len(), 40):
        part = posts.filter(pl.col("author_id").is_in(ids_all.slice(k, 40).implode()))
        parts.append(feature_stage(part, st, nd))
        rss(f"features {k}")
    df = pl.concat(parts, how="vertical_relaxed")
    assert df.height == posts.height and df["post_id"].n_unique() == df.height
    del parts, posts
    df = apply_mutual(df, beta).join(ts_, on="author_id", how="left").join(ex, on="author_id", how="left").join(meta.select("author_id", "acct_kind", "kind_from", "in_brand_set"), on="author_id", how="left") \
        .join(src_of, on="post_id", how="left")
    df = df.with_columns(
        pl.coalesce("_aud", ~pl.col("in_brand_set").fill_null(False)).alias("in_audience"),
        pl.coalesce("_ts", "acct_tech_share").alias("acct_tech_share"), pl.coalesce("_os", "acct_off_share").alias("acct_off_share"),
        pl.coalesce("_ver", "verified").alias("verified"), pl.coalesce("_rs", "reply_share").alias("reply_share"),
        pl.coalesce("_ms", "_mut").alias("mutual_share_est"))
    df = labels.build(df, cfg.labels)
    # Split: the same cutoff; an existing account stays on its side of train/validation; a tenth of the new accounts is validation.
    newval = (pl.col("author_id").hash(seed=cfg.splits.seed) % 10) == 0
    df = df.with_columns(pl.when(pl.col("created_at_ms") >= CUT).then(pl.lit("test")).when(pl.coalesce("_val", newval)).then(pl.lit("val")).otherwise(pl.lit("train")).alias("split"),
                         pl.col("_aud").is_not_null().alias("acct_in_original"))
    df = df.drop("_aud", "_ts", "_os", "_ver", "_rs", "_ms", "_mut", "_val")
    miss = [c for c in rows_schema.names() if c not in df.columns]
    log.info("columns of rows.parquet absent from the new rows (left null): %s", miss)
    df = df.with_columns([pl.lit(None, dtype=rows_schema[c]).alias(c) for c in miss])
    extra = ["expand_src", "acct_kind", "kind_from", "in_brand_set", "acct_in_original"]
    df = df.select([pl.col(c).cast(rows_schema[c], strict=False) for c in rows_schema.names()] + extra)
    assert df["post_id"].n_unique() == df.height and df.join(rows_n, on="post_id", how="semi").height == 0
    assert df.filter((pl.col("split") != "test") & (pl.col("created_at_ms") >= CUT)).height == 0
    assert df.filter(pl.col("author_id").is_in(cal_ids.implode())).height == 0
    df.write_parquet(out / "new_rows.parquet", compression="zstd")
    counts.update(new_rows=df.height, new_accounts=df["author_id"].n_unique(), seconds=round(time.time() - t0))
    counts["by_source"] = {r["expand_src"]: {k: v for k, v in r.items() if k != "expand_src"} for r in df.group_by("expand_src").agg(
        pl.len().alias("rows"), pl.col("author_id").n_unique().alias("accounts"), pl.col("views").is_not_null().sum().alias("with_views"),
        (pl.col("split") == "train").sum().alias("train"), (pl.col("split") == "val").sum().alias("val"), (pl.col("split") == "test").sum().alias("test")).iter_rows(named=True)}
    (out / "build_counts.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
