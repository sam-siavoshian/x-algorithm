#!/usr/bin/env python3
"""Are outcome levels comparable between the new sources and the existing rows?  Writes data/expand/source_levels.json.

1. the same post read by two sources (different snapshot dates): ratio of the like counts
2. the same account on both sides of a source boundary: median likes of its last posts in one source against its first in the other
3. likes per follower and views per follower by source and follower tier; likes per view by source
4. level by year inside twitter100m accounts (the period effect the model cannot see)
"""
import json
import numpy as np, polars as pl
D = "data/"
def q(s, p): return float(s.quantile(p)) if s.len() else None
def ratio_stats(d, a, b):
    r = ((d[a] + 1).log10() - (d[b] + 1).log10())
    return {"posts": d.height, "median_log10_ratio": q(r, .5), "p10": q(r, .1), "p90": q(r, .9), "share_within_1.25x": float((r.abs() <= np.log10(1.25)).mean()) if d.height else None,
            "median_ratio": float(10 ** q(r, .5)) if d.height else None}
out = {}
rows = pl.read_parquet(D + "work/dataset/rows.parquet", columns=["post_id", "author_id", "author_handle", "created_at_ms", "source", "likes", "views", "followers_at_post", "follower_tier", "in_audience", "split"])
new = pl.read_parquet(D + "expand/new_rows.parquet", columns=["post_id", "author_id", "author_handle", "created_at_ms", "source", "likes", "views", "followers_at_post", "follower_tier", "expand_src", "acct_kind", "split", "acct_in_original"])
# 1. same post, two sources
filt = pl.scan_parquet(D + "work/filtered/posts.parquet").select("post_id", pl.col("likes").alias("likes_corpus"), pl.col("source").alias("src_corpus"), pl.col("snapshot_at_ms").alias("snap_corpus"), "created_at_ms").collect()
tw = pl.scan_parquet(D + "tracks/data/sources/twitter100m/posts/*.parquet").filter(pl.col("kind").is_in(["original", "quote"])).select("post_id", pl.col("likes").alias("likes_tw"), pl.col("author_handle").str.to_lowercase().alias("h")).collect()
zb = pl.scan_parquet(D + "tracks/data/sources/zenodo_brands/posts.parquet").select("post_id", pl.col("likes").alias("likes_zb"), "created_at_ms").collect()
j = tw.join(filt, on="post_id")
out["same_post"] = {"twitter100m_vs_existing_corpus": {s: ratio_stats(g, "likes_tw", "likes_corpus") | {"note": "log10(twitter100m likes / corpus likes)"} for (s,), g in j.group_by("src_corpus")}}
j["created_at_ms"].describe()
age = j.with_columns(((1682035200000 - pl.col("created_at_ms")) / 86_400_000).alias("age_d"))
out["same_post"]["twitter100m_vs_existing_corpus_by_age_at_twitter100m_snapshot"] = {lab: ratio_stats(age.filter((pl.col("age_d") >= lo) & (pl.col("age_d") < hi)), "likes_tw", "likes_corpus")
                                                                              for lab, lo, hi in (("under 3 days", -1, 3), ("3 to 14 days", 3, 14), ("14 to 60 days", 14, 60), ("60 days and over", 60, 1e9))}
j2 = tw.join(zb, on="post_id")
out["same_post"]["twitter100m_vs_zenodo_brands"] = ratio_stats(j2, "likes_tw", "likes_zb") | {"note": "log10(likes read April 2023 / likes read December 2025), same posts"}
j3 = zb.join(filt, on="post_id")
out["same_post"]["zenodo_brands_vs_existing_corpus"] = ratio_stats(j3, "likes_zb", "likes_corpus")
# 2. same account across the boundary: last 30 posts in twitter100m (to April 2023) against first 30 in the existing rows (2023 on)
both = new.filter((pl.col("expand_src") == "twitter100m") & pl.col("acct_in_original"))
a = both.sort("created_at_ms").group_by("author_id").agg(pl.col("likes").tail(30).median().alias("m_new"), pl.len().alias("n_new"), pl.col("created_at_ms").max().alias("t_new"))
b = rows.sort("created_at_ms").group_by("author_id").agg(pl.col("likes").head(30).median().alias("m_old"), pl.len().alias("n_old"), pl.col("created_at_ms").min().alias("t_old"), pl.col("source").first().alias("src_old"))
ab = a.join(b, on="author_id").filter((pl.col("n_new") >= 10) & (pl.col("n_old") >= 10)).with_columns(((pl.col("t_old") - pl.col("t_new")) / 86_400_000).alias("gap_days"))
out["same_account_boundary"] = {"design": "accounts with rows on both sides; median likes of the last 30 twitter100m posts against the first 30 existing rows", "accounts": ab.height,
                                "all": ratio_stats(ab, "m_new", "m_old") | {"median_gap_days": q(ab["gap_days"], .5)},
                                "boundary_within_180_days": ratio_stats(ab.filter(pl.col("gap_days").abs() <= 180), "m_new", "m_old"),
                                "by_existing_source": {s: ratio_stats(g, "m_new", "m_old") for (s,), g in ab.group_by("src_old")}}
# 3. levels by source and tier
allr = pl.concat([rows.with_columns(pl.col("source").alias("src")).select("src", "likes", "views", "followers_at_post", "follower_tier", "created_at_ms", "author_id"),
                  new.with_columns(pl.col("expand_src").alias("src")).select("src", "likes", "views", "followers_at_post", "follower_tier", "created_at_ms", "author_id")])
allr = allr.with_columns(((pl.col("likes") + 1) / (pl.col("followers_at_post") + 1)).log10().alias("lpf"), ((pl.col("views") + 1) / (pl.col("followers_at_post") + 1)).log10().alias("vpf"),
                         ((pl.col("likes") + 1) / (pl.col("views") + 1)).log10().alias("lpv"), pl.from_epoch("created_at_ms", time_unit="ms").dt.year().alias("year"))
def lv(g):
    k = g.filter(pl.col("followers_at_post") > 0)
    return {"posts": g.height, "accounts": g["author_id"].n_unique(), "with_followers": k.height, "median_likes": q(g["likes"], .5), "median_log10_likes_per_follower": q(k["lpf"].drop_nulls(), .5),
            "with_views": int(g["views"].is_not_null().sum()), "median_views": q(g["views"].drop_nulls(), .5), "median_log10_views_per_follower": q(k["vpf"].drop_nulls(), .5), "median_log10_likes_per_view": q(g["lpv"].drop_nulls(), .5)}
out["levels_by_source"] = {s: lv(g) for (s,), g in allr.group_by("src")}
out["levels_by_source_and_tier"] = {f"{s} | {t}": lv(g) for (s, t), g in allr.filter(pl.col("followers_at_post") > 0).group_by("src", "follower_tier") if g.height >= 200}
out["levels_by_source_and_year"] = {f"{s} | {y}": lv(g) for (s, y), g in allr.group_by("src", "year") if g.height >= 500}
# 4. inside accounts over time, twitter100m: each account's yearly median likes over its own overall median
t = new.filter(pl.col("expand_src") == "twitter100m").with_columns(pl.from_epoch("created_at_ms", time_unit="ms").dt.year().alias("year"))
t = t.with_columns(((pl.col("likes") + 1).log10() - (pl.col("likes") + 1).log10().median().over("author_id")).alias("rel"))
out["twitter100m_within_account_by_year"] = {str(y): {"posts": g.height, "accounts": g["author_id"].n_unique(), "median_log10_likes_vs_account_median": q(g["rel"], .5)} for (y,), g in t.group_by("year")}
# last days before the snapshot: immature counts?
t2 = new.filter(pl.col("expand_src") == "twitter100m").with_columns(((1682035200000 - pl.col("created_at_ms")) / 86_400_000).alias("age_d"), ((pl.col("likes") + 1).log10() - (pl.col("likes") + 1).log10().median().over("author_id")).alias("rel"))
out["twitter100m_by_age_at_snapshot"] = {lab: {"posts": g.height, "median_log10_likes_vs_account_median": q(g["rel"], .5)} for lab, g in ((lab, t2.filter((pl.col("age_d") >= lo) & (pl.col("age_d") < hi))) for lab, lo, hi in (("3 to 7 days", 3, 7), ("7 to 30 days", 7, 30), ("30 to 180 days", 30, 180), ("180 days and over", 180, 1e9)))}
json.dump(out, open(D + "expand/source_levels.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
