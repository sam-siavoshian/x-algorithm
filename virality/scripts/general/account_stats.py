#!/usr/bin/env python3
"""Per-account counts of usable posts in each converted source (narrow scan), with flags for accounts any model has seen.

    python scripts/general/account_stats.py        # writes data/general/account_stats.parquet
"""
import polars as pl
from xviral.util.core import date_to_ms
D = "data/tracks/data/sources/"
LOW = pl.col("author_handle").str.to_lowercase().alias("handle")
ok_kind = pl.col("kind").is_in(["original", "quote"])
out = []
for name, path, floor, ceil in (("twitter100m", D + "twitter100m/posts/*.parquet", "2020-01-01", "2023-04-18"),
                                ("nitter_2026_replies", D + "nitter_2026_replies/posts/*.parquet", "2023-01-01", "2026-05-01"),
                                ("zenodo_brands", D + "zenodo_brands/posts.parquet", "2023-01-01", "2025-12-07")):
    lf = pl.scan_parquet(path).filter(ok_kind & (pl.col("created_at_ms") >= date_to_ms(floor)) & (pl.col("created_at_ms") < date_to_ms(ceil)))
    en = pl.col("lang_src").is_null() | (pl.col("lang_src") == "en")
    g = lf.group_by(LOW).agg(pl.len().alias("n_posts"), en.sum().alias("n_en"), pl.col("views").is_not_null().sum().alias("n_views"),
                             pl.col("author_followers").max().alias("followers"), pl.col("likes").median().alias("med_likes"),
                             pl.col("created_at_ms").min().alias("first_ms"), pl.col("created_at_ms").max().alias("last_ms"),
                             (pl.col("created_at_ms") < date_to_ms("2026-04-23")).sum().alias("n_pre_cut")).collect()
    out.append(g.with_columns(pl.lit(name).alias("src")))
    print(name, g.height, g.filter(pl.col("n_posts") >= 20).height, flush=True)
st = pl.concat(out, how="vertical_relaxed")
rows = pl.read_parquet("data/work/dataset/rows.parquet", columns=["author_handle", "split"]).with_columns(LOW)
ex = pl.read_parquet("data/tracks/expand/rows/new_rows.parquet", columns=["author_handle", "split", "expand_src"]).with_columns(LOW)
filt = pl.scan_parquet("data/work/filtered/posts.parquet").select(LOW).unique().collect()
bench = pl.read_parquet("data/tracks/pairs/benchmark_pairs.parquet")
st = st.with_columns(pl.col("handle").is_in(rows["handle"].unique().implode()).alias("in_rows"),
                     pl.col("handle").is_in(ex.filter(pl.col("split") == "train")["handle"].unique().implode()).alias("in_expand_train"),
                     pl.col("handle").is_in(ex["handle"].unique().implode()).alias("in_expand"),
                     pl.col("handle").is_in(filt["handle"].implode()).alias("in_filtered"))
tw = pl.read_parquet(D + "twitter100m/accounts.parquet").select(pl.col("author_handle").str.to_lowercase().alias("handle"), pl.col("followers").alias("tw_followers"), "bio_kind", "tech_bio")
st = st.join(tw, on="handle", how="left")
st.write_parquet("data/general/account_stats.parquet")
print(bench.columns)
pl.Config.set_tbl_rows(60)
e = st.filter(pl.col("n_en") >= 30)
fol = pl.coalesce("followers", "tw_followers")
tier = pl.when(fol.is_null() | (fol <= 0)).then(pl.lit("unknown")).when(fol < 1000).then(pl.lit("a<1k")).when(fol < 5000).then(pl.lit("b1k-5k")).when(fol < 10000).then(pl.lit("c5k-10k")).when(fol < 100000).then(pl.lit("d10k-100k")).otherwise(pl.lit("e100k+"))
print(e.with_columns(tier.alias("tier")).group_by("src", "tier").agg(pl.len().alias("accts"), (~pl.col("in_rows") & ~pl.col("in_expand")).sum().alias("unseen"), (~pl.col("in_rows") & ~pl.col("in_expand") & ~pl.col("in_filtered")).sum().alias("unseen_not_in_corpus"),
      pl.col("n_en").median().alias("med_posts"), (pl.col("n_views") >= 30).sum().alias("with30views")).sort("src", "tier"))
