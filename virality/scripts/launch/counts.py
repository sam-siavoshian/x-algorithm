#!/usr/bin/env python3
"""Counts of company and founder accounts and of their own announcements: by type, by follower tier, with and without
view counts, in work/filtered/posts.parquet and in work/hygiene/posts.parquet. Writes data/launch/counts.json."""
import json, polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200)
T = pl.read_parquet("data/launch/table.parquet")
A = pl.read_parquet("data/launch/accounts.parquet")
hy = pl.read_parquet("data/work/hygiene/posts.parquet", columns=["post_id", "author_id"])
T = T.join(hy.select("post_id").with_columns(pl.lit(True).alias("in_hygiene")), on="post_id", how="left").with_columns(pl.col("in_hygiene").fill_null(False))
tier = pl.when(pl.col("f").is_null()).then(pl.lit("unknown")).when(pl.col("f") < 1000).then(pl.lit("<1k")).when(pl.col("f") < 10000).then(pl.lit("1k-10k")).when(pl.col("f") < 100000).then(pl.lit("10k-100k")).otherwise(pl.lit("100k+"))
A = A.with_columns(tier.alias("tier")).join(hy.select("author_id").unique().with_columns(pl.lit(True).alias("in_hygiene")), on="author_id", how="left").with_columns(pl.col("in_hygiene").fill_null(False))
out = {}
acc = A.filter(pl.col("akind") != "other").group_by("akind", "tier").agg(pl.len().alias("accounts"), pl.col("in_hygiene").sum().alias("accounts_in_hygiene"), (pl.col("nv") > 0).sum().alias("accounts_with_view_counts"), pl.col("n").sum().alias("posts")).sort("akind", "tier")
print(acc); out["accounts_by_kind_and_tier"] = acc.to_dicts()
out["accounts"] = A.group_by("akind").agg(pl.len().alias("accounts"), pl.col("in_hygiene").sum().alias("in_hygiene"), (pl.col("nv") > 0).sum().alias("with_view_counts")).to_dicts(); print(out["accounts"])
P = T.filter(pl.col("ptype") != "other")
g = lambda *k: P.group_by(*k).agg(pl.len().alias("posts"), pl.col("author_id").n_unique().alias("accounts"), pl.col("views").is_not_null().sum().alias("with_views"), pl.col("views").is_null().sum().alias("without_views"),
                                 pl.col("lift_likes").is_not_null().sum().alias("with_likes_baseline"), pl.col("lift_views").is_not_null().sum().alias("with_views_baseline"), pl.col("in_hygiene").sum().alias("in_hygiene")).sort(*k)
for name, keys in (("by_account_kind_and_type", ("akind", "ptype")), ("by_type", ("ptype",)), ("company_founder_by_type_and_tier", ("ptype", "tier"))):
    t = g(*keys) if name != "company_founder_by_type_and_tier" else P.filter(pl.col("akind") != "other").group_by(*keys).agg(pl.len().alias("posts"), pl.col("author_id").n_unique().alias("accounts"), pl.col("views").is_not_null().sum().alias("with_views"),
            pl.col("views").is_null().sum().alias("without_views"), pl.col("lift_likes").is_not_null().sum().alias("with_likes_baseline"), pl.col("in_hygiene").sum().alias("in_hygiene")).sort(*keys)
    print(name); print(t); out[name] = t.to_dicts()
json.dump(out, open("data/launch/counts.json", "w"), indent=1)
