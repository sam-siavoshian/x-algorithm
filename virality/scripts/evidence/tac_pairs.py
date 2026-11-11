#!/usr/bin/env python3
# Usage: python scripts/evidence/tac_pairs.py <dir>   where <dir> holds work/ (pipeline output); results go to <dir>/evidence/
"""Topic- and author-controlled pairs (after Tan, Lee, Pang 2014): same account, same link or same quoted post, different wording."""
import sys
import numpy as np, polars as pl
S = sys.argv[1]
import os
os.makedirs(f"{S}/evidence", exist_ok=True)
cols = ["post_id", "author_id", "author_handle", "created_at_ms", "text", "urls", "quoted_post_id", "likes", "views", "source", "media"]
d = pl.scan_parquet(f"{S}/work/filtered/posts.parquet").select(cols).collect()
print("posts", d.height)
u = d.select("post_id", pl.col("urls").list.eval(pl.element().struct.field("expanded")).alias("u")).explode("u").filter(pl.col("u").is_not_null())
u = u.with_columns(pl.col("u").str.to_lowercase().str.replace(r"^https?://(www\.)?", "").str.replace(r"[?#].*$", "").str.strip_chars_end("/").alias("key"))
u = u.filter(~pl.col("key").str.contains(r"^(x\.com|twitter\.com|t\.co)") & (pl.col("key").str.len_chars() > 12) & pl.col("key").str.contains("/"))
u = u.unique(["post_id", "key"]).select("post_id", pl.concat_str([pl.lit("url:"), pl.col("key")]).alias("key"))
q = d.filter(pl.col("quoted_post_id").is_not_null()).select("post_id", pl.concat_str([pl.lit("quote:"), pl.col("quoted_post_id").cast(pl.String)]).alias("key"))
k = pl.concat([u, q]).join(d.select("post_id", "author_id", "created_at_ms", "text", "likes", "views", "source", (pl.col("media").list.len() > 0).alias("has_media")), on="post_id")
norm = pl.col("text").fill_null("").str.to_lowercase().str.replace_all(r"https?://\S+", "").str.replace_all(r"\s+", " ").str.strip_chars()
k = k.with_columns(norm.alias("nt"))
g = k.group_by("author_id", "key").agg(pl.len().alias("n"), pl.col("nt").n_unique().alias("wordings")).filter((pl.col("n") >= 2) & (pl.col("wordings") >= 2))
print("author+topic groups with at least two wordings:", g.height, "posts in them:", int(g["n"].sum()), "accounts:", g["author_id"].n_unique())
print(g.with_columns(pl.col("key").str.split(":").list.first().alias("kind")).group_by("kind").agg(pl.len().alias("groups"), pl.col("n").sum().alias("posts")))
kk = k.join(g.select("author_id", "key"), on=["author_id", "key"], how="semi").sort("author_id", "key", "created_at_ms")
pairs = kk.join(kk, on=["author_id", "key"], suffix="_b").filter((pl.col("post_id") < pl.col("post_id_b")) & (pl.col("nt") != pl.col("nt_b")))
pairs = pairs.with_columns(((pl.col("created_at_ms_b") - pl.col("created_at_ms")).abs() / 3_600_000).alias("gap_h"))
print("pairs:", pairs.height, "within 30 days:", pairs.filter(pl.col("gap_h") <= 720).height, "within 12 hours:", pairs.filter(pl.col("gap_h") <= 12).height)
p30 = pairs.filter(pl.col("gap_h") <= 720)
print("  with views on both:", p30.filter(pl.col("views").is_not_null() & pl.col("views_b").is_not_null()).height,
      " likes differ by 2x or more:", p30.filter(((pl.col("likes") + 1) / (pl.col("likes_b") + 1)).log().abs() >= np.log(2)).height)
p30.write_parquet(f"{S}/evidence/tac_pairs.parquet")
ex = p30.filter(pl.col("key").str.starts_with("url:") & (pl.col("likes") + pl.col("likes_b") > 200)).sample(6, seed=4)
for r in ex.iter_rows(named=True):
    print("---", r["key"][:70]); print("  A", r["likes"], "|", r["text"][:150].replace("\n", " ")); print("  B", r["likes_b"], "|", r["text_b"][:150].replace("\n", " "))
