#!/usr/bin/env python3
"""Which video posts to acquire: every video/GIF post of company and founder accounts (tracks/launch accounts_labeled),
plus video posts on either side of the two benchmarks. Writes data/video/targets.parquet."""
import polars as pl
A = pl.read_parquet("data/tracks/launch/accounts_labeled.parquet").filter(pl.col("akind") != "other").select("author_id", "akind")
L = pl.read_parquet("data/tracks/launch/launch_posts.parquet").select("post_id", "ptype")
d = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id", "author_id", "author_handle", "source", "created_at_ms", "likes", "media").collect()
d = d.with_columns(pl.col("media").list.eval(pl.element().struct.field("type").is_in(["video", "gif"])).list.any().fill_null(False).alias("v")).filter(pl.col("v"))
m = d.explode("media").unnest("media").filter(pl.col("type").is_in(["video", "gif"])).group_by("post_id", maintain_order=True).agg(pl.col("type").first().alias("mtype"), pl.col("url").first().alias("thumb_url"),
      pl.col("width").first().alias("meta_w"), pl.col("height").first().alias("meta_h"), pl.len().alias("n_videos"))
d = d.drop("media", "v").join(m, on="post_id").join(A, on="author_id", how="left").join(L, on="post_id", how="left").with_columns(pl.col("akind").fill_null("other"), pl.col("ptype").fill_null("other"))
B = pl.read_parquet("data/tracks/pairs/benchmark_pairs.parquet")
bids = set(B["post_id"].to_list()) | set(B["post_id_b"].to_list()) if "post_id_b" in B.columns else set()
d = d.with_columns(pl.col("post_id").is_in(list(bids)).alias("in_bench"))
ANN = ["launch", "open_source", "feature", "funding", "milestone", "hiring", "demo"]
d = d.with_columns(pl.when(pl.col("ptype").is_in(["launch", "open_source"])).then(0).when(pl.col("ptype").is_in(ANN)).then(1).when(pl.col("in_bench")).then(2).when(pl.col("akind") != "other").then(3).otherwise(9).alias("prio"))
t = d.filter(pl.col("prio") < 9).sort("prio", "post_id")
t.write_parquet("data/video/targets.parquet")
print(t.group_by("prio", "mtype", "source").len().sort("prio", "mtype"))
print(B.columns, B.height)
