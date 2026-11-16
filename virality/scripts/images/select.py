#!/usr/bin/env python3
"""Choose which images to fetch, in priority order (first photo of each post only):
  0  own announcements of company and founder accounts            (tracks/launch table)
  1  own announcements of every other account
  2  posts on either side of the benchmark pairs and the launch test pairs
  3  other posts of company and founder accounts with a like baseline (within-account comparison)
  4  other training-table posts of accounts that have an announcement, newest first, at most 60 per account
  5  the remaining training-table posts with a photo, at most 25 per account
    python scripts/images/select.py     # writes data/images/selection.parquet
"""
import numpy as np, polars as pl
P = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id", "author_id", "created_at_ms", "media", "source").filter(pl.col("media").list.len() > 0).collect()
m = P.with_columns(pl.col("media").list.eval(pl.element().struct.field("type") == "photo").list.arg_max().alias("media_idx"),
                   pl.col("media").list.eval(pl.element().struct.field("type") == "photo").list.any().alias("hp"),
                   pl.col("media").list.eval(pl.element().struct.field("type").is_in(["video", "gif"])).list.any().alias("hv")).filter(pl.col("hp") & ~pl.col("hv"))
m = m.with_columns(pl.col("media").list.get(pl.col("media_idx")).alias("mm"), pl.col("media").list.len().alias("n_media")).with_columns(pl.col("mm").struct.field("url").alias("url"), pl.col("mm").struct.field("width").alias("width"), pl.col("mm").struct.field("height").alias("height")).drop("media", "mm", "hp", "hv")
m = m.filter(pl.col("url").str.contains("pbs.twimg.com/media/"))
print("posts with a photo and no video:", m.height)
T = pl.read_parquet("data/launch/table.parquet").select("post_id", "ptype", "akind", "lift_likes")
rows = pl.read_parquet("data/work/dataset/rows.parquet", columns=["post_id", "in_audience", "split"])
B = pl.read_parquet("data/tracks/pairs/benchmark_pairs.parquet"); L = pl.read_parquet("data/tracks/launch/launch_scores.parquet")
bench = set(B["post_id"].to_list()) | set(B["post_id_b"].to_list()) | set(L["post_id"].to_list())
m = m.join(T, on="post_id", how="left").join(rows, on="post_id", how="left").with_columns(pl.col("post_id").is_in(list(bench)).alias("bench"))
ann = pl.col("ptype").is_not_null() & (pl.col("ptype") != "other"); cf = pl.col("akind").is_not_null() & (pl.col("akind") != "other")
m = m.with_columns(pl.col("created_at_ms").rank("ordinal", descending=True).over("author_id").alias("rk"))
m = m.with_columns(pl.when(ann & cf).then(0).when(ann).then(1).when(pl.col("bench")).then(2).when(cf & pl.col("lift_likes").is_not_null()).then(3)
                   .when(pl.col("ptype").is_not_null() & pl.col("split").is_not_null() & (pl.col("rk") <= 60)).then(4).when(pl.col("split").is_not_null() & (pl.col("rk") <= 25)).then(5).otherwise(9).alias("priority"))
m = m.filter(pl.col("priority") < 9).with_columns(pl.int_range(pl.len()).shuffle(seed=0).alias("order"))
print(m.group_by("priority").agg(pl.len(), pl.col("author_id").n_unique().alias("accounts")).sort("priority"))
m.write_parquet("data/images/selection.parquet")
