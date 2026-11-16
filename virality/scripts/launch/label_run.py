import sys, polars as pl
sys.path.insert(0, "scripts/launch")
from labels import add_types, ORDER
d=pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id","author_id","text","media").collect()
d=add_types(d).with_columns(pl.col("media").list.eval(pl.element().struct.field("type").is_in(["video","animated_gif","gif"])).list.any().fill_null(False).alias("has_video"))
print(d.group_by("ptype_raw").agg(pl.len(), pl.col("techy").mean(), pl.col("fp_head").mean(), pl.col("author_id").n_unique()).sort("len",descending=True))
d.select("post_id","author_id","ptype_raw","techy","fp_head","opener","reprint","at_start","has_video",*["m_"+k for k in ORDER]).write_parquet("data/launch/ptype_raw.parquet")
