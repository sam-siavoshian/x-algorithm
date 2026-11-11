#!/usr/bin/env python3
# Usage: python scripts/evidence/tac_order_control.py <dir>   where <dir> holds work/ (pipeline output); results go to <dir>/evidence/
"""Same pairs, controlling for order and time gap: restrict to pairs posted within 48 hours and check order balance."""
import sys
import numpy as np, polars as pl
S = sys.argv[1]
import os
os.makedirs(f"{S}/evidence", exist_ok=True)
p = pl.read_parquet(f"{S}/evidence/tac_pairs.parquet")
cnt = pl.concat([p.select("author_id", "key", "post_id"), p.select("author_id", "key", pl.col("post_id_b").alias("post_id"))]).unique().group_by("author_id", "key").len()
p = p.join(cnt.filter(pl.col("len") <= 4).select("author_id", "key"), on=["author_id", "key"], how="semi")
d = ((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log())
p = p.with_columns(d.alias("d"), (pl.col("created_at_ms") < pl.col("created_at_ms_b")).alias("a_first")).filter(pl.col("d").abs() >= np.log(1.25))
for name, q in (("all pairs within 30 days", p), ("within 48 hours", p.filter(pl.col("gap_h") <= 48)), ("more than 48 hours apart", p.filter(pl.col("gap_h") > 48))):
    m = q.filter(pl.col("has_media") != pl.col("has_media_b")).with_columns(
        pl.when(pl.col("has_media")).then(pl.col("d")).otherwise(-pl.col("d")).alias("gain"),
        (pl.col("has_media") == pl.col("a_first")).alias("media_first"))
    for lab, mm in (("media version posted first", m.filter(pl.col("media_first"))), ("media version posted second", m.filter(~pl.col("media_first")))):
        if mm.height:
            print(f"{name:<28} {lab:<28} media wins {100*(mm['gain']>0).mean():5.1f}%  pairs {mm.height}")
    first = q.with_columns(pl.when(pl.col("a_first")).then(pl.col("d")).otherwise(-pl.col("d")).alias("g"))
    print(f"{name:<28} first-posted version wins {100*(first['g']>0).mean():5.1f}%  pairs {first.height}")
