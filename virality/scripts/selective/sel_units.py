#!/usr/bin/env python3
"""Selective track: the unit lists scored by sel_score.py.

    python scripts/selective/sel_units.py      -> data/sel/units_{bench,dev,launch_test,launch_pre}.parquet

bench: the 3,530 benchmark pairs (sorted as tac_serve_eval.py sorts them, so shard i/4 hits tracks/pairs/bench/cache_i.pkl).
dev: the replicate track's 2,805 development pairs (both posts before the cutoff).
launch_test: the launch track's 582 test-period launch posts, each alone at its own time.
launch_pre: launch-type posts created before the cutoff by an account with at least two of them (for pre-cutoff launch pairs).
"""
import datetime as dt
import polars as pl

CUT = int(dt.datetime(2026, 4, 23, tzinfo=dt.timezone.utc).timestamp() * 1000)
tms = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id", "created_at_ms").collect()
for nm, f in (("bench", "data/tracks/pairs/benchmark_pairs.parquet"), ("dev", "data/tracks/replicate/dev_pairs.parquet")):
    p = pl.read_parquet(f).select("post_id", "post_id_b", "author_id").join(tms, on="post_id").sort("post_id", "post_id_b")
    p = p.with_row_index("unit_id").select(pl.col("unit_id").cast(pl.Int64), "author_id", pl.col("created_at_ms").alias("when_ms"), pl.concat_list("post_id", "post_id_b").alias("post_ids"))
    p.write_parquet(f"data/sel/units_{nm}.parquet"); print(nm, p.height)
L = pl.read_parquet("data/tracks/launch/launch_posts.parquet")
S = pl.read_parquet("data/tracks/launch/launch_scores.parquet").select("post_id", "author_id").join(tms, on="post_id")
S.with_row_index("unit_id").select(pl.col("unit_id").cast(pl.Int64), "author_id", pl.col("created_at_ms").alias("when_ms"), pl.concat_list("post_id").alias("post_ids")).write_parquet("data/sel/units_launch_test.parquet")
print("launch_test", S.height)
P = L.filter(pl.col("ptype").is_in(["launch", "open_source"]) & (pl.col("created_at_ms") < CUT)).filter(pl.len().over("author_id") >= 2).select("post_id", "author_id", "created_at_ms").sort("post_id")
P.with_row_index("unit_id").select(pl.col("unit_id").cast(pl.Int64), "author_id", pl.col("created_at_ms").alias("when_ms"), pl.concat_list("post_id").alias("post_ids")).write_parquet("data/sel/units_launch_pre.parquet")
print("launch_pre", P.height, P["author_id"].n_unique())
