#!/usr/bin/env python3
"""Selective track: one feature row per pair.   python scripts/selective/sel_feats.py dev launch_pre [bench launch_test]
Writes data/sel/feat_<set>_<ensemble>.parquet for the calling ensembles ft10, start10 and base3."""
import sys
import numpy as np, polars as pl
sys.path.insert(0, "scripts/selective")
import sel_lib as L
from xviral import selective as S

SRC = {"bench": "data/tracks/pairs/benchmark_pairs.parquet", "dev": "data/tracks/replicate/dev_pairs.parquet"}
texts = None
for name in sys.argv[1:]:
    D = L.load_set(name)
    # post type from the text rules, added to each version's feature row (the product computes the same from the draft text)
    ids = [r["real_post_id"] for r in D["rows"].values()]
    t = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id", "text").filter(pl.col("post_id").is_in(ids)).collect()
    from xviral.text import launch_labels as LL
    tt = LL.add_types(t).select("post_id", "ptype_raw")
    pt = dict(zip(tt["post_id"].to_list(), tt["ptype_raw"].to_list()))
    for r in D["rows"].values():
        k = pt.get(r["real_post_id"], "other")
        r["pt_launch"], r["pt_any"] = float(k in ("launch", "open_source")), float(k != "other")
    kind = "launch" if name.startswith("launch") else "same_topic"
    pairs = L.launch_pairs(name) if kind == "launch" else L.same_topic_pairs(name, SRC[name])
    uidx = {int(u): i for i, u in enumerate(D["unit_id"])}
    pairs = pairs.with_columns(pl.col("ua").replace_strict(uidx, default=None), pl.col("ub").replace_strict(uidx, default=None)).drop_nulls(["ua", "ub"])
    for ens in ("ft10", "start10", "base3"):
        F = L.pair_rows(D, pairs, kind, ens)
        F = F.filter(pl.col("d").abs() >= np.log(1.25))
        F.write_parquet(f"data/sel/feat_{name}_{ens}.parquet")
        print(name, ens, F.height, "pairs,", F["author_id"].n_unique(), "accounts,", len([c for c in F.columns if c.startswith("f_")]), "features", flush=True)
