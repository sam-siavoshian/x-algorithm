#!/usr/bin/env python3
"""One table per pair set with every scored member's gap, from the serving-path scores of scripts/selective/sel_score.py.

    python scripts/combine/pair_sets.py --scores data/combine/sel/te3l --scores data/combine/sel/bge --out data/combine/pairs

Sets: dev (2,805 pre-cutoff same-topic development pairs), launchpre (3,655 pre-cutoff launch pairs), bench (the 3,530
benchmark pairs), launch (the 850 launch pairs after the cutoff). The pair lists, like-count gaps, times, split cells and
the current product's gap (s3) are the judge_product track's (tracks/judge_product/jp_pairs_*.parquet); this adds
g_<member> = log10(predicted median likes + 1) of post_id minus that of post_id_b, for every member found.
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np, polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--scores", action="append", required=True); ap.add_argument("--out", default="data/combine/pairs")
ap.add_argument("--jp", default="data/tracks/judge_product"); ap.add_argument("--units", default="data/tracks/selective")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
SETS = {"dev": "dev", "launchpre": "launch_pre", "bench": "bench", "launch": "launch_test"}
# accounts with rows in the training split of the original table: their pre-cutoff posts were trained on by every model
TRAINED = pl.read_parquet("data/work/dataset/rows.parquet", columns=["author_id", "split"]).filter(pl.col("split") == "train")["author_id"].unique().implode()
for name, sel in SETS.items():
    P = pl.read_parquet(f"{a.jp}/jp_pairs_{name}.parquet")
    U = pl.read_parquet(f"{a.units}/units_{sel}.parquet").sort("unit_id")
    two = U["post_ids"].list.len().max() == 2
    if two:
        key = {(r["post_ids"][0], r["post_ids"][1]): r["unit_id"] for r in U.iter_rows(named=True)}
        ua = np.array([key.get((x, y), -1) for x, y in zip(P["post_id"].to_list(), P["post_id_b"].to_list())])
        ub, ca, cb = ua, 0, 1
        flip = ua < 0                      # stored in the other order
        ua2 = np.array([key.get((y, x), -1) for x, y in zip(P["post_id"].to_list(), P["post_id_b"].to_list())])
        ua = np.where(flip, ua2, ua); ub = ua
    else:
        key = {r["post_ids"][0]: r["unit_id"] for r in U.iter_rows(named=True)}
        ua = np.array([key.get(x, -1) for x in P["post_id"].to_list()]); ub = np.array([key.get(x, -1) for x in P["post_id_b"].to_list()])
        flip = np.zeros(len(ua), bool); ca = cb = 0
    assert (ua >= 0).all() and (ub >= 0).all(), (name, int((ua < 0).sum()))
    cols = {}
    for root in a.scores:
        files = sorted(glob.glob(f"{root}/{sel}/shard_*.npz"))
        for group in sorted({os.path.basename(f).split(".")[0].split("__")[-1] if "__" in os.path.basename(f) else "" for f in files}):
            fs = [f for f in files if (("__" + group) in os.path.basename(f)) == bool(group)]
            zs = [np.load(f) for f in fs]
            uid = np.concatenate([z["unit_id"] for z in zs]); o = np.argsort(uid)
            Q = np.concatenate([z["Q"][:, :, :, 1] for z in zs], 1)[:, o]          # [members, units, cands] log10 median likes
            pos = {int(u): i for i, u in enumerate(uid[o])}
            ia, ib = np.array([pos[int(u)] for u in ua]), np.array([pos[int(u)] for u in ub])
            for m, nm in enumerate(str(x) for x in zs[0]["names"]):
                if two:
                    g = Q[m, ia, 0] - Q[m, ia, 1]
                    g = np.where(flip, -g, g)
                else:
                    g = Q[m, ia, 0] - Q[m, ib, 0]
                cols[f"g_{nm}"] = g.astype(np.float64)
    out = P.with_columns([pl.Series(k, v) for k, v in cols.items()] + [pl.col("author_id").is_in(TRAINED).alias("acct_in_train")])
    bad = {k: int(np.isnan(v).sum()) for k, v in cols.items() if np.isnan(v).any()}
    out.write_parquet(f"{a.out}/{name}.parquet")
    chk = {k: round(float(np.corrcoef(v[~np.isnan(v)], P["s3"].to_numpy()[~np.isnan(v)])[0, 1]), 3) for k, v in cols.items()}
    print(name, out.height, "members:", list(cols), "missing:", bad, "correlation with the current product's gap:", chk, flush=True)
