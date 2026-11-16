#!/usr/bin/env python3
"""A copy of the post-type table whose training weight (w_row) is multiplied for the account's own announcements.

    python scripts/posttype/weight_table.py --weight 10 [--table data/work/model/table_pt] [--out data/work/model/table_pt_w10]
Only rows the rules type as an own announcement are touched (launch, open source, funding, feature, milestone, hiring, demo).
"""
import argparse, os, numpy as np, polars as pl
ap = argparse.ArgumentParser(); ap.add_argument("--weight", type=float, required=True); ap.add_argument("--table", default="data/work/model/table_pt"); ap.add_argument("--out", default=None)
args = ap.parse_args(); out = args.out or f"{args.table}_w{args.weight:g}"
os.makedirs(out, exist_ok=True)
for f in os.listdir(args.table):
    if f != "w_row.npy" and not os.path.exists(os.path.join(out, f)): os.symlink(os.path.realpath(os.path.join(args.table, f)), os.path.join(out, f))
ty = pl.read_parquet(f"{args.table}/post_type.parquet")["pt_type"].to_numpy(); w = np.load(f"{args.table}/w_row.npy")
m = ty != "other"; w2 = np.where(m, w * args.weight, w).astype(np.float32)
np.save(f"{out}/w_row.npy", w2); print(out, "rows upweighted:", int(m.sum()), "share of weight before", round(float(w[m].sum() / w.sum()), 4), "after", round(float(w2[m].sum() / w2.sum()), 4))
