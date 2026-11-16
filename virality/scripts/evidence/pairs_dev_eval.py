#!/usr/bin/env python3
"""Every checkpoint on the same held-out controlled pairs (the dev accounts of train_pairs.py), by tier and gap.

    python scripts/evidence/pairs_dev_eval.py --pair-table <dir> --pair-info train_pairs.parquet --out dev.json name=ckpt ...
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_one import FEATURE_SETS  # noqa: E402

from xviral.model import pairs, train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pair-table", required=True)
ap.add_argument("--pair-info", required=True)
ap.add_argument("--dev-frac", type=float, default=0.15)
ap.add_argument("--out", required=True)
ap.add_argument("ckpt", nargs="+")
args = ap.parse_args()
ptable, frame = pairs.load(args.pair_table, FEATURE_SETS["ts46"])
frame = frame.join(pl.read_parquet(args.pair_info).select("post_id", "post_id_b", "tier", "cos"), on=["post_id", "post_id_b"], how="left", maintain_order="left")
is_dev = (frame["author_id"].hash(seed=7) % 1000 < int(args.dev_frac * 1000)).to_numpy()
res = {}
for part, fr in (("dev", frame.filter(pl.Series(is_dev))), ("train", frame.filter(pl.Series(~is_dev)))):
    ps = pairs.pair_set(ptable, fr)
    d = ps.d.numpy()
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        model, tcfg, _ = train.load(path, "cpu")
        s = pairs.scores(model, ps, tcfg)
        r = {}
        for tier in ("all", "tac", "strict", "loose"):
            mt = np.ones(len(d), bool) if tier == "all" else (fr["tier"] == tier).to_numpy()
            for g in (1.25, 2.0, 3.0):
                m = mt & (np.abs(d) >= math.log(g))
                r[f"{tier}_{g}x"] = {"pairs": int(m.sum()), "accounts": int(fr.filter(pl.Series(m))["author_id"].n_unique()), "accuracy": float(((s * d) > 0)[m].mean())}
        res.setdefault(name, {})[part] = r
        print(part, name, {k: f"{100 * v['accuracy']:.1f}% n={v['pairs']}" for k, v in r.items() if k.endswith("1.25x") or k == "all_2.0x"})
json.dump(res, open(args.out, "w"), indent=1)
