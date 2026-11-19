#!/usr/bin/env python3
"""Per-row predictions on the audience test rows (after the cutoff) for checkpoints, for paired account-resampled comparisons
of ordinary same-account pairs.   python scripts/sota/test_rows.py --out data/sota/test name=ckpt.pt ...  -> <out>/<name>.npy"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, "scripts")
from train_one import FEATURE_SETS
from xviral.model import evaluate, train
from xviral.model.features import FeatureTable
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("ckpt", nargs="+"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
table = FeatureTable.load("data/work/model/table", "cpu").without_num(FEATURE_SETS["ts46"])
rows = table.rows_of_split("test", audience_only=True)
np.save(f"{a.out}/rows.npy", rows.numpy())
for spec in a.ckpt:
    name, path = spec.split("=", 1)
    if os.path.exists(f"{a.out}/{name}.npy"): continue
    m, tcfg, ck = train.load(path, "cpu")
    p = evaluate.predict_rows(m, table, rows, tcfg, temperature=ck.get("temperature") or None, with_nll=False)
    np.save(f"{a.out}/{name}.npy", p["likes_mean"]); print(name, flush=True)
