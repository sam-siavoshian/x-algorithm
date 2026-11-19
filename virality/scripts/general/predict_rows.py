#!/usr/bin/env python3
"""Per-row predictions of any checkpoint on the general table's evaluation rows, saved for scripts/general/report.py.

    python scripts/general/predict_rows.py --run name=path/best.pt [--run ...] [--customer] --out data/general/pred

Rows scored: the test bed (row_info bed_eval), the general validation accounts (to fit the level correction), and the existing
audience test rows. Every model reads the same table narrowed to its own inputs. --customer scores the bed and validation rows
again with src_name "unknown" in every row (the account's history included), which is what a new customer's export is; the
sampling regime stays what it is. Each row is scored alone with its own history (model.evaluate.predict_rows).
"""
from __future__ import annotations
import argparse, logging
from pathlib import Path
import numpy as np, polars as pl, torch
from xviral.model import episodes, evaluate, train as tr
from xviral.model.features import FeatureTable

ap = argparse.ArgumentParser()
ap.add_argument("--table", default="data/general/table"); ap.add_argument("--run", action="append", required=True)
ap.add_argument("--out", default="data/general/pred"); ap.add_argument("--customer", action="store_true"); ap.add_argument("--skip-test-aud", action="store_true")
args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s"); log = logging.getLogger("pred")
full = FeatureTable.load(args.table, "cpu")
info = pl.read_parquet(Path(args.table) / "row_info.parquet")
role, split, aud = info["g_role"].to_numpy(), full.t["split"].numpy(), full.t["audience"].numpy()
val_rows = (role == "val") & (split == 1)
a_start = full.t["a_start"].numpy()
val_rows &= (np.arange(full.n) - a_start) >= 24            # validation accounts: rows with some history
sets = {"bed": info["bed_eval"].to_numpy(), "val": val_rows}
if not args.skip_test_aud:
    sets["test_aud"] = (split == 2) & aud & (role == "orig")
Path(args.out).mkdir(parents=True, exist_ok=True)
KEEP = ("row", "likes_mean", "likes_q50", "likes_q10", "likes_q90", "views_mean", "views_q50", "y_likes", "y_views", "nll_likes", "nll_views")
for spec in args.run:
    name, path = spec.split("=", 1)
    model, tcfg, ck = tr.load(path, "cpu")
    m = ck["table_meta"]
    table = full.without_num(lambda n: n not in set(m["num_base"])).without_cat([c for c in full.meta["cat_names"] if c not in m["cat_names"]])
    assert table.meta["num_names"] == m["num_names"] and table.meta["cat_names"] == m["cat_names"], name
    modes = [("", table)]
    if args.customer and "src_name" in m["cat_names"]:
        t2 = FeatureTable(meta=table.meta, t={**table.t, "cat": table.t["cat"].clone()}, texts=table.texts)
        t2.t["cat"][:, m["cat_names"].index("src_name")] = 0
        modes.append(("_customer", t2))
    for suffix, tab in modes:
        for sname, mask in sets.items():
            if suffix and sname == "test_aud":
                continue
            rows = torch.from_numpy(np.flatnonzero(mask))
            pred = evaluate.predict_rows(model, tab, rows, tcfg, temperature=ck.get("temperature") or None)
            ctx = np.concatenate([episodes.build(tab, rows[i:i + 2048], model.cfg, tcfg, train=False).ctx.numpy() for i in range(0, len(rows), 2048)])
            keep = {k: pred[k] for k in KEEP if k in pred}
            keep.update(own_med_likes=ctx[:, 3], own_med_views=ctx[:, 5], own_has_likes=ctx[:, 7], own_has_views=ctx[:, 8])
            np.savez(Path(args.out) / f"{name}{suffix}__{sname}.npz", **keep)
            log.info("%s%s %s: %d rows", name, suffix, sname, len(rows))
print("PRED_DONE")
