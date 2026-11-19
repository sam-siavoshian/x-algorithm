#!/usr/bin/env python3
"""Per-row predictions of checkpoints on a broad table's evaluation rows (scripts/general/predict_rows.py, with train and
validation rows added so that train, validation and test can be reported side by side).

    python scripts/combine/predict_rows.py --table data/combine/table_te3l_broad --run name=best.pt [--run ...] --out data/combine/pred

Row sets: train_aud (700 tech-audience accounts of the training split, the embedding track's sample), val_aud (tech-audience
validation accounts, before the cutoff), val_broad (the general track's validation accounts), test_aud (tech-audience rows
after the cutoff), bed (the frozen generalization test bed). bed and val_broad are also scored "as a customer": src_name
unknown in every row, the sampling regime kept. Each row is scored alone with its own history.
"""
from __future__ import annotations
import argparse, logging
from pathlib import Path
import numpy as np, polars as pl, torch
from xviral.model import episodes, evaluate, train as tr
from xviral.model.features import FeatureTable

ap = argparse.ArgumentParser()
ap.add_argument("--table", required=True); ap.add_argument("--run", action="append", required=True)
ap.add_argument("--out", default="data/combine/pred"); ap.add_argument("--sets", default="train_aud,val_aud,val_broad,test_aud,bed"); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s"); log = logging.getLogger("pred")
full = FeatureTable.load(a.table, "cpu")
info = pl.read_parquet(Path(a.table) / "row_info.parquet")
role, split, aud, author = info["g_role"].to_numpy(), full.t["split"].numpy(), full.t["audience"].numpy(), full.t["author_idx"].numpy()
pos = np.arange(full.n) - full.t["a_start"].numpy()
tr_mask = (split == 0) & aud & (role == "orig")
pick = np.random.default_rng(0).choice(np.unique(author[tr_mask]), 700, replace=False)
sets = {"train_aud": tr_mask & np.isin(author, pick), "val_aud": (split == 1) & aud & (role == "orig"),
        "val_broad": (role == "val") & (split == 1) & (pos >= 24), "test_aud": (split == 2) & aud & (role == "orig"), "bed": info["bed_eval"].to_numpy()}
sets = {k: v for k, v in sets.items() if k in a.sets.split(",")}
Path(a.out).mkdir(parents=True, exist_ok=True)
KEEP = ("row", "likes_mean", "likes_q50", "views_mean", "views_q50", "y_likes", "y_views", "nll_likes", "nll_views")
for spec in a.run:
    name, path = spec.split("=", 1)
    model, tcfg, ck = tr.load(path, "cpu")
    m = ck["table_meta"]
    assert int(m["emb_dim"]) == int(full.meta["emb_dim"]), f"{name}: the checkpoint's embedding ({m['emb_dim']}) is not this table's ({full.meta['emb_dim']})"
    table = full.without_num(lambda n: n not in set(m["num_base"])).without_cat([c for c in full.meta["cat_names"] if c not in m["cat_names"]])
    assert table.meta["num_names"] == m["num_names"] and table.meta["cat_names"] == m["cat_names"], name
    modes = [("", table)]
    if "src_name" in m["cat_names"]:
        t2 = FeatureTable(meta=table.meta, t={**table.t, "cat": table.t["cat"].clone()}, texts=table.texts)
        t2.t["cat"][:, m["cat_names"].index("src_name")] = 0
        modes.append(("_customer", t2))
    for suffix, tab in modes:
        for sname, mask in sets.items():
            if suffix and sname not in ("bed", "val_broad"):
                continue
            f = Path(a.out) / f"{name}{suffix}__{sname}.npz"
            if f.exists():
                continue
            rows = torch.from_numpy(np.flatnonzero(mask))
            pred = evaluate.predict_rows(model, tab, rows, tcfg, temperature=ck.get("temperature") or None)
            keep = {k: pred[k] for k in KEEP if k in pred}
            if sname in ("bed", "val_broad"):
                ctx = np.concatenate([episodes.build(tab, rows[i:i + 2048], model.cfg, tcfg, train=False).ctx.numpy() for i in range(0, len(rows), 2048)])
                keep.update(own_med_likes=ctx[:, 3])
            np.savez(f, **keep)
            log.info("%s%s %s: %d rows", name, suffix, sname, len(rows))
print("PRED_DONE")
