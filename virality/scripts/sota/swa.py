#!/usr/bin/env python3
"""Stochastic weight averaging and snapshots from one run's saved steps (train_sota.py saves step<N>.pt at every validation).

    python scripts/sota/swa.py --run data/sota/runs/snap_s1 --steps 1000,2000,3000

Writes <run>/swa.pt (uniform average of the snapshots' weights) and <run>/snap<N>.pt (each snapshot), all with temperatures
fitted on the validation accounts as train_one.py does for best.pt, and <run>/swa.json with validation and test metrics of
each, of the weight average and of the snapshots' prediction average (their score-level ensemble on ordinary pairs).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "scripts")
from train_one import FEATURE_SETS
from xviral.model import evaluate, train
from xviral.model.features import FeatureTable

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--steps", required=True); ap.add_argument("--table", default="data/work/model/table")
a = ap.parse_args()
run = Path(a.run)
table = FeatureTable.load(a.table, "cpu").without_num(FEATURE_SETS["ts46"])
val, test = table.rows_of_split("val", audience_only=True), table.rows_of_split("test", audience_only=True)
steps = [int(s) for s in a.steps.split(",")]
models = [train.load(run / f"step{s}.pt", "cpu") for s in steps]
avg = {k: torch.stack([m.state_dict()[k].float() for m, _, _ in models]).mean(0).to(models[0][0].state_dict()[k].dtype) for k in models[0][0].state_dict()}
swa, tcfg, ck = train.load(run / f"step{steps[-1]}.pt", "cpu")
swa.load_state_dict(avg)
out, preds = {}, {"val": [], "test": []}
K = ("pairwise_likes", "within_rho_likes", "nll_likes", "pairwise_views", "nll_rel_like")
for name, m in [(f"snap{s}", mm[0]) for s, mm in zip(steps, models)] + [("swa", swa)]:
    temps = evaluate.fit_temperature(m, table, val, tcfg)
    train.save(run / f"{name}.pt", m, m.cfg, tcfg, ck["table_meta"], {}, temps)
    out[name] = {}
    for split, rows in (("val", val), ("test", test)):
        p = evaluate.predict_rows(m, table, rows, tcfg, temperature=temps)
        r = evaluate.metrics(p, table, tcfg)
        out[name][split] = {k: r[k] for k in K if k in r}
        if name != "swa":
            preds[split].append(p)
    print(name, out[name], flush=True)
out["snapshot_ensemble"] = {}
for split in ("val", "test"):
    p = dict(preds[split][0])
    for k in ("likes_mean", "views_mean"):
        p[k] = np.mean([q[k] for q in preds[split]], 0)
    r = evaluate.metrics(p, table, tcfg)
    out["snapshot_ensemble"][split] = {k: r[k] for k in ("pairwise_likes", "within_rho_likes", "pairwise_views") if k in r}
print("snapshot_ensemble", out["snapshot_ensemble"])
(run / "swa.json").write_text(json.dumps(out, indent=1, default=float))
