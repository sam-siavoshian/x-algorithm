#!/usr/bin/env python3
"""Train, validation and test numbers side by side for one trained model: the same metrics train_one.py reports on
the test split (ordinary same-account pairs, audience accounts), also on the validation split (held-out accounts,
before the cutoff) and on a sample of the training split (accounts the model was fitted on, before the cutoff).

    python research/tracks/embed_api/eval_splits.py --table <table> --run <run dir> [--train-accounts 700]
Writes <run>/splits.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from xviral.model import evaluate, train
from xviral.model.features import FeatureTable

ap = argparse.ArgumentParser()
ap.add_argument("--table", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--train-accounts", type=int, default=700)
a = ap.parse_args()
run = Path(a.run)
table = FeatureTable.load(a.table, "cpu").without_num(lambda n: False)
model, tcfg, ck = train.load(run / "best.pt", "cpu")
temps = ck.get("temperature") or ck.get("temps")
KEYS = ("n", "pairwise_likes", "pairs_likes", "within_rho_likes", "pairwise_views", "pairs_views", "within_rho_views", "nll_likes", "nll_views")
out = {}
for split in ("train", "val"):
    rows = table.rows_of_split(split, audience_only=True)
    if split == "train":
        au = table.t["author_idx"][rows].numpy()
        pick = np.random.default_rng(0).choice(np.unique(au), a.train_accounts, replace=False)
        rows = rows[torch.from_numpy(np.isin(au, pick))]
    m = evaluate.metrics(evaluate.predict_rows(model, table, rows, tcfg, temperature=temps), table, tcfg)
    out[split] = {k: m[k] for k in KEYS if k in m}
t = json.loads((run / "metrics.json").read_text())
out["test"] = {k: t[k] for k in KEYS if k in t}
out["best_step"] = t.get("best_step")
(run / "splits.json").write_text(json.dumps(out, indent=1, default=float))
print(run.name, " ".join(f"{s}: likes {100 * out[s]['pairwise_likes']:.1f}% rho {out[s]['within_rho_likes']:.3f} nll {out[s]['nll_likes']:.3f} (n={out[s]['n']})" for s in ("train", "val", "test")))
