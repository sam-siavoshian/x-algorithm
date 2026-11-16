#!/usr/bin/env python3
"""Warm-start fine-tune of a trained small model on the image-subset table, with or without the image inputs.

    python scripts/images/finetune.py --table data/images/table_img --init data/runs/ts46_s1/best.pt --out data/images/runs/img_s1 --steps 600 --seed 1

The image arm and the control arm start from the same checkpoint, see the same rows in the same order (same seed) and
differ only in the auxiliary image inputs (new tensors, initialised fresh; the control table has none). Writes best.pt
with fitted temperatures, metrics.json (test split, audience rows) and table.txt (the table it was trained on).
"""
import argparse, json, logging, os, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from train_one import FEATURE_SETS
from xviral.model import evaluate, train
from xviral.model.config import TrainCfg
from xviral.model.features import FeatureTable
ap = argparse.ArgumentParser()
ap.add_argument("--table", required=True); ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=600); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--evals", type=int, default=2)
a = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "table.txt").write_text(a.table)
table = FeatureTable.load(a.table, "cpu").without_num(FEATURE_SETS["ts46"])
mcfg = train.model_cfg_for(table, "small")
tcfg = TrainCfg(steps=a.steps, batch_size=64, lr=a.lr, warmup_steps=50, eval_every=max(a.steps // a.evals, 1), seed=a.seed, eval_max_rows=5000)
res = train.train(table, out, mcfg, tcfg, log_every=50, init_from=a.init)
model, tcfg, ck = train.load(out / "best.pt", "cpu")
temps = evaluate.fit_temperature(model, table, table.rows_of_split("val", audience_only=True), tcfg)
train.save(out / "best.pt", model, model.cfg, tcfg, ck["table_meta"], ck["metrics"], temps)
pred = evaluate.predict_rows(model, table, table.rows_of_split("test", audience_only=True), tcfg, temperature=temps)
m = evaluate.metrics(pred, table, tcfg) | {"best_step": res["best"].get("step"), "args": vars(a), "aux_dims": table.meta.get("aux_dims")}
(out / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
print(json.dumps({k: round(v, 4) for k, v in m.items() if k in ("pairwise_likes", "within_rho_likes", "pairwise_views", "nll_likes")}))
