#!/usr/bin/env python3
"""Track "sota": train one candidate or its control and record train / validation / test curves side by side.

    python scripts/sota/train_sota.py --out data/sota/runs/<name> --seed 1 [--steps 3000] [--set key=value ...] [--mset key=value ...]

--set changes a TrainCfg field (pair_loss=soft, hist_outcome_drop=0.15, mask_hist_p=0.15, lr_cycles=3, ...), --mset a
ModelCfg field (use_hist_ridge=true, use_neighbors=false). With neither it is the control: scripts/train_one.py's
46-column `small` model. Everything else is identical for every run: data, batch order generator, steps, evaluation rows.

At every validation the averaged weights are scored on
  train   a fixed sample of audience training rows (posts the model was fitted on),
  val     the held-out validation accounts (model selection: lowest mean NLL, as in train.py; nothing else selects),
  test    a fixed sample of audience rows after the cutoff (reported only),
  dev     the replicate track's 2,805 pre-cutoff same-topic pairs scored as alternative drafts (reported only).
The chosen checkpoint is then scored on all validation and test rows and on the development pairs (scores saved).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_one import FEATURE_SETS  # noqa: E402

from xviral.model import evaluate, pairs, train  # noqa: E402
from xviral.model.config import ModelCfg, TrainCfg  # noqa: E402
from xviral.model.features import FeatureTable  # noqa: E402

KEYS = ("pairwise_likes", "within_rho_likes", "nll_likes", "pairwise_views", "within_rho_views", "nll_rel_like", "own_p90_auc_rel_like")


def parse(kv: list[str], model) -> dict:
    out = {}
    for item in kv:
        k, v = item.split("=", 1)
        ann = model.model_fields[k].annotation
        out[k] = v.lower() in ("1", "true", "yes") if ann is bool else ann(v) if ann in (int, float, str) else json.loads(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/work/model/table")
    ap.add_argument("--dev-table", default="data/tracks/replicate/dev_table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--features", default="ts46")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--evals", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--curve-rows", type=int, default=8000)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--mset", action="append", default=[])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out)
    table = FeatureTable.load(args.table, "cpu").without_num(FEATURE_SETS[args.features])
    ptable, frame = pairs.load(args.dev_table, FEATURE_SETS[args.features])
    dev = pairs.pair_set(ptable, frame)
    mcfg = train.model_cfg_for(table, args.preset, **parse(args.mset, ModelCfg))
    tcfg = TrainCfg(steps=args.steps, batch_size=64, lr=args.lr, eval_every=max(args.steps // args.evals, 1), seed=args.seed,
                    save_snapshots=True, **parse(args.set, TrainCfg))
    g = torch.Generator().manual_seed(12345)        # the same curve rows for every run and seed
    pick = lambda rows: rows[torch.randperm(len(rows), generator=g)[:args.curve_rows]].sort().values  # noqa: E731
    curve = {"train": pick(table.rows_of_split("train", audience_only=True)), "test": pick(table.rows_of_split("test", audience_only=True))}
    plain = TrainCfg(**{**tcfg.model_dump(), "hist_outcome_drop": 0.0, "hist_drop_all": 0.0, "mask_hist_p": 0.0})

    def hook(model, step: int) -> dict:
        m = {}
        for name, rows in curve.items():
            r = evaluate.metrics(evaluate.predict_rows(model, table, rows, plain), table, plain)
            m.update({f"{name}_{k}": r[k] for k in KEYS if k in r})
        m.update({f"dev_{k[4:]}": v for k, v in pairs.accuracy(model, dev, plain).items() if k.startswith("ctl_acc_") and k[-1] == "x"})
        logging.info("curves @%d: %s", step, {k: round(v, 4) for k, v in m.items() if "pairwise_likes" in k or k.startswith("dev_")})
        return m

    t0 = time.time()
    res = train.train(table, out, mcfg, tcfg, eval_hook=hook)
    train_s = time.time() - t0
    model, tcfg2, ck = train.load(out / "best.pt", "cpu")
    temps = evaluate.fit_temperature(model, table, table.rows_of_split("val", audience_only=True), plain)
    train.save(out / "best.pt", model, model.cfg, tcfg2, ck["table_meta"], ck["metrics"], temps)
    final = {}
    for name in ("val", "test"):
        rows = table.rows_of_split(name, audience_only=True)
        final[name] = evaluate.metrics(evaluate.predict_rows(model, table, rows, plain, temperature=temps), table, plain)
    final["train_sample"] = evaluate.metrics(evaluate.predict_rows(model, table, curve["train"], plain, temperature=temps), table, plain)
    s = pairs.scores(model, dev, plain)
    np.save(out / "dev_scores.npy", s)
    final["dev"] = pairs.accuracy(model, dev, plain)
    (out / "metrics.json").write_text(json.dumps({"final": final, "best_step": res["best"].get("step"), "history": res["history"], "params": res["params"],
                                                  "train_seconds": train_s, "args": vars(args), "temperature": temps}, indent=1, default=float))
    print(json.dumps({"best_step": res["best"].get("step"), **{f"{n}_{k}": round(final[n][k], 4) for n in ("train_sample", "val", "test") for k in ("pairwise_likes", "nll_likes") if k in final[n]},
                      "dev_2x": round(final["dev"]["ctl_acc_2.0x"], 4)}))


if __name__ == "__main__":
    main()
