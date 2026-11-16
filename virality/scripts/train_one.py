#!/usr/bin/env python3
"""Train and evaluate one model variant on a prepared feature table.

    python scripts/train_one.py --table <work>/model/table --out <dir> --seed 1 --features ts20 [--preset small] [--steps 4000]

--features chooses the text-structure inputs: ts0 (none), ts20 (opening, layout, link placement),
ts46 (all six groups). Writes <dir>/best.pt with fitted temperatures and <dir>/metrics.json with
the audience test metrics. Independent of every other run, so variants can train on separate machines.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from xviral.features import text_structure as ts
from xviral.model import evaluate, train
from xviral.model.config import TrainCfg
from xviral.model.features import FeatureTable

FEATURE_SETS = {
    "ts0": lambda n: n.startswith("ts_"),
    "ts20": lambda n: n in {c for g in ("typography", "voice", "self_relative") for c in ts.GROUPS[g]},
    "ts46": lambda n: False,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", default="ts20", choices=list(FEATURE_SETS))
    ap.add_argument("--train-audience-only", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--drop-cat", default="", help="comma-separated categorical inputs to leave out (e.g. src_regime)")
    ap.add_argument("--train-exclude", default=None, help="npy of post ids left out of training and validation (they stay in the table as "
                    "their accounts' history only): trains on a subset of an expanded table without rebuilding it")
    ap.add_argument("--eval-rows", default=None, help="npy of post ids: the test metrics are computed on these rows only "
                    "(the original test rows of an expanded table), so runs on different tables report the same posts")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out)
    table = FeatureTable.load(args.table, args.device).without_num(FEATURE_SETS[args.features])
    if args.drop_cat:
        table = table.without_cat([c for c in args.drop_cat.split(",") if c])
    if args.train_exclude:
        import numpy as np
        import torch

        drop = torch.from_numpy(np.isin(table.t["post_id"].numpy(), np.load(args.train_exclude)))
        table.t["split"] = torch.where(drop, torch.full_like(table.t["split"], 3), table.t["split"])       # 3: no split, never an anchor or a candidate
        logging.info("left out of training: %d rows", int(drop.sum()))
    mcfg = train.model_cfg_for(table, args.preset)
    tcfg = TrainCfg(steps=args.steps, batch_size=args.batch_size, lr=args.lr, eval_every=max(args.steps // 6, 1), seed=args.seed,
                    train_audience_only=args.train_audience_only)
    res = train.train(table, out, mcfg, tcfg)
    model, tcfg, ck = train.load(out / "best.pt", args.device)
    temps = evaluate.fit_temperature(model, table, table.rows_of_split("val", audience_only=True), tcfg)
    train.save(out / "best.pt", model, model.cfg, tcfg, ck["table_meta"], ck["metrics"], temps)
    test_rows = table.rows_of_split("test", audience_only=True)
    if args.eval_rows:
        import numpy as np
        import torch

        test_rows = test_rows[torch.from_numpy(np.isin(table.t["post_id"][test_rows].numpy(), np.load(args.eval_rows)))]
    pred = evaluate.predict_rows(model, table, test_rows, tcfg, temperature=temps)
    m = evaluate.metrics(pred, table, tcfg) | {"params": res["params"], "best_step": res["best"].get("step"), "temperature": temps,
                                               "args": vars(args), "n_num": int(table.t["num"].shape[1])}
    (out / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
    print(json.dumps({k: round(v, 4) for k, v in m.items() if k in ("pairwise_views", "within_rho_views", "pairwise_likes", "within_rho_likes",
                                                                    "auc_reach_10x", "ece_reach_10x", "own_p90_auc_rel_view", "nll_views")}))


if __name__ == "__main__":
    main()
