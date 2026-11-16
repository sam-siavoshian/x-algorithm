#!/usr/bin/env python3
"""Train or fine-tune a model with the loss on controlled pairs added.

    python scripts/train_pairs.py --table <work>/model/table --pair-table <dir> --out <dir> --features ts46 \
        [--init <run>/best.pt] --steps 800 --lr 2e-4 --w-ctl 1.0 [--ctl-loss ranknet|margin] [--ctl-weight gap|flat]

The pair table comes from scripts/build_pair_table.py. Its pairs are split by account: --dev-frac of the accounts
are held out and scored at every validation (never the benchmark pairs). Every validation writes step<N>.pt; the
final choice among them is made on the held-out controlled pairs and the ordinary validation metrics, and the
chosen checkpoint gets fitted temperatures like train_one.py's best.pt.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from train_one import FEATURE_SETS  # noqa: E402

from xviral.model import evaluate, pairs, train  # noqa: E402
from xviral.model.config import TrainCfg  # noqa: E402
from xviral.model.features import FeatureTable  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--pair-table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--features", default="ts46", choices=list(FEATURE_SETS))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--evals", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-ctl", type=float, default=1.0)
    ap.add_argument("--ctl-batch", type=int, default=64)
    ap.add_argument("--ctl-loss", default="ranknet")
    ap.add_argument("--ctl-weight", default="gap")
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--max-train-pairs", type=int, default=0)
    ap.add_argument("--pair-info", default=None, help="parquet with (post_id, post_id_b, tier, tac, cos) from select_pairs.py")
    ap.add_argument("--drop-cat", default="", help="comma-separated categorical inputs to leave out of both tables (e.g. src_regime)")
    ap.add_argument("--eval-rows", default=None, help="npy of post ids: the test metrics are computed on these rows only")
    ap.add_argument("--only", default=None, help="train only on pairs of these tiers, comma separated: tac (same link or quoted post), strict, loose")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out)
    table = FeatureTable.load(args.table, "cpu").without_num(FEATURE_SETS[args.features])
    ptable, frame = pairs.load(args.pair_table, FEATURE_SETS[args.features])
    if args.drop_cat:
        drop = [c for c in args.drop_cat.split(",") if c]
        table, ptable = table.without_cat(drop), ptable.without_cat(drop)
    assert table.meta["cat_names"] == ptable.meta["cat_names"], "the pair table and the training table differ in categorical inputs"
    if args.pair_info:
        info = pl.read_parquet(args.pair_info).select("post_id", "post_id_b", "tier", "tac", "cos")
        frame = frame.join(info, on=["post_id", "post_id_b"], how="left", maintain_order="left")
        assert frame["tier"].null_count() == 0
    is_dev = (frame["author_id"].hash(seed=7) % 1000 < int(args.dev_frac * 1000)).to_numpy()
    tr_frame, dev_frame = frame.filter(pl.Series(~is_dev)), frame.filter(pl.Series(is_dev))
    if args.only:
        tr_frame = tr_frame.filter(pl.col("tier").is_in(args.only.split(",")))
    if args.max_train_pairs:
        tr_frame = tr_frame.sample(min(args.max_train_pairs, tr_frame.height), seed=0)
    ctl, ctl_dev = pairs.pair_set(ptable, tr_frame), pairs.pair_set(ptable, dev_frame)
    logging.info("controlled pairs: %d train (%d accounts), %d dev (%d accounts)", len(ctl), tr_frame["author_id"].n_unique(), len(ctl_dev), dev_frame["author_id"].n_unique())
    mcfg = train.model_cfg_for(table, args.preset)
    tcfg = TrainCfg(steps=args.steps, batch_size=args.batch_size, lr=args.lr, eval_every=max(args.steps // args.evals, 1), seed=args.seed, warmup_steps=args.warmup,
                    w_ctl=args.w_ctl, ctl_batch=args.ctl_batch, ctl_loss=args.ctl_loss, ctl_weight=args.ctl_weight)
    if args.init:
        m0, t0, _ = train.load(args.init, "cpu")
        base = train.validate(m0, table, table.rows_of_split("val", audience_only=True)[:tcfg.eval_max_rows], tcfg) | pairs.accuracy(m0, ctl_dev, tcfg)
        logging.info("start: val nll %.4f pairwise_likes %.4f dev %s", base["val_nll"], base.get("pairwise_likes", float("nan")), {k: round(v, 4) for k, v in base.items() if k.startswith("ctl_")})
        out.mkdir(parents=True, exist_ok=True)
        (out / "start.json").write_text(json.dumps(base, indent=1, default=float))
    res = train.train(table, out, mcfg, tcfg, init_from=args.init, ctl=ctl, ctl_dev=ctl_dev)
    hist = res["history"]
    # Choice among the saved steps: the best accuracy on held-out controlled pairs (2x gaps), provided the ordinary
    # validation loss is within 0.02 nats of the best seen in this run.
    floor = min(h["val_nll"] for h in hist) + 0.02
    ok = [h for h in hist if h["val_nll"] <= floor]
    pick = max(ok, key=lambda h: h["ctl_acc_2.0x"])
    shutil.copy(out / f"step{pick['step']}.pt", out / "best.pt")
    model, tcfg, ck = train.load(out / "best.pt", "cpu")
    temps = evaluate.fit_temperature(model, table, table.rows_of_split("val", audience_only=True), tcfg)
    train.save(out / "best.pt", model, model.cfg, tcfg, ck["table_meta"], ck["metrics"], temps)
    test_rows = table.rows_of_split("test", audience_only=True)
    if args.eval_rows:
        import torch

        test_rows = test_rows[torch.from_numpy(np.isin(table.t["post_id"][test_rows].numpy(), np.load(args.eval_rows)))]
    pred = evaluate.predict_rows(model, table, test_rows, tcfg, temperature=temps)
    m = evaluate.metrics(pred, table, tcfg) | {"picked_step": pick["step"], "history": hist, "temperature": temps, "args": vars(args),
                                               "ctl_train_pairs": len(ctl), "ctl_dev_pairs": len(ctl_dev)}
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k in ("pairwise_likes", "within_rho_likes", "nll_likes", "picked_step")}))


if __name__ == "__main__":
    main()
