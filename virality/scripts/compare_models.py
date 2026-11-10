#!/usr/bin/env python3
"""Train the surrogate and its ablations on a prepared feature table and compare them, on the
time-forward test split, with the author-median baseline and gradient-boosted trees.

    python scripts/compare_models.py --table <work>/model/table --out <work>/model/compare \
        --preset small --steps 6000 --device cpu

Every model is scored the same way (xviral.model.evaluate.metrics): each test post alone with its
own history, same-author pairs within 7 days, the author's own top 10% from visible history.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from xviral.model import evaluate, train
from xviral.model.config import TrainCfg
from xviral.model.features import FeatureTable

KEYS = ["pairwise_like", "within_rho_like", "own_p90_auc_rel_like", "own_p90_ece_rel_like", "nll_rel_like",
        "pairwise_reach", "within_rho_reach", "own_p90_auc_rel_view", "auc_reach_10x", "ece_reach_10x", "nll_reach"]


def gbdt_scores(table: FeatureTable, target: str, with_emb: bool, seed: int = 0) -> np.ndarray:
    import lightgbm as lgb

    t = table.t
    X = np.concatenate([t["num"].cpu().numpy(), t["cat"].cpu().numpy().astype(np.float32)], axis=1)
    split = t["split"].cpu().numpy()
    y = t[target].cpu().numpy()
    if with_emb and "emb" in t:
        from sklearn.decomposition import PCA

        E = t["emb"].float().cpu().numpy()
        X = np.concatenate([X, PCA(n_components=min(48, E.shape[1]), random_state=seed).fit(E[split == 0]).transform(E)], axis=1)
    ok = ~np.isnan(y)
    tr, va = (split == 0) & ok, (split == 1) & ok
    params = dict(objective="huber", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=seed)
    dtr = lgb.Dataset(X[tr], y[tr])
    model = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[lgb.Dataset(X[va], y[va], reference=dtr)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    return model.predict(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--variants", nargs="*", default=["full", "no_history_outcomes", "no_content", "no_neighbors"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = FeatureTable.load(args.table, args.device)
    test = table.rows_of_split("test")
    val = table.rows_of_split("val")
    results: dict[str, dict] = {}
    overrides = {"full": {}, "no_history_outcomes": {"use_history_outcomes": False}, "no_content": {"use_content": False},
                 "no_neighbors": {"use_neighbors": False}}
    ref_pred = None
    for name in args.variants:
        if name == "no_neighbors" and "nbr_idx" not in table.t:
            continue
        mcfg = train.model_cfg_for(table, args.preset, **overrides[name])
        tcfg = TrainCfg(steps=args.steps, batch_size=args.batch_size, lr=args.lr, eval_every=max(args.steps // 6, 1), seed=args.seed)
        res = train.train(table, out / name, mcfg, tcfg)
        model, tcfg, ck = train.load(out / name / "best.pt", args.device)
        temps = evaluate.fit_temperature(model, table, val, tcfg)
        train.save(out / name / "best.pt", model, model.cfg, tcfg, ck["table_meta"], ck["metrics"], temps)
        pred = evaluate.predict_rows(model, table, test, tcfg, temperature=temps)
        results[name] = evaluate.metrics(pred, table, tcfg) | {"params": res["params"], "best_step": res["best"].get("step"), "temperature": temps}
        if name == "full" or ref_pred is None:
            ref_pred = pred
        logging.info("%s: %s", name, {k: round(v, 4) for k, v in results[name].items() if k in KEYS})
        (out / "results.json").write_text(json.dumps(results, indent=2, default=float))

    mcfg, tcfg = train.model_cfg_for(table, args.preset), TrainCfg()
    if ref_pred is None:
        ref_pred = evaluate.predict_rows(train.Surrogate(mcfg, table.meta["action_index"]).to(args.device), table, test, tcfg, with_nll=False)
    rows = test.cpu().numpy()
    base = evaluate.baseline_author_median(table, test, mcfg, tcfg)
    rank_only = lambda m: {k: v for k, v in m.items() if k.startswith(("pairwise", "within_rho", "global_rho", "pairs", "n"))}  # noqa: E731
    results["author_median"] = rank_only(evaluate.metrics({**ref_pred, **base}, table, tcfg))
    for with_emb in (False, True):
        scores = {}
        for target, key in (("y_like", "like_mean"), ("y_reach", "reach_mean")):
            if not torch.isnan(table.t[target]).all():
                scores[key] = gbdt_scores(table, target, with_emb, args.seed)[rows]
        # Trees give one score per post; it also ranks posts for the own-top-10% question.
        pred = {**ref_pred, **scores}
        if "like_mean" in scores:
            pred["p_own_p90_rel_like"] = scores["like_mean"]
        if "reach_mean" in scores:
            pred["p_own_p90_rel_view"] = scores["reach_mean"]
            for mult in ("1x", "10x", "100x"):
                pred[f"p_reach_{mult}"] = scores["reach_mean"]
        m = evaluate.metrics(pred, table, tcfg)
        results["gbdt_tabular" + ("_plus_embedding" if with_emb else "")] = {k: v for k, v in m.items() if not k.startswith(("nll", "own_p90_ece", "ece"))}
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))

    cols = [k for k in KEYS if any(k in r and not np.isnan(r[k]) for r in results.values())]
    lines = ["| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[k]:.3f}" if k in r and not np.isnan(r[k]) else "" for k in cols) + " |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
