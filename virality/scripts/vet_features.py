#!/usr/bin/env python3
"""Vet candidate feature groups before they enter the model's feature list.

    python scripts/vet_features.py --table <work>/model/table --rows <work>/dataset/rows.parquet --out vet.json

For each group in xviral.features.text_structure.GROUPS, trains gradient-boosted trees on the
model's current inputs (scalars, categoricals, 48 principal components of the post embedding) with
and without the group, and compares same-author ranking on a development slice.

The development slice is the most recent 15% of the training rows by time. The test split is not
touched, so choosing features here does not leak into the reported test numbers. A group passes
when the gain in pairwise accuracy is positive in at least 95% of author-level bootstrap resamples.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import polars as pl

from xviral.features import text_structure as ts
from xviral.model import evaluate
from xviral.model.features import FeatureTable


THREADS = 2   # more threads than free cores makes LightGBM far slower, not faster


def fit_predict(X, y, tr, va, seed):
    import lightgbm as lgb

    params = dict(objective="huber", learning_rate=0.08, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=seed, num_threads=THREADS)
    dtr = lgb.Dataset(X[tr], y[tr])
    m = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[lgb.Dataset(X[va], y[va], reference=dtr)],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
    logging.info("  fit seed %d: %d rounds", seed, m.best_iteration)
    return m.predict(X), m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default=None, help="y_views or y_likes; default: views when the table has them")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--dev-share", type=float, default=0.15)
    ap.add_argument("--groups", nargs="*", default=[*ts.GROUPS, "all"])
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    global THREADS
    THREADS = args.threads
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    table = FeatureTable.load(args.table)
    t = {k: v.cpu().numpy() for k, v in table.t.items() if k in ("num", "cat", "split", "created_s", "author_idx", "post_id", "y_views", "y_likes")}
    target = args.target or ("y_views" if (~np.isnan(t["y_views"])).mean() > 0.5 else "y_likes")
    y = t[target]

    from sklearn.decomposition import PCA

    E = table.t["emb"].float().cpu().numpy()
    train_all = t["split"] == 0
    cut = np.quantile(t["created_s"][train_all], 1 - args.dev_share)
    ok = ~np.isnan(y)
    tr = train_all & (t["created_s"] < cut) & ok
    dev = train_all & (t["created_s"] >= cut) & ok
    va = (t["split"] == 1) & ok
    # The base is the model's inputs without any text-structure column already admitted, so an admitted
    # group can be re-vetted on new data (for instance on view labels) against the same reference.
    keep_num = [i for i, name in enumerate(table.meta["num_names"]) if not name.startswith("ts_")]
    base = np.concatenate([t["num"][:, keep_num], t["cat"].astype(np.float32), PCA(48, random_state=0).fit(E[tr]).transform(E)], axis=1)

    rows = pl.read_parquet(args.rows)
    cand = ts.text_structure(rows)
    C = pl.DataFrame({"post_id": t["post_id"]}).join(cand, on="post_id", how="left").select(ts.ALL).to_numpy().astype(np.float32)
    col = {c: i for i, c in enumerate(ts.ALL)}
    logging.info("target %s: train %d, dev %d, early-stop %d rows; %d candidate columns", target, tr.sum(), dev.sum(), va.sum(), C.shape[1])

    d = np.flatnonzero(dev)
    author, created, yd = t["author_idx"][d], t["created_s"][d], y[d]
    I, J = evaluate.same_author_pairs(author, created, yd, 7 * 86400, 0.1)
    pair_author = author[I]
    uniq, inv = np.unique(pair_author, return_inverse=True)

    def pair_correct(score):
        s = score[d]
        return ((s[I] - s[J]) * (yd[I] - yd[J]) > 0).astype(np.float64)

    def scores_for(X):
        return np.mean([fit_predict(X, y, tr, va, s)[0] for s in range(args.seeds)], axis=0)

    s_base = scores_for(base)
    c_base = pair_correct(s_base)
    rng = np.random.default_rng(0)
    boots = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(400)]
    per_author_n = np.bincount(inv, minlength=len(uniq)).astype(np.float64)

    def summarize(c):
        diff = np.bincount(inv, weights=c - c_base, minlength=len(uniq))
        deltas = np.array([(w * diff).sum() / (w * per_author_n).sum() for w in boots])
        return float(c.mean()), float((c - c_base).mean()), float((deltas > 0).mean()), [float(np.quantile(deltas, q)) for q in (0.05, 0.95)]

    res = {"target": target, "pairs": int(len(I)), "authors": int(len(uniq)), "base": {"pairwise": float(c_base.mean()),
           "within_rho": evaluate.spearman(s_base[d] - _author_mean(s_base[d], author), yd - _author_mean(yd, author))}, "groups": {}}
    logging.info("base: pairwise %.4f on %d pairs from %d accounts", c_base.mean(), len(I), len(uniq))
    for g in args.groups:
        cols = ts.ALL if g == "all" else [c for part in g.split("+") for c in ts.GROUPS[part]]   # "a+b" joins groups
        s = scores_for(np.concatenate([base, C[:, [col[c] for c in cols]]], axis=1))
        acc, delta, p_pos, ci = summarize(pair_correct(s))
        res["groups"][g] = {"pairwise": acc, "delta": delta, "share_of_resamples_positive": p_pos, "delta_ci90": ci,
                            "passes": bool(p_pos >= 0.95), "columns": cols}
        logging.info("%-14s pairwise %.4f  delta %+.4f  ci90 [%+.4f, %+.4f]  positive in %.0f%% of resamples  %s",
                     g, acc, delta, ci[0], ci[1], 100 * p_pos, "PASS" if p_pos >= 0.95 else "")
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


def _author_mean(v: np.ndarray, author: np.ndarray) -> np.ndarray:
    u, inv = np.unique(author, return_inverse=True)
    return (np.bincount(inv, weights=v) / np.bincount(inv))[inv]


if __name__ == "__main__":
    main()
