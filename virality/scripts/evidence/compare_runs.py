#!/usr/bin/env python3
"""Compare trained variants on the pair benchmark: accuracy, and whether confidence means anything.

    python scripts/evidence/compare_runs.py --run name=path/to/tac.parquet [--run ...] [--group label=name1,name2] --out table.json

For each run (the .parquet written next to tac_serve_eval.py's output) and each group (members'
scores averaged, a score-level ensemble): accuracy over all scored pairs, and accuracy by confidence
band with the confidence scale cross-fitted over accounts, so no pair is judged with a scale fitted
on its own account. A model is fit for the advice layer when accuracy rises with stated confidence.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from xviral.model import evaluate

BANDS = ((0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 1.01))


def load(path: str) -> pl.DataFrame:
    return pl.read_parquet(path).select("post_id", "post_id_b", "author_id", "likes", "likes_b", "s_likes").unique(["post_id", "post_id_b"])


def analyse(R: pl.DataFrame, seed: int = 0) -> dict:
    d = ((R["likes"].cast(pl.Float64) + 1).log() - (R["likes_b"].cast(pl.Float64) + 1).log()).to_numpy()
    keep = np.abs(d) >= np.log(1.25)
    s, d, auth = R["s_likes"].to_numpy()[keep], d[keep], R["author_id"].to_numpy()[keep]
    first, correct = d > 0, (s * d) > 0
    rng = np.random.default_rng(seed)
    ua = np.unique(auth)
    fold = dict(zip(ua, rng.integers(0, 5, len(ua))))
    fo = np.array([fold[a] for a in auth])
    conf, ks = np.zeros(len(s)), []
    for k_ in range(5):
        k = evaluate.fit_pair_scale(s[fo != k_], first[fo != k_])
        ks.append(k)
        p = 1 / (1 + np.exp(-k * s[fo == k_]))
        conf[fo == k_] = np.maximum(p, 1 - p)
    out = {"pairs": int(len(s)), "accuracy": float(correct.mean()), "scale": float(np.mean(ks)), "bands": {}}
    for lo, hi in BANDS:
        m = (conf >= lo) & (conf < hi)
        out["bands"][f"{lo:.2f}-{min(hi, 1):.2f}"] = {"pairs": int(m.sum()), "accuracy": float(correct[m].mean()) if m.sum() else None}
    m = conf >= 0.6
    out["confident"] = {"share": float(m.mean()), "pairs": int(m.sum()), "accuracy": float(correct[m].mean()) if m.sum() else None}
    # Does accuracy rise with confidence? Rank correlation between confidence and being right, over pairs.
    out["confidence_auc"] = evaluate.auroc(correct.astype(float), conf)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=path.parquet")
    ap.add_argument("--group", action="append", default=[], help="label=name1,name2,...")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = {r.split("=", 1)[0]: load(r.split("=", 1)[1]) for r in args.run}
    res = {n: analyse(R) for n, R in runs.items()}
    for g in args.group:
        label, members = g.split("=", 1)
        frames = [runs[m].rename({"s_likes": f"s_{i}"}) for i, m in enumerate(members.split(","))]
        J = frames[0]
        for f in frames[1:]:
            J = J.join(f.select("post_id", "post_id_b", pl.col("^s_\\d+$")), on=["post_id", "post_id_b"])
        J = J.with_columns(pl.mean_horizontal(pl.col("^s_\\d+$")).alias("s_likes"))
        res[label] = analyse(J) | {"members": members.split(",")}
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"{'model':<26} {'pairs':>5} {'acc':>6} | " + " ".join(f"{b:>13}" for b in res[next(iter(res))]["bands"]) + f" | {'>=0.60 share':>12} {'acc':>6} {'conf AUC':>8}")
    for n, r in res.items():
        bands = " ".join((f"{100 * v['accuracy']:5.1f}% n={v['pairs']:<4}" if v["accuracy"] is not None else f"{'-':>13}") for v in r["bands"].values())
        c = r["confident"]
        print(f"{n:<26} {r['pairs']:>5} {100 * r['accuracy']:5.1f}% | {bands} | {100 * c['share']:>11.0f}% " + (f"{100 * c['accuracy']:5.1f}%" if c["accuracy"] is not None else f"{'-':>6}") + f" {r['confidence_auc']:>8.3f}")


if __name__ == "__main__":
    main()
