#!/usr/bin/env python3
"""The general pair benchmark as the tracks README states it: accuracy by size of the real gap and by confidence band,
with pair counts, for single models and score-level ensembles, and paired differences against the first one.

    python scripts/posttype/bench_table.py --model label=dir1,dir2,... [--model ...] --out data/posttype/bench_table.json
dirN is a folder of shard_*.parquet written by tac_serve_eval.py. Confidence is cross-fitted over accounts exactly as
scripts/evidence/compare_runs.py does it (5 folds, scale fitted on the other folds).
"""
import argparse, json, sys, numpy as np, polars as pl
sys.path.insert(0, "scripts/evidence")
import compare_runs as CR
from xviral.model import evaluate
ap = argparse.ArgumentParser(); ap.add_argument("--model", action="append", required=True); ap.add_argument("--out", required=True)
args = ap.parse_args()
def frame(dirs):
    J = None
    for i, d in enumerate(dirs):
        R = CR.load(f"{d}/shard_*.parquet").rename({"s_likes": f"s_{i}"})
        J = R if J is None else J.join(R.select("post_id", "post_id_b", f"s_{i}"), on=["post_id", "post_id_b"])
    return J.with_columns(pl.mean_horizontal(pl.col(r"^s_\d+$")).alias("s_likes")).sort("post_id", "post_id_b")
def analyse(R, seed=0):
    d = ((R["likes"].cast(pl.Float64) + 1).log() - (R["likes_b"].cast(pl.Float64) + 1).log()).to_numpy()
    keep = np.abs(d) >= np.log(1.25)
    s, d, auth = R["s_likes"].to_numpy()[keep], d[keep], R["author_id"].to_numpy()[keep]
    first, correct = d > 0, (s * d) > 0
    rng = np.random.default_rng(seed); ua = np.unique(auth); fold = dict(zip(ua, rng.integers(0, 5, len(ua)))); fo = np.array([fold[a] for a in auth])
    conf, ks = np.zeros(len(s)), []
    for k_ in range(5):
        k = evaluate.fit_pair_scale(s[fo != k_], first[fo != k_]); ks.append(k)
        p = 1 / (1 + np.exp(-k * s[fo == k_])); conf[fo == k_] = np.maximum(p, 1 - p)
    return {"key": R.select("post_id", "post_id_b").filter(pl.Series(keep)), "d": d, "auth": auth, "correct": correct.astype(float), "conf": conf, "scale": float(np.mean(ks))}
res, raw = {}, {}
for spec in args.model:
    label, dirs = spec.split("=", 1); A = analyse(frame(dirs.split(","))); raw[label] = A
    out = {"members": dirs.split(","), "scale": round(A["scale"], 4), "by_gap": {}, "by_confidence": {}}
    for g in (1.25, 2, 3, 5):
        m = np.abs(A["d"]) >= np.log(g); mc = m & (A["conf"] >= 0.6)
        out["by_gap"][f"{g}x"] = {"pairs": int(m.sum()), "accuracy": round(float(A["correct"][m].mean()), 4), "accuracy_ci": [round(x, 4) for x in CR.cluster_ci(A["auth"][m], lambda i: A["correct"][m][i].mean())],
                                 "confident_share": round(float(mc.sum() / m.sum()), 4), "confident_pairs": int(mc.sum()), "confident_accuracy": round(float(A["correct"][mc].mean()), 4),
                                 "confident_ci": [round(x, 4) for x in CR.cluster_ci(A["auth"][mc], lambda i: A["correct"][mc][i].mean())]}
    for lo, hi in ((0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
        m = (A["conf"] >= lo) & (A["conf"] < hi)
        out["by_confidence"][f"{lo:.2f}-{min(hi, 1):.2f}"] = {"pairs": int(m.sum()), "accuracy": round(float(A["correct"][m].mean()), 4) if m.sum() else None}
    res[label] = out
    print(f"\n== {label}  (scale {out['scale']})")
    for g, v in out["by_gap"].items(): print(f"  gap {g:<5} n={v['pairs']:<5} acc {100*v['accuracy']:.1f} [{100*v['accuracy_ci'][0]:.1f},{100*v['accuracy_ci'][1]:.1f}]  confident {100*v['confident_share']:.0f}% n={v['confident_pairs']:<4} acc {100*v['confident_accuracy']:.1f} [{100*v['confident_ci'][0]:.1f},{100*v['confident_ci'][1]:.1f}]")
    print("  bands", {b: (v["pairs"], v["accuracy"]) for b, v in out["by_confidence"].items()})
labels = list(raw); b = raw[labels[0]]; res["_versus_" + labels[0]] = {}
for lb in labels[1:]:
    a = raw[lb]; J = b["key"].with_row_index("ib").join(a["key"].with_row_index("ia"), on=["post_id", "post_id_b"]); ib, ia = J["ib"].to_numpy(), J["ia"].to_numpy()
    cb, ca, fb, fa, auth, dd = b["correct"][ib], a["correct"][ia], b["conf"][ib] >= 0.6, a["conf"][ia] >= 0.6, b["auth"][ib], np.abs(b["d"][ib])
    r = {"pairs": int(len(ib))}
    for g in (1.25, 2):
        m = dd >= np.log(g)
        def cg(i, m=m): return (ca[m][i][fa[m][i]].mean() if fa[m][i].sum() else np.nan) - (cb[m][i][fb[m][i]].mean() if fb[m][i].sum() else np.nan)
        r[f"{g}x"] = {"accuracy_gap": round(float(ca[m].mean() - cb[m].mean()), 4), "ci": [round(x, 4) for x in CR.cluster_ci(auth[m], lambda i, m=m: ca[m][i].mean() - cb[m][i].mean())],
                      "confident_gap": round(float(cg(np.arange(m.sum()))), 4), "confident_ci": [round(x, 4) for x in CR.cluster_ci(auth[m], cg)]}
    res["_versus_" + labels[0]][lb] = r
    print(f"{lb} minus {labels[0]} ({r['pairs']} pairs): " + "  ".join(f"{g}: all {100*v['accuracy_gap']:+.1f} [{100*v['ci'][0]:+.1f},{100*v['ci'][1]:+.1f}] confident {100*v['confident_gap']:+.1f} [{100*v['confident_ci'][0]:+.1f},{100*v['confident_ci'][1]:+.1f}]" for g, v in r.items() if g != "pairs"))
json.dump(res, open(args.out, "w"), indent=1)
