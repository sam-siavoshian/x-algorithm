#!/usr/bin/env python3
"""Selective track: validation. Account-grouped 5-fold, repeated, on pre-cutoff pairs only.

    python scripts/selective/sel_cv.py --train dev_ft10 [--train launch_pre_start10 ...] --eval-kind same_topic|launch|all --out data/sel/cv_x.json

Every configuration (method x features x weighting) is fitted on the training folds (Platt scale from inner folds) and
scored on the held-out accounts. Metric: mean accuracy of the surest 20/30/40/50% of held-out pairs 2x apart.
"""
import argparse, json, sys, time
import numpy as np, polars as pl
sys.path.insert(0, "scripts/selective")
import sel_lib as L
import sel_models as M
import torch
torch.set_num_threads(1)

ap = argparse.ArgumentParser()
ap.add_argument("--train", action="append", required=True)
ap.add_argument("--eval-kind", default="all")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--only", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()
F = pl.concat([pl.read_parquet(f"data/sel/feat_{t}.parquet").with_columns(pl.lit(t).alias("set")) for t in a.train], how="diagonal_relaxed")
fcols = [c[2:] for c in F.columns if c.startswith("f_")]
X = {c: F["f_" + c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in fcols}
y, d, auth = F["correct"].to_numpy(), F["d"].to_numpy(), F["author_id"].to_numpy()
ev = np.ones(len(y), bool) if a.eval_kind == "all" else (X["kind_launch"] == (1.0 if a.eval_kind == "launch" else 0.0))
HIST = M.HIST
ALL = M.all_features(X)
print(len(y), "pairs", len(np.unique(auth)), "accounts; evaluated:", int(ev.sum()), "| 2x:", int((ev & (np.abs(d) >= np.log(2))).sum()), "| features", len(ALL), flush=True)

CFG = M.configs(ALL, HIST)
if a.only:
    CFG = {k: v for k, v in CFG.items() if any(k.startswith(p) for p in a.only.split(","))}

res, oof = {}, {}
for wname in ("gap", "flat", "2x"):
    w = M.weights(d, wname)
    for name, make in CFG.items():
        if name.startswith("mlp") and wname != "gap":      # the network is slow to refit 75 times; one weighting
            continue
        t0 = time.time()
        P = np.zeros((a.repeats, len(y)))
        for r in range(a.repeats):
            fo = M.folds(auth, 5, r)
            for k in range(5):
                tr, te = fo != k, fo == k
                Xtr = {c: v[tr] for c, v in X.items()}
                m, cal = M.fit_calibrated(make, Xtr, y[tr], w[tr], auth[tr], seed=r * 10 + k)
                P[r, te] = M.predict(m, cal, {c: v[te] for c, v in X.items()})
        key = f"{name}|{wname}"
        oof[key] = P
        ye, de, ae = y[ev], d[ev], auth[ev]
        big = np.abs(de) >= np.log(2)
        au = float(np.mean([L.aurc(P[r][ev], ye, de) for r in range(a.repeats)]))
        c40 = float(np.mean([ye[big][L.top_mask(P[r][ev][big], 0.4)].mean() for r in range(a.repeats)]))
        pm = P.mean(0)[ev]
        ll = float(-np.mean(ye * np.log(pm + 1e-9) + (1 - ye) * np.log(1 - pm + 1e-9)))
        sp = pm[big] >= 0.6
        res[key] = {"aurc_2x": au, "cov40_2x": c40, "logloss": ll, "speak60_share_2x": float(sp.mean()), "speak60_acc_2x": float(ye[big][sp].mean()) if sp.sum() else None,
                    "seconds": round(time.time() - t0, 1)}
        print(f"{key:<28} aurc2x {100 * au:5.1f}  cov40 {100 * c40:5.1f}  logloss {ll:.4f}  speak60 {100 * sp.mean():4.0f}% at {100 * (ye[big][sp].mean() if sp.sum() else 0):5.1f}  ({time.time() - t0:.0f}s)", flush=True)

# paired against the scale rule (same weighting "gap"): difference in the validation metric, accounts resampled
base = oof["scale|gap"]
ye, de, ae = y[ev], d[ev], auth[ev]
big = np.abs(de) >= np.log(2)
yb, ab = ye[big], ae[big]


def cols(P):
    out = []
    for r in range(a.repeats):
        for c in (0.2, 0.3, 0.4, 0.5):
            m = L.top_mask(P[r][ev][big], c)
            out += [yb * m, m.astype(float)]
    return out


nb = len(cols(base)) // 2
for key, P in oof.items():
    if key == "scale|gap":
        continue
    cc = cols(P) + cols(base)
    stat = lambda *s: np.mean([s[2 * i] / s[2 * i + 1] for i in range(nb)], 0) - np.mean([s[2 * (nb + i)] / s[2 * (nb + i) + 1] for i in range(nb)], 0)      # noqa: E731
    ci = L.boot(ab, cc, stat)
    res[key]["minus_scale"] = res[key]["aurc_2x"] - res["scale|gap"]["aurc_2x"]
    res[key]["minus_scale_ci"] = ci
best = sorted(res.items(), key=lambda kv: -kv[1]["aurc_2x"])
print("\nranked by validation metric (difference to the scale rule, 95% interval):")
for key, r in best[:12]:
    print(f"{key:<28} {100 * r['aurc_2x']:5.1f}  " + (f"{100 * r['minus_scale']:+.1f} ({100 * r['minus_scale_ci'][0]:+.1f} to {100 * r['minus_scale_ci'][1]:+.1f})" if "minus_scale" in r else "-"))
json.dump({"train": a.train, "eval_kind": a.eval_kind, "pairs": int(ev.sum()), "pairs_2x": int(big.sum()), "accounts": int(len(np.unique(ae))), "results": res}, open(a.out, "w"), indent=1)
np.savez_compressed(a.out.replace(".json", "_oof.npz"), **{k.replace("|", "__"): v.astype(np.float32) for k, v in oof.items()}, y=y, d=d, auth=auth, ev=ev)
