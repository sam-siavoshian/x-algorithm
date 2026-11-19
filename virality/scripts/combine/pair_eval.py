#!/usr/bin/env python3
"""Pair accuracy of model sets on the four pair sets, with confidence scales fitted before the cutoff only.

    python scripts/combine/pair_eval.py --ens "Label=member1,member2" [--ens ...] --base Label [--sets dev,launchpre] --out data/combine/eval/pairs_step1

An ensemble's gap is the mean of its members' gaps (log10 predicted median likes). Its confidence scale is fitted per
kind of pair on pre-cutoff pairs (dev for same-topic pairs, launchpre for launch pairs; scripts/selective/sel_models.py
fit_scale) and applied unchanged to the test sets (bench, launch): nothing is fitted or rescaled on a test set. On the pre-cutoff sets
themselves the stated confidence is cross-fitted over accounts (five folds).
Slices: all pairs 1.25x apart; 2x apart; both posts with 3 or more likes (1.25x apart); 2x apart and both posts 3+ likes.
Per slice: share correct, share of pairs stated at 60% or more and share correct there, share correct on the surest 40%
(ranked by the size of the gap, no scale), correct confident calls per 100 pairs. Intervals resample accounts; differences
to --base are paired. --sets restricts which sets are read (the test sets are read only when named).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, polars as pl
sys.path.insert(0, "scripts/selective")
from sel_models import fit_scale  # noqa: E402

KIND = {"dev": "same_topic", "bench": "same_topic", "launchpre": "launch", "launch": "launch"}
FIT = {"same_topic": "dev", "launch": "launchpre"}
L = np.log


def boot(auth, cols, f, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    ua, inv = np.unique(auth, return_inverse=True)
    S = np.stack([np.bincount(inv, weights=np.asarray(c, float), minlength=len(ua)) for c in cols])
    b = rng.integers(0, len(ua), (n, len(ua)))
    with np.errstate(all="ignore"):
        v = np.array([f(*S[:, i].sum(1)) for i in b])
    return [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]


def slices(d, lo):
    out = {"1.25x": np.abs(d) >= L(1.25), "2x": np.abs(d) >= L(2), "1.25x, both 3+ likes": (np.abs(d) >= L(1.25)) & (lo >= 3), "2x, both 3+ likes": (np.abs(d) >= L(2)) & (lo >= 3)}
    if UNTRAINED is not None and 0 < UNTRAINED.sum() < len(d):       # pre-cutoff sets: accounts whose posts no model was trained on
        out.update({"1.25x, accounts not trained on": out["1.25x"] & UNTRAINED, "2x, accounts not trained on": out["2x"] & UNTRAINED})
    return out


UNTRAINED = None


def crossfit_conf(g, d, auth, folds=5, seed=0):
    """Stated confidence with the scale fitted on the other accounts (for the pre-cutoff sets, where the scale itself is fitted)."""
    ua = np.unique(auth)
    fold = dict(zip(ua.tolist(), np.random.default_rng(seed).integers(0, folds, len(ua)).tolist()))
    fo = np.array([fold[x] for x in auth.tolist()])
    clear = np.abs(d) >= L(1.25)
    p = np.zeros(len(g))
    for f in range(folds):
        k = fit_scale(np.abs(g[(fo != f) & clear]), ((g * d) > 0)[(fo != f) & clear].astype(float))["scale"]
        p[fo == f] = 1 / (1 + np.exp(-k * np.abs(g[fo == f])))
    return p


def stats(g, d, auth, lo, k, cross=False):
    c = np.where(g == 0, 0.5, (g * d) > 0).astype(float)
    conf = (crossfit_conf(g, d, auth) if cross else 1 / (1 + np.exp(-k * np.abs(g)))) >= 0.6
    out = {}
    for nm, m in slices(d, lo).items():
        if m.sum() < 30:
            continue
        one = np.ones(int(m.sum()))
        idx = np.flatnonzero(m); top = np.zeros(len(d)); top[idx[np.argsort(-np.abs(g[idx]), kind="stable")[: int(round(.4 * len(idx)))]]] = 1
        cm, fm, tm, am = c[m], conf[m].astype(float), top[m], auth[m]
        out[nm] = {"pairs": int(m.sum()), "accounts": int(len(set(am))), "acc": float(cm.mean()), "acc_ci": boot(am, [cm, one], lambda x, n_: x / n_),
                   "conf_share": float(fm.mean()), "conf_share_ci": boot(am, [fm, one], lambda x, n_: x / n_), "conf_pairs": int(fm.sum()),
                   "conf_acc": float((cm * fm).sum() / max(fm.sum(), 1)), "conf_acc_ci": boot(am, [cm * fm, fm], lambda x, n_: x / n_),
                   "top40_acc": float((cm * tm).sum() / max(tm.sum(), 1)), "top40_ci": boot(am, [cm * tm, tm], lambda x, n_: x / n_),
                   "correct_conf_per_100": float(100 * (cm * fm).mean())}
    return out, c, conf.astype(float)


def versus(a, b, d, auth, lo, ga, gb):
    out = {}
    for nm, m in slices(d, lo).items():
        if m.sum() < 30:
            continue
        ca, cb, fa, fb, am = a[0][m], b[0][m], a[1][m], b[1][m], auth[m]
        one = np.ones(int(m.sum()))
        idx = np.flatnonzero(m); n40 = int(round(.4 * len(idx)))
        ta = np.zeros(len(d)); ta[idx[np.argsort(-np.abs(ga[idx]), kind="stable")[:n40]]] = 1
        tb = np.zeros(len(d)); tb[idx[np.argsort(-np.abs(gb[idx]), kind="stable")[:n40]]] = 1
        ta, tb = ta[m], tb[m]
        out[nm] = {"pairs": int(m.sum()), "acc": [float(ca.mean() - cb.mean())] + boot(am, [ca, cb, one], lambda x, y, n_: (x - y) / n_),
                   "conf_share": [float(fa.mean() - fb.mean())] + boot(am, [fa, fb, one], lambda x, y, n_: (x - y) / n_),
                   "conf_acc": [float((ca * fa).sum() / max(fa.sum(), 1) - (cb * fb).sum() / max(fb.sum(), 1))] + boot(am, [ca * fa, fa, cb * fb, fb], lambda p, q, r, s: p / q - r / s),
                   "top40_acc": [float((ca * ta).sum() / n40 - (cb * tb).sum() / n40)] + boot(am, [ca * ta, ta, cb * tb, tb], lambda p, q, r, s: p / q - r / s),
                   "correct_conf_per_100": [float(100 * ((ca * fa).mean() - (cb * fb).mean()))] + [100 * x for x in boot(am, [ca * fa, cb * fb, one], lambda x, y, n_: (x - y) / n_)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens", action="append", required=True); ap.add_argument("--base", required=True); ap.add_argument("--versus", action="append", default=[])
    ap.add_argument("--pairs", default="data/combine/pairs"); ap.add_argument("--sets", default="dev,launchpre,bench,launch"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ens = {x.split("=")[0]: x.split("=")[1].split(",") for x in a.ens}
    sets = a.sets.split(",")
    T = {s: pl.read_parquet(f"{a.pairs}/{s}.parquet") for s in set(sets) | set(FIT.values())}

    def gap(s, members):
        return np.mean([T[s]["s3"].to_numpy() if m == "s3" else T[s][f"g_{m}"].to_numpy() for m in members], axis=0)
    res = {"ensembles": ens, "scales": {}, "sets": {}, "versus": {}}
    for lab, mem in ens.items():
        res["scales"][lab] = {}
        for kind, s in FIT.items():
            g, d = gap(s, mem), T[s]["d"].to_numpy()
            m = np.abs(d) >= L(1.25)
            res["scales"][lab][kind] = fit_scale(np.abs(g[m]), ((g * d) > 0)[m].astype(float))["scale"]
    md = ["Confidence scales fitted on pre-cutoff pairs: " + "; ".join(f"{lab}: same-topic {v['same_topic']:.2f}, launch {v['launch']:.2f}" for lab, v in res["scales"].items()), ""]
    for s in sets:
        d, auth = T[s]["d"].to_numpy(), T[s]["author_id"].to_numpy()
        lo = np.minimum(T[s]["likes"].to_numpy(), T[s]["likes_b"].to_numpy())
        global UNTRAINED
        UNTRAINED = ~T[s]["acct_in_train"].to_numpy() if s in FIT.values() and "acct_in_train" in T[s].columns else None
        R, keep = {}, {}
        for lab, mem in ens.items():
            g = gap(s, mem)
            R[lab], c, f = stats(g, d, auth, lo, res["scales"][lab][KIND[s]], cross=s in FIT.values())
            keep[lab] = (c, f, g)
        res["sets"][s] = R
        md += [f"### {s} ({'pre-cutoff: validation' if s in ('dev', 'launchpre') else 'test'})", "",
               "| Model set | Slice | Pairs | Correct | Stated 60%+: share (pairs) | Correct there | Surest 40% correct | Correct confident calls per 100 |", "|---|---|---|---|---|---|---|---|"]
        for lab in ens:
            for nm, r in R[lab].items():
                md.append(f"| {lab} | {nm} | {r['pairs']:,} | {100 * r['acc']:.1f}% ({100 * r['acc_ci'][0]:.1f} to {100 * r['acc_ci'][1]:.1f}) | {100 * r['conf_share']:.0f}% ({r['conf_pairs']}) | "
                          f"{100 * r['conf_acc']:.1f}% ({100 * r['conf_acc_ci'][0]:.1f} to {100 * r['conf_acc_ci'][1]:.1f}) | {100 * r['top40_acc']:.1f}% ({100 * r['top40_ci'][0]:.1f} to {100 * r['top40_ci'][1]:.1f}) | {r['correct_conf_per_100']:.1f} |")
        pairs_ = [(lab, a.base) for lab in ens if lab != a.base] + [tuple(v.split(",")) for v in a.versus]
        md += ["", "| A minus B | Slice | Pairs | Correct, points | Stated 60%+ share, points | Correct there, points | Surest 40%, points | Correct confident calls per 100 |", "|---|---|---|---|---|---|---|---|"]
        f3 = lambda v, k=100: f"{k * v[0]:+.1f} ({k * v[1]:+.1f} to {k * v[2]:+.1f})"      # noqa: E731
        for x, y in pairs_:
            V = versus(keep[x][:2], keep[y][:2], d, auth, lo, keep[x][2], keep[y][2])
            res["versus"][f"{s}|{x}|{y}"] = V
            for nm, r in V.items():
                md.append(f"| {x} minus {y} | {nm} | {r['pairs']:,} | {f3(r['acc'])} | {f3(r['conf_share'])} | {f3(r['conf_acc'])} | {f3(r['top40_acc'])} | {f3(r['correct_conf_per_100'], 1)} |")
        md.append("")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
