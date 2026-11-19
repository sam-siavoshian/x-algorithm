#!/usr/bin/env python3
"""Selective track: fit the chosen configurations on all pre-cutoff training pairs, score the test sets once, export the rule.

    python scripts/selective/sel_final.py --kind same_topic --ens ft10 --train dev --test bench --cv data/sel/cv_dev_ft10.json --out data/sel/final_same_topic_ft10

Methods scored: the scale rule (the current rule; flat weights, as the product fits it), and the best configuration of each
family (linear, gbm, mlp) by the validation metric in --cv. Also: risk-control thresholds from out-of-fold training predictions.
Writes <out>.json (all tables), <out>.md, <out>_pairs.parquet (per-pair probabilities) and <out>_rule.json (model specs).
"""
import argparse, json, sys
import numpy as np, polars as pl
from scipy.stats import binom
sys.path.insert(0, "scripts/selective")
import sel_lib as L
import sel_models as M
from xviral import selective as S

ap = argparse.ArgumentParser()
ap.add_argument("--kind", required=True)
ap.add_argument("--ens", required=True)
ap.add_argument("--train", required=True, help="comma-separated training sets (feat_<set>_<ens>.parquet)")
ap.add_argument("--test", required=True)
ap.add_argument("--cv", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--train-part", default=None, help="secondary configuration: also train on this part of the test set and test on the rest (e.g. calibration)")
ap.add_argument("--families", default="linear,gbm,mlp")
a = ap.parse_args()


def load(names):
    F = pl.concat([pl.read_parquet(f"data/sel/feat_{n}_{a.ens}.parquet").with_columns(pl.lit(n).alias("set")) for n in names], how="diagonal_relaxed")
    X = {c[2:]: F[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in F.columns if c.startswith("f_")}
    return F, X, F["correct"].to_numpy(), F["d"].to_numpy(), F["author_id"].to_numpy()


Ftr, Xtr, ytr, dtr, atr = load(a.train.split(","))
Fte, Xte, yte, dte, ate = load([a.test])
if a.train_part:
    k = (Fte["part"] == a.train_part).to_numpy()
    Xtr = {c: np.concatenate([Xtr[c], Xte[c][k]]) for c in Xtr}
    ytr, dtr, atr = np.concatenate([ytr, yte[k]]), np.concatenate([dtr, dte[k]]), np.concatenate([atr, ate[k]])
    Xte = {c: v[~k] for c, v in Xte.items()}
    yte, dte, ate, Fte = yte[~k], dte[~k], ate[~k], Fte.filter(pl.Series(~k))
ALL = M.all_features(Xtr)
CFG = M.configs(ALL, M.HIST)
cv = json.load(open(a.cv))["results"]
chosen = {"scale": "scale|flat"}
for fam in [f for f in a.families.split(",") if f]:
    cand = {k: v for k, v in cv.items() if k.startswith(fam)}
    chosen[fam] = max(cand, key=lambda k: cand[k]["aurc_2x"])
best = max((f for f in chosen if f != "scale"), key=lambda f: cv[chosen[f]]["aurc_2x"]) if len(chosen) > 1 else "scale"
ships = cv[chosen[best]].get("minus_scale_ci", [0, 0])[0] > 0
print("chosen:", chosen, "| best family:", best, "| replaces the scale rule by the pre-registered test:", ships, flush=True)

out = {"kind": a.kind, "ensemble": a.ens, "train": a.train, "test": a.test, "train_pairs": int(len(ytr)), "train_accounts": int(len(np.unique(atr))),
       "test_pairs": int(len(yte)), "test_accounts": int(len(np.unique(ate))), "chosen": chosen, "best_family": best, "replaces_scale_rule": bool(ships),
       "validation": {f: cv[k] for f, k in chosen.items()}, "methods": {}}
P, specs, oofs = {}, {}, {}
for fam, key in chosen.items():
    name, wname = key.split("|")
    w = M.weights(dtr, wname)
    m, cal = M.fit_calibrated(CFG[name], Xtr, ytr, w, atr)
    P[fam] = M.predict(m, cal, Xte)
    specs[fam] = m.spec(cal)
    # the exported JSON must give the same numbers as the fitted model
    rows = [{c: (float(Xte[c][i]) if not np.isnan(Xte[c][i]) else float("nan")) for c in Xte} for i in range(min(300, len(yte)))]
    pj = np.array([np.nan if (v := S.predict(specs[fam], r)) is None else v for r in rows])      # None: too many inputs missing, the product falls back
    ok = ~np.isnan(pj)
    assert ok.mean() > 0.9 and np.allclose(pj[ok], P[fam][:len(rows)][ok], atol=1e-5), (fam, float(ok.mean()), float(np.abs(pj - P[fam][:len(rows)])[ok].max()))
    # out-of-fold probabilities on the training pairs, for risk control
    fo = M.folds(atr, 5, 0)
    z = np.zeros(len(ytr))
    for j in range(5):
        tr, te = fo != j, fo == j
        mm, cc = M.fit_calibrated(CFG[name], {c: v[tr] for c, v in Xtr.items()}, ytr[tr], w[tr], atr[tr], seed=j)
        z[te] = M.predict(mm, cc, {c: v[te] for c, v in Xtr.items()})
    oofs[fam] = z
    out["methods"][fam] = {"config": key, "curve": L.curve(P[fam], yte, ate, dte), "speak60": L.at_threshold(P[fam], yte, ate, dte),
                           "logloss": float(-np.mean(yte * np.log(P[fam] + 1e-9) + (1 - yte) * np.log(1 - P[fam] + 1e-9))),
                           "train_oof_curve": L.curve(z, ytr, atr, dtr), "train_oof_speak60": L.at_threshold(z, ytr, atr, dtr),
                           "train_in_sample_curve": L.curve(M.predict(m, cal, Xtr), ytr, atr, dtr)}
    if fam != "scale":
        out["methods"][fam]["minus_scale"] = {f"{g}x_cov{int(c * 100)}": L.paired_cov(P[fam], P["scale"], yte, yte, ate, dte, c, g) for g in (1.25, 2.0) for c in (0.2, 0.3, 0.4, 0.5)}
    if fam == "linear":
        out["methods"][fam]["coefficients"] = dict(zip(specs[fam]["features"], specs[fam]["coef"]))
    if fam == "gbm":
        imp = dict(zip(m.feats, m.m.booster_.feature_importance("gain").tolist()))
        out["methods"][fam]["gain_top"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:12])

# reference: the earlier tracks' convention, scale cross-fitted over accounts on the test pairs themselves
fo = M.folds(ate, 5, 0)
px = np.zeros(len(yte))
for j in range(5):
    px[fo == j] = M.sigmoid(M.fit_scale(Xte["gap"][fo != j], yte[fo != j])["scale"] * Xte["gap"][fo == j])
out["reference_scale_crossfitted_on_test"] = {"speak60": L.at_threshold(px, yte, ate, dte)}

# --- risk control on the best score and on the scale rule: thresholds from training out-of-fold probabilities, pairs 2x apart ---
big_tr, big_te = np.abs(dtr) >= np.log(2), np.abs(dte) >= np.log(2)
out["risk_control"] = {}
for fam in dict.fromkeys(("scale", best)):
    p_tr, c_tr, p_te, c_te = oofs[fam][big_tr], ytr[big_tr], P[fam][big_te], yte[big_te]
    grid = np.round(np.arange(0.95, 0.4999, -0.005), 3)
    res = {}
    for alpha in (0.25, 0.30):
        lam = None                                       # fixed-sequence test from the strictest threshold down (learn then test), delta 0.1
        for g in grid:
            m = p_tr >= g
            n, k = int(m.sum()), int((1 - c_tr[m]).sum())
            if n < 30:
                continue
            if binom.cdf(k, n, alpha) <= 0.10:
                lam = float(g)
            else:
                break
        r = {"threshold": lam}
        if lam is not None:
            mt, m0 = p_te >= lam, p_tr >= lam
            r |= {"train_share": float(m0.mean()), "train_accuracy": float(c_tr[m0].mean()), "test_share": float(mt.mean()), "test_pairs": int(mt.sum()),
                  "test_accuracy": float(c_te[mt].mean()) if mt.sum() else None, "test_ci": L.boot(ate[big_te], [c_te * mt, mt], lambda x, y: x / y) if mt.sum() else None,
                  "held": bool(mt.sum() and 1 - c_te[mt].mean() <= alpha)}
        res[f"ltt_error_{alpha}"] = r
    for beta in (0.10, 0.12):                            # conformal risk control on the monotone loss "spoken and wrong" (per pair 2x apart)
        n, lam = len(p_tr), None
        for g in grid[::-1]:                             # from the loosest threshold up: the smallest that satisfies the bound
            if (n / (n + 1)) * float(((p_tr >= g) & (c_tr == 0)).mean()) + 1 / (n + 1) <= beta:
                lam = float(g)
                break
        r = {"threshold": lam}
        if lam is not None:
            mt = p_te >= lam
            r |= {"train_share": float((p_tr >= lam).mean()), "test_share": float(mt.mean()), "test_spoken_and_wrong": float((mt & (c_te == 0)).mean()),
                  "test_accuracy": float(c_te[mt].mean()) if mt.sum() else None, "held": bool(float((mt & (c_te == 0)).mean()) <= beta)}
        res[f"crc_spoken_wrong_{beta}"] = r
    out["risk_control"][fam] = res

# --- stability: halves of the benchmark, follower tiers, accounts with and without training pairs ---
groups = {}
if "part" in Fte.columns:
    for v in Fte["part"].unique().to_list():
        groups[f"half: {v}"] = (Fte["part"] == v).to_numpy()
fol = Xte["log_followers"]
groups |= {"followers under 10k": fol < 4, "followers 10k to 100k": (fol >= 4) & (fol < 5), "followers 100k and over": fol >= 5, "followers unknown": np.isnan(fol)}
seen = np.isin(ate, np.unique(atr))
groups |= {"account has training pairs": seen, "account has no training pairs": ~seen}
if "f_days_apart" in Fte.columns and a.kind == "launch":
    da = Xte["days_apart"]
    groups |= {"posts within 30 days": da <= 30, "posts 30 to 180 days apart": (da > 30) & (da <= 180), "posts over 180 days apart": da > 180}
out["stability"] = {}
for gname, gm in groups.items():
    gm = np.asarray(gm, bool)
    if gm.sum() < 80:
        continue
    out["stability"][gname] = {fam: {"curve": L.curve(P[fam][gm], yte[gm], ate[gm], dte[gm], covs=(0.4,)), "speak60": L.at_threshold(P[fam][gm], yte[gm], ate[gm], dte[gm])}
                               for fam in dict.fromkeys(("scale", best))}

json.dump(out, open(a.out + ".json", "w"), indent=1)
json.dump({"kind": a.kind, "ensemble": a.ens, "chosen": chosen, "best_family": best, "replaces_scale_rule": bool(ships), "specs": specs,
           "train_pairs": int(len(ytr)), "train": a.train}, open(a.out + "_rule.json", "w"))
Fte.select([c for c in ("post_id", "post_id_b", "author_id", "d", "correct", "part") if c in Fte.columns]).with_columns(
    [pl.Series(f"p_{fam}", P[fam]) for fam in P]).write_parquet(a.out + "_pairs.parquet")

pc = lambda x: "-" if x is None else f"{100 * x:.1f}%"      # noqa: E731
ci = lambda c: "" if c is None else f" ({100 * c[0]:.1f} to {100 * c[1]:.1f})"      # noqa: E731
lines = [f"### {a.kind}, calls by {a.ens}: trained on {a.train} ({len(ytr):,} pairs, {len(np.unique(atr))} accounts), tested on {a.test} ({len(yte):,} pairs, {len(np.unique(ate))} accounts)", "",
         "| Method (configuration) | Real gap at least | Pairs | All calls | Surest 20% | 30% | 40% | 50% | Stated 60%+: share | Correct there |", "|---|---|---|---|---|---|---|---|---|---|"]
for fam, r in out["methods"].items():
    for g in ("1.25x", "2.0x"):
        c, s = r["curve"][g], r["speak60"][g]
        lines.append(f"| {fam} ({r['config']}) | {g} | {c['pairs']:,} | {pc(c['accuracy'])} | " + " | ".join(f"{pc(c[f'cov{k}']['accuracy'])}{ci(c[f'cov{k}']['ci'])}" for k in (20, 30, 40, 50))
                     + f" | {pc(s['share'])} ({s['spoken']}) | {pc(s['accuracy'])}{ci(s['ci'])} |")
lines += ["", "Difference to the scale rule, accuracy of the surest share of pairs (points, 95% interval from resampling accounts):", "",
          "| Method | Gap | 20% | 30% | 40% | 50% |", "|---|---|---|---|---|---|"]
for fam, r in out["methods"].items():
    if fam == "scale":
        continue
    for g in (1.25, 2.0):
        lines.append(f"| {fam} | {g}x | " + " | ".join(f"{100 * r['minus_scale'][f'{g}x_cov{k}']['gap']:+.1f} ({100 * r['minus_scale'][f'{g}x_cov{k}']['ci'][0]:+.1f} to {100 * r['minus_scale'][f'{g}x_cov{k}']['ci'][1]:+.1f})" for k in (20, 30, 40, 50)) + " |")
lines += ["", "Train, validation and test side by side (pairs 2x apart, accuracy of the surest 40%):", "", "| Method | Train, in sample | Validation, accounts held out | Test |", "|---|---|---|---|"]
lines += [f"| {fam} | {pc(r['train_in_sample_curve']['2.0x']['cov40']['accuracy'])} | {pc(r['train_oof_curve']['2.0x']['cov40']['accuracy'])} | {pc(r['curve']['2.0x']['cov40']['accuracy'])}{ci(r['curve']['2.0x']['cov40']['ci'])} |" for fam, r in out["methods"].items()]
lines += ["", "Stability (pairs 2x apart; surest 40% within the group, and stated 60%+):", "", "| Group | Pairs 2x apart | Method | All calls | Surest 40% | Stated 60%+: share | Correct there |", "|---|---|---|---|---|---|---|"]
for gname, r in out["stability"].items():
    for fam, x in r.items():
        c, s = x["curve"]["2.0x"], x["speak60"]["2.0x"]
        lines.append(f"| {gname} | {c['pairs']:,} | {fam} | {pc(c['accuracy'])} | {pc(c['cov40']['accuracy'])}{ci(c['cov40']['ci'])} | {pc(s['share'])} ({s['spoken']}) | {pc(s['accuracy'])}{ci(s['ci'])} |")
lines += ["", "Risk control (thresholds set on training out-of-fold probabilities of pairs 2x apart):", "", "| Score | Rule | Threshold | Share spoken, train | Share spoken, test | Correct, test | Held on test |", "|---|---|---|---|---|---|---|"]
for fam, r in out["risk_control"].items():
    for k, x in r.items():
        lines.append(f"| {fam} | {k} | {x['threshold']} | {pc(x.get('train_share'))} | {pc(x.get('test_share'))} | {pc(x.get('test_accuracy'))}{ci(x.get('test_ci'))} | {x.get('held', '-')} |")
open(a.out + ".md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
