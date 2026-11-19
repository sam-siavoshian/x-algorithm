#!/usr/bin/env python3
"""Fit the product rule on development pairs only (before 2026-04-23), near-identical pairs removed.

    python scripts/judge_product/fit_rule.py          # writes data/llm/jp_fit.json, jp_fit.md and xviral/data/judge_rule.json

Per kind of pair (same_topic: the 2,805 development pairs; launch: the 3,655 launch pairs from before the cutoff) and per
judge model:
  1. choices, made with the model fitted on the `fit` cell (both posts before T, account half A) and read on the `val` cell
     (later post on or after T, account half B: later in time AND other accounts): the clip of the judge's log-odds, and the
     speak rule;
  2. the shipped weights: refitted on every clean development pair the judge scored, same clip and speak rule.
Inputs are on fixed scales: the surrogate's gap in log10 predicted likes (serving path), the judge's order-averaged
log-odds clipped at +-c. Nothing is rescaled per pair set. The test sets are not read here."""
import json, sys, os
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(__file__))
from jp import *

# xviral/data/selective_rule.json, base3 (fitted on pre-cutoff pairs). JP_SURR_SCALE='{"same_topic": k, "launch": k}' gives another
# model set's scales, JP_SURROGATE its description for the rule file (research/tracks/combine).
SURR_SCALE = json.loads(os.environ.get("JP_SURR_SCALE") or '{"same_topic": 2.77, "launch": 1.84}')
SETS = {"same_topic": "dev", "launch": "launchpre"}
CLIPS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 14.0)
# Candidates start at 60%: the product never recommends below a stated 60% (its documented threshold), whatever a looser
# rule would score on validation.
RULES = {"p>=0.60": dict(p=.60), "p>=0.65": dict(p=.65), "p>=0.70": dict(p=.70),
         "p>=0.60 and the two agree": dict(p=.60, require_agreement=True), "the two agree": dict(p=.5, require_agreement=True),
         "p>=0.60 and (agree or judge alone >=0.70)": dict(p=.60, require_agreement=True, or_judge_p=.70)}
md = []


def say(*x):
    s = " ".join(str(v) for v in x); print(s); md.append(s)


def fit(sg, j, y):
    X = np.stack([sg, j], 1)
    m = LogisticRegression(C=1e4, fit_intercept=False).fit(np.vstack([X, -X]), np.concatenate([y, 1 - y]).astype(int))
    return m.coef_[0]


def fit1(x, y):
    m = LogisticRegression(C=1e4, fit_intercept=False).fit(np.concatenate([x, -x])[:, None], np.concatenate([y, 1 - y]).astype(int))
    return float(m.coef_[0][0])


def nll(z, y):
    p = np.clip(sig(z), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def speak_mask(rule, z, sg, j, kj):
    conf = np.maximum(sig(z), 1 - sig(z)); m = conf >= rule["p"]
    if rule.get("require_agreement"):
        agree = (sg * j) > 0
        alt = np.zeros(len(z), bool)
        if rule.get("or_judge_p"):
            pj = sig(kj * j); alt = (np.maximum(pj, 1 - pj) >= rule["or_judge_p"]) & ((j > 0) == (z > 0))
        m = m & (agree | alt)
    return m


def evaluate(z, d, a, spoken, gaps=(1.25, 2, 3)):
    """Per gap: all calls, share spoken, accuracy when spoken, correct spoken calls per 100 pairs; account-resampled intervals."""
    c = correct(z, d); out = {}
    for g in gaps:
        m = np.abs(d) >= np.log(g)
        if m.sum() < 20:
            continue
        sp = m & spoken
        cov = rate(spoken.astype(float), m, a)
        out[f"{g}x"] = {"pairs": int(m.sum()), "acc": rate(c, m, a), "spoken_share": cov, "spoken_pairs": int(sp.sum()), "spoken_acc": rate(c, sp, a),
                        "correct_spoken_per_100": 100 * float((c * sp)[m].mean())}
    return out


def row(lab, e, g="2x"):
    r = e.get(g)
    if not r:
        return f"| {lab} | {g} | - | - | - | - |"
    return f"| {lab} | {g} | {r['pairs']:,} | {fmt_rate(r['acc'])} | {100 * r['spoken_share'][1]:.0f}% ({100 * r['spoken_share'][2]:.0f} to {100 * r['spoken_share'][3]:.0f}; {r['spoken_pairs']}) | {fmt_rate(r['spoken_acc'])} |"


def main():
    R = {"kinds": {}}
    rule_file = {"version": 1, "fitted": "2026-11-19, research/tracks/judge_product/REPORT.md, scripts/judge_product/fit_rule.py",
                 "surrogate": os.environ.get("JP_SURROGATE") or "base3: ts46_s1, ts46_s2, compare_v2/full (bge-small); a rule belongs to the forecasting models it was fitted with",
                 "inputs": {"gap": "forecasting model, first version minus second, log10 of predicted likes (mean over members of log10(median + 1))",
                            "judge": "order-averaged log-odds that the first version does better, clipped at +-clip",
                            "history_posts": 30, "prompt": "xviral/judge.py SYSTEM, both orders"},
                 "min_history_posts": 0, "models": {}}
    for kind, name in SETS.items():
        P = table(name).filter(pl.col("sim") < 0.9)
        say(f"\n## {kind}: {name}, {P.height} clean development pairs (near-identical removed), {P['author_id'].n_unique()} accounts\n")
        R["kinds"][kind] = {}
        judges = [c for c in ("luna", "sol", "dsflash", "qwen", "oss") if c in P.columns and P[c].is_not_null().sum() >= 250]
        for jn in judges:
            Q = P.filter(pl.col(jn).is_not_null())
            d, a, sg, jr, cell = Q["d"].to_numpy(), Q["author_id"].to_numpy(), Q["sg"].to_numpy(), Q[jn].to_numpy(), Q["cell"].to_numpy()
            y = (d > 0).astype(float)
            F, V = cell == "fit", cell == "val"
            res = {"pairs": Q.height, "fit_pairs": int(F.sum()), "val_pairs": int(V.sum()), "accounts": int(len(set(a)))}
            say(f"\n### {kind}, judge {jn}: {Q.height} pairs scored (fit cell {int(F.sum())}, validation cell {int(V.sum())}, other {int((~F & ~V).sum())})\n")
            if F.sum() < 100 or V.sum() < 100:
                # too few pairs in the strict cells for this judge (a sample drawn from the validation cell only): split the validation cell by account instead
                ua = np.unique(a[V]); rng = np.random.default_rng(3); pick = set(rng.choice(ua, len(ua) // 2, replace=False).tolist())
                F = V & np.array([x in pick for x in a]); V = V & ~F
                res["note"] = "sample drawn from the validation cell only; fitted on half of its accounts, read on the other half"
                say(f"(sample from the validation cell only: fitted on {int(F.sum())} pairs of half its accounts, read on the other {int(V.sum())})\n")
            # 1a. clip, chosen by validation log-loss
            cl = {}
            for c in CLIPS:
                j = np.clip(jr, -c, c); w = fit(sg[F], j[F], y[F])
                cl[c] = {"w": w.tolist(), "val_nll": nll(sg[V] * w[0] + j[V] * w[1], y[V]), "fit_nll": nll(sg[F] * w[0] + j[F] * w[1], y[F])}
            c = min(cl, key=lambda k: cl[k]["val_nll"])
            j = np.clip(jr, -c, c); w = np.array(cl[c]["w"])
            ks, kj = fit1(sg[F], y[F]), fit1(j[F], y[F])
            res["clip_search"] = {str(k): v for k, v in cl.items()}; res["clip"] = c
            say("clip of the judge's log-odds: validation log-loss " + ", ".join(f"{k:g}: {v['val_nll']:.4f}" for k, v in cl.items()) + f" -> clip {c:g}")
            say(f"weights fitted on the fit cell: surrogate {w[0]:.3f} per log10, judge {w[1]:.3f} per unit of clipped log-odds. Validation log-loss: blend {cl[c]['val_nll']:.4f}, "
                f"surrogate alone (scale {ks:.2f}) {nll(ks * sg[V], y[V]):.4f}, judge alone (scale {kj:.2f}) {nll(kj * j[V], y[V]):.4f}, coin 0.6931")
            # 1b. train / validation side by side, and the speak rule on validation
            z = sg * w[0] + j * w[1]
            say("\n| Scored by | Cell | Pairs 1.25x apart | Correct | Pairs 2x apart | Correct |"); say("|---|---|---|---|---|---|")
            side = {}
            for lab, s_ in (("surrogate", sg), ("judge", j), ("blend", z)):
                for cn, M in (("fit (train)", F), ("validation", V)):
                    r1, r2 = rate(correct(s_, d), M, a), rate(correct(s_, d), M & (np.abs(d) >= L2), a)
                    side[f"{lab}|{cn}"] = {"1.25x": r1, "2x": r2}
                    say(f"| {lab} | {cn} | {r1[0]} | {fmt_rate(r1)} | {r2[0]} | {fmt_rate(r2)} |")
            res["train_validation"] = side
            dv, av = d[V], a[V]
            base = evaluate(sg[V], dv, av, np.maximum(sig(SURR_SCALE[kind] * sg[V]), 1 - sig(SURR_SCALE[kind] * sg[V])) >= .6)
            say(f"\nSpeak rules on the validation cell (model fitted on the fit cell):\n\n| Rule | Real gap at least | Pairs | All calls correct | Spoken share (interval; pairs) | Correct when spoken |\n|---|---|---|---|---|---|")
            say(row("surrogate alone, shipped scale, p>=0.60", base)); say(row("surrogate alone, shipped scale, p>=0.60", base, "1.25x"))
            cand = {}
            for rn, rl in RULES.items():
                e = evaluate(z[V], dv, av, speak_mask(rl, z[V], sg[V], j[V], kj)); cand[rn] = e
                say(row(f"blend, {rn}", e)); say(row(f"blend, {rn}", e, "1.25x"))
            res["val_rules"] = {"surrogate": base, **cand}
            ok = {k: v for k, v in cand.items() if v.get("2x") and v["2x"]["spoken_acc"][1] is not None and v["2x"]["spoken_acc"][1] >= .75}
            if ok:
                best = max(ok, key=lambda k: ok[k]["2x"]["correct_spoken_per_100"])
                if ok.get("p>=0.60") and ok["p>=0.60"]["2x"]["correct_spoken_per_100"] >= ok[best]["2x"]["correct_spoken_per_100"] - 1.0:
                    best = "p>=0.60"
                why = "most correct spoken calls per 100 pairs 2x apart among rules at 75% or better on validation (the plain 60% threshold wins ties within 1 per 100)"
            else:
                big = {k: v for k, v in cand.items() if v.get("2x") and v["2x"]["spoken_share"][1] >= .4}
                best = max(big, key=lambda k: big[k]["2x"]["spoken_acc"][1]) if big else "p>=0.60"
                why = "no rule reaches 75% on validation pairs 2x apart; the most accurate rule that speaks on 40% or more of them"
            res["speak_rule"], res["speak_why"] = best, why
            say(f"\nChosen speak rule: **{best}** ({why}).")
            # calibration of the blend on validation
            pz = np.maximum(sig(z[V]), 1 - sig(z[V])); cz = correct(z[V], dv)
            cal = [{"stated": f"{lo:.2f} to {hi:.2f}", "pairs": int(((pz >= lo) & (pz < hi)).sum()), "correct": float(cz[(pz >= lo) & (pz < hi)].mean()) if ((pz >= lo) & (pz < hi)).sum() else None}
                   for lo, hi in ((.5, .55), (.55, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01))]
            res["val_calibration"] = cal
            say("Stated probability against outcome on validation: " + "; ".join(f"{x['stated']}: {100 * x['correct']:.0f}% of {x['pairs']}" for x in cal if x["pairs"]))
            ag = (sg[V] * j[V]) > 0
            res["val_agreement"] = {"agree_share": float(ag.mean()), "right_when_agree": float(correct(sg[V], dv)[ag].mean()), "surrogate_right_when_disagree": float(correct(sg[V], dv)[~ag].mean()),
                                    "judge_right_when_disagree": float(correct(j[V], dv)[~ag].mean())}
            q = res["val_agreement"]
            say(f"Agreement on validation: same side on {100 * q['agree_share']:.0f}% of pairs; right {100 * q['right_when_agree']:.1f}% when they agree; when they disagree the surrogate is right {100 * q['surrogate_right_when_disagree']:.1f}%, the judge {100 * q['judge_right_when_disagree']:.1f}%.")
            # history available to the judge
            nh = Q["n_hist"].to_numpy()
            res["by_history"] = {lab: {"pairs": int(m.sum()), "judge": float(correct(j, d)[m].mean()), "surrogate": float(correct(sg, d)[m].mean())} for lab, m in (("under 10 earlier posts", nh < 10), ("10 to 29", (nh >= 10) & (nh < 30)), ("30 or more", nh >= 30)) if m.sum() >= 30}
            say("By earlier posts available (all cells): " + "; ".join(f"{k}: judge {100 * v['judge']:.1f}%, surrogate {100 * v['surrogate']:.1f}% ({v['pairs']} pairs)" for k, v in res["by_history"].items()))
            # 2. shipped weights: all clean development pairs
            wf = fit(sg, j, y); ksf, kjf = fit1(sg, y), fit1(j, y)
            res["weights_all"] = wf.tolist()
            say(f"Shipped weights (all {Q.height} clean development pairs the judge scored): surrogate {wf[0]:.3f}, judge {wf[1]:.3f}, clip {c:g}; judge-alone scale {kjf:.3f}.")
            model = {"luna": "gpt-6-luna", "sol": "gpt-6.1-sol", "dsflash": "deepseek-ai/DeepSeek-V4.1-Flash", "qwen": "Qwen/Qwen3.5-397B-A17B", "oss": "openai/gpt-oss-120b"}[jn]
            v2 = cand[best].get("2x") or {}
            rule_file["models"].setdefault(model, {"kinds": {}})["kinds"][kind] = {
                "w_surrogate": float(wf[0]), "w_judge": float(wf[1]), "clip": float(c), "surrogate_scale": SURR_SCALE[kind], "judge_scale": float(kjf),
                "speak": RULES[best], "speak_rule": best,
                "fitted_on": {"pairs": Q.height, "accounts": int(len(set(a))), "set": f"{name}: pre-cutoff {kind.replace('_', '-')} pairs, text similarity under 0.9"},
                "development_validation": {"pairs_2x": v2.get("pairs"), "spoken_share_2x": v2.get("spoken_share", [None, None])[1], "correct_when_spoken_2x": v2.get("spoken_acc", [None, None])[1]}}
            R["kinds"][kind][jn] = res
    json.dump(R, open(D + "jp_fit.json", "w"), indent=1, default=float)
    open(D + "jp_fit.md", "w").write("\n".join(md) + "\n")
    json.dump(rule_file, open(D + "judge_rule_fitted.json", "w"), indent=1)
    print("\nwrote " + D + "judge_rule_fitted.json")


if __name__ == "__main__":
    main()
