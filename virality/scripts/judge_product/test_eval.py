#!/usr/bin/env python3
"""Score the fitted rule (data/llm/judge_rule_fitted.json, development pairs only) on the test sets, once.

    python scripts/judge_product/test_eval.py [--debug]     # --debug runs the same code on the development tables instead

Benchmark (same-topic rule) and launch pairs after the cutoff (launch rule), per judge model: every table for all pairs,
without the near-identical pairs, after the judge's knowledge cutoff, and both. Fixed scales: nothing is fitted or
rescaled on a test set. Writes data/llm/jp_test.json, jp_test.md and the shipped rule file with its validated numbers."""
import json, sys, os, argparse
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(__file__))
from jp import *
from fit_rule import SURR_SCALE, speak_mask

MODEL = {"luna": "gpt-6-luna", "sol": "gpt-6.1-sol", "dsflash": "deepseek-ai/DeepSeek-V4.1-Flash", "qwen": "Qwen/Qwen3.5-397B-A17B", "oss": "openai/gpt-oss-120b"}
CUTDATE = {"gpt-6-luna": "2026-05-18", "gpt-6.1-sol": "2026-04-30", "Qwen/Qwen3.5-397B-A17B": "2026-02-16"}
CUTWORD = {"Qwen/Qwen3.5-397B-A17B": "the day the model's weights were published"}
md = []


def say(*x):
    s = " ".join(str(v) for v in x); print(s); md.append(s)


def block(z, zs, d, a, spoken, spoken_s, gaps=(1.25, 2, 3, 5)):
    c, cs = correct(z, d), correct(zs, d); out = {}
    for g in gaps:
        m = np.abs(d) >= np.log(g)
        if m.sum() < 30:
            continue
        r = {"pairs": int(m.sum()), "accounts": int(len(set(a[m])))}
        for nm, cc, sp, zz in (("blend", c, spoken, z), ("surrogate", cs, spoken_s, zs)):
            spm = m & sp
            idx = np.flatnonzero(m); top = np.zeros(len(d), bool); top[idx[np.argsort(-np.abs(zz[idx]), kind="stable")[: int(round(.4 * len(idx)))]]] = True
            r[nm] = {"acc": rate(cc, m, a), "spoken_share": rate(sp.astype(float), m, a), "spoken_pairs": int(spm.sum()), "spoken_acc": rate(cc, spm, a), "top40": rate(cc, top, a)}
        lo, hi = ci(a, lambda i: (c[i] - cs[i])[m[i]].mean() if m[i].sum() else np.nan)
        r["acc_diff"] = [float((c - cs)[m].mean()), lo, hi]
        f = lambda i: ((c[i] * spoken[i])[m[i]].mean() - (cs[i] * spoken_s[i])[m[i]].mean()) * 100 if m[i].sum() else np.nan     # noqa: E731
        r["correct_spoken_per_100_diff"] = [float(f(np.arange(len(d))))] + ci(a, f)
        out[f"{g}x"] = r
    return out


def lines(lab, B):
    for g, r in B.items():
        b, s = r["blend"], r["surrogate"]
        say(f"| {lab} | {g} | {r['pairs']:,} | {fmt_rate(s['acc'])} | {100 * s['spoken_share'][1]:.0f}% ({s['spoken_pairs']}) | {fmt_rate(s['spoken_acc'])} | {fmt_rate(b['acc'])} | "
            f"{100 * b['spoken_share'][1]:.0f}% ({100 * b['spoken_share'][2]:.0f} to {100 * b['spoken_share'][3]:.0f}; {b['spoken_pairs']}) | {fmt_rate(b['spoken_acc'])} | "
            f"{100 * r['acc_diff'][0]:+.1f} ({100 * r['acc_diff'][1]:+.1f} to {100 * r['acc_diff'][2]:+.1f}) |")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true"); a_ = ap.parse_args()
    rule = json.load(open("data/llm/judge_rule_fitted.json"))
    R = {}
    sets = {"same_topic": "dev", "launch": "launchpre"} if a_.debug else {"same_topic": "bench", "launch": "launch"}
    for kind, name in sets.items():
        P = table(name)
        d, a, sg, sim = P["d"].to_numpy(), P["author_id"].to_numpy(), P["sg"].to_numpy(), P["sim"].to_numpy()
        t0, t1 = np.minimum(P["t_a"].to_numpy(), P["t_b"].to_numpy()), np.maximum(P["t_a"].to_numpy(), P["t_b"].to_numpy())
        fol = P["followers"].to_numpy().astype(float); nh = P["n_hist"].to_numpy()
        say(f"\n## {kind}: {name}, {P.height} pairs 1.25x apart, {len(set(a))} accounts\n")
        near = sim >= .9
        later_b = P["t_b"].to_numpy() > P["t_a"].to_numpy(); later_won = (d > 0) != later_b
        facts = {"pairs": P.height, "near_identical": int(near.sum()), "near_identical_later_won": float(later_won[near].mean()) if near.sum() else None,
                 "later_won_all": float(later_won.mean()), "both_after_2026-04-23": int((t0 >= CUT["test"]).sum()), "both_after_2026-04-30_sol": int((t0 >= CUT["gpt-6.1-sol"]).sum()),
                 "both_after_2026-05-18_luna": int((t0 >= CUT["gpt-6-luna"]).sum()), "earliest_post": str(np.datetime64(int(t0.min()), "ms"))[:10], "latest_post": str(np.datetime64(int(t1.max()), "ms"))[:10],
                 "median_days_apart": float(np.median(np.abs(P["t_a"].to_numpy() - P["t_b"].to_numpy())) / 864e5), "median_history": float(np.median(nh))}
        say("Set facts:", json.dumps(facts))
        R[kind] = {"set": name, "facts": facts, "judges": {}}
        for jn in ("luna", "sol", "dsflash", "qwen", "oss"):
            model = MODEL[jn]
            spec = (rule["models"].get(model) or {"kinds": {}})["kinds"].get(kind)
            if jn not in P.columns or spec is None or P[jn].is_not_null().sum() < 200:
                continue
            ok = P[jn].is_not_null().to_numpy()
            jr = np.nan_to_num(P[jn].to_numpy().astype(float)); j = np.clip(jr, -spec["clip"], spec["clip"])
            z = spec["w_surrogate"] * sg + spec["w_judge"] * j
            zs = SURR_SCALE[kind] * sg
            spoken = speak_mask(spec["speak"], z, sg, j, spec["judge_scale"]); spoken_s = np.maximum(sig(zs), 1 - sig(zs)) >= .6
            cut = CUT.get(model)
            say(f"\n### {kind}, surrogate + {model} ({int(ok.sum())} of {P.height} pairs scored by the judge; rule: {spec['speak_rule']}; weights {spec['w_surrogate']:.3f}, {spec['w_judge']:.3f}, clip {spec['clip']:g})\n")
            sl = {"all pairs": ok, "wording differs (near-identical removed)": ok & ~near}
            if cut:
                sl[f"after the judge's knowledge cutoff ({CUTDATE[model]})"] = ok & (t0 >= cut)
                sl["wording differs and after the cutoff"] = ok & ~near & (t0 >= cut)
                sl[f"before the cutoff, wording differs"] = ok & ~near & (t0 < cut)
            sl.update({"near-identical only": ok & near, "under 50k followers, wording differs": ok & ~near & (fol < 50_000), "50k followers and over, wording differs": ok & ~near & (fol >= 50_000),
                       "under 30 earlier posts available, wording differs": ok & ~near & (nh < 30), "30 or more earlier posts, wording differs": ok & ~near & (nh >= 30)})
            # near-zero posts: a pair "2x apart" with a post of 0 to 2 likes is easy and noisy; the same tables with both posts at 3+ and at 10+ likes
            lo = np.minimum(P["likes"].to_numpy(), P["likes_b"].to_numpy())
            post = (t0 >= cut) if cut else np.ones(P.height, bool)
            sl.update({"a post with 0 to 2 likes, wording differs": ok & ~near & (lo < 3), "both posts 3+ likes, wording differs": ok & ~near & (lo >= 3), "both posts 10+ likes, wording differs": ok & ~near & (lo >= 10),
                       "both posts 3+ likes, wording differs, after the cutoff": ok & ~near & (lo >= 3) & post, "both posts 10+ likes, wording differs, after the cutoff": ok & ~near & (lo >= 10) & post,
                       "both posts 3+ likes, all pairs": ok & (lo >= 3)})
            if kind == "launch":
                gap_days = np.abs(P["t_a"].to_numpy() - P["t_b"].to_numpy()) / 864e5
                sl.update({"posts within 30 days of each other": ok & (gap_days <= 30), "posts more than 30 days apart": ok & (gap_days > 30)})
            res = {"scored": int(ok.sum()), "slices": {}}
            say("| Slice | Real gap at least | Pairs | Surrogate: correct | Surrogate: spoken share (pairs) | Surrogate: correct when spoken | Blend: correct | Blend: spoken share (interval; pairs) | Blend: correct when spoken | Blend minus surrogate, all calls |")
            say("|---|---|---|---|---|---|---|---|---|---|")
            for lab, m in sl.items():
                if m.sum() < 60:
                    say(f"| {lab} | | {int(m.sum())} | too few | | | | | | |"); continue
                B = block(z[m], zs[m], d[m], a[m], spoken[m], spoken_s[m]); res["slices"][lab] = B
                lines(lab, B)
            # judge alone, agreement, calibration on the clean slice
            m = ok & ~near
            cj, cs_, cz = correct(j, d), correct(sg, d), correct(z, d)
            ag = (sg * j) > 0; dis = (sg * j) < 0
            res["judge_alone"] = {lab: {"1.25x": rate(cj, mm, a), "2x": rate(cj, mm & (np.abs(d) >= L2), a)} for lab, mm in sl.items() if mm.sum() >= 60}
            say("\nJudge alone: " + "; ".join(f"{lab}: {fmt_rate(v['1.25x'])}, at 2x {fmt_rate(v['2x'])}" for lab, v in list(res["judge_alone"].items())[:5]))
            res["agreement"] = {"pairs": int(m.sum()), "agree_share": float(ag[m].mean()), "right_when_agree": rate(cs_, m & ag, a), "blend_right_when_disagree": rate(cz, m & dis, a),
                                "surrogate_right_when_disagree": rate(cs_, m & dis, a), "judge_right_when_disagree": rate(cj, m & dis, a),
                                "spoken_share_when_agree": float(spoken[m & ag].mean()), "spoken_share_when_disagree": float(spoken[m & dis].mean()) if (m & dis).sum() else None,
                                "right_when_spoken_and_agree": rate(cz, m & ag & spoken, a), "right_when_spoken_and_disagree": rate(cz, m & dis & spoken, a),
                                "at_2x": {"right_when_agree": rate(cs_, m & ag & (np.abs(d) >= L2), a), "agree_share": float(ag[m & (np.abs(d) >= L2)].mean())}}
            q = res["agreement"]
            say(f"Agreement (wording differs, {q['pairs']} pairs): the two prefer the same version on {100 * q['agree_share']:.0f}%; right when they agree {fmt_rate(q['right_when_agree'])} (at 2x {fmt_rate(q['at_2x']['right_when_agree'])}); "
                f"when they disagree the surrogate is right {fmt_rate(q['surrogate_right_when_disagree'])}, the judge {fmt_rate(q['judge_right_when_disagree'])}, the blend {fmt_rate(q['blend_right_when_disagree'])}. "
                f"The rule speaks on {100 * q['spoken_share_when_agree']:.0f}% of agreeing pairs (right {fmt_rate(q['right_when_spoken_and_agree'])}) and {100 * (q['spoken_share_when_disagree'] or 0):.0f}% of disagreeing pairs (right {fmt_rate(q['right_when_spoken_and_disagree'])}).")
            pz = np.maximum(sig(z), 1 - sig(z))
            res["calibration"] = [{"stated": f"{lo:.2f} to {hi:.2f}", "pairs": int((m & (pz >= lo) & (pz < hi)).sum()), "correct": float(cz[m & (pz >= lo) & (pz < hi)].mean()) if (m & (pz >= lo) & (pz < hi)).sum() else None}
                                  for lo, hi in ((.5, .55), (.55, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01))]
            say("Stated probability against outcome (wording differs): " + "; ".join(f"{x['stated']}: {100 * x['correct']:.0f}% of {x['pairs']}" for x in res["calibration"] if x["pairs"]))
            R[kind]["judges"][jn] = res
            # what the product may state: the slice that matches its use (wording differs, both posts after the judge's knowledge cutoff)
            key = "wording differs and after the cutoff" if cut and "wording differs and after the cutoff" in res["slices"] else "wording differs (near-identical removed)"
            B = res["slices"][key]; b1, b2 = B["1.25x"], B.get("2x")
            what = "launch posts by the same account" if kind == "launch" else "posts by the same account about the same topic"
            st = (f"With {model} as second opinion, checked on {b1['pairs']:,} real pairs of {what} that differ in wording"
                  + (f" and were posted after {CUTWORD.get(model, "the language model's knowledge cutoff")} ({CUTDATE[model]})" if cut and key.endswith("cutoff") else "")
                  + f": the two models together picked the better post {100 * b1['blend']['acc'][1]:.0f}% of the time ({100 * b1['blend']['acc'][2]:.0f} to {100 * b1['blend']['acc'][3]:.0f}), "
                    f"the forecasting model alone {100 * b1['surrogate']['acc'][1]:.0f}%.")
            if b2:
                st += (f" Where the real gap was 2x or more ({b2['pairs']:,} pairs) a verdict was given on {100 * b2['blend']['spoken_share'][1]:.0f}% of pairs and was right "
                       f"{100 * b2['blend']['spoken_acc'][1]:.0f}% of the time ({100 * b2['blend']['spoken_acc'][2]:.0f} to {100 * b2['blend']['spoken_acc'][3]:.0f}); "
                       f"the forecasting model alone: {100 * b2['surrogate']['spoken_share'][1]:.0f}% of pairs, {100 * b2['surrogate']['spoken_acc'][1]:.0f}%.")
            if kind == "launch":
                st += " These are two different launch posts weeks or months apart, not two drafts of one launch; no set of same-day launch drafts with outcomes exists."
            spec["validated"] = {"statement": st, "slice": key, "set": name, "pairs": b1["pairs"], "accounts": b1["accounts"], "accuracy": b1["blend"]["acc"][1], "accuracy_lo": b1["blend"]["acc"][2],
                                 "accuracy_hi": b1["blend"]["acc"][3], "forecast_model_alone": b1["surrogate"]["acc"][1],
                                 "pairs_2x": b2["pairs"] if b2 else None, "spoken_share_2x": b2["blend"]["spoken_share"][1] if b2 else None,
                                 "correct_when_spoken_2x": b2["blend"]["spoken_acc"][1] if b2 else None, "correct_when_spoken_2x_lo": b2["blend"]["spoken_acc"][2] if b2 else None,
                                 "correct_when_spoken_2x_hi": b2["blend"]["spoken_acc"][3] if b2 else None,
                                 "forecast_model_alone_spoken_share_2x": b2["surrogate"]["spoken_share"][1] if b2 else None, "forecast_model_alone_correct_when_spoken_2x": b2["surrogate"]["spoken_acc"][1] if b2 else None}
            say("Product statement: " + st)
    tag = "_debug" if a_.debug else ""
    json.dump(R, open(f"data/llm/jp_test{tag}.json", "w"), indent=1, default=float)
    open(f"data/llm/jp_test{tag}.md", "w").write("\n".join(md) + "\n")
    if not a_.debug:
        # only what was scored on the test sets ships: a model or kind without validated numbers is left out of the product file
        rule["models"] = {m: v for m, v in rule["models"].items() if all("validated" in v["kinds"].get(k, {}) for k in ("same_topic", "launch"))}
        json.dump(rule, open("xviral/data/judge_rule.json", "w"), indent=1)
        print("wrote xviral/data/judge_rule.json")


if __name__ == "__main__":
    main()
