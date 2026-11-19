#!/usr/bin/env python3
"""The remaining tables of the judge_product report: history length and prompt variants on development samples, the stronger
model on test pairs it had and had not seen before this track, the open models against luna and sol on development
validation pairs, latency per call, and dollars per endpoint.

    python scripts/judge_product/extras.py        # writes data/llm/jp_extras.json and jp_extras.md"""
import glob, json, os, sys
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(__file__))
from jp import *
from fit_rule import fit, fit1

md = []


def say(*x):
    s = " ".join(str(v) for v in x); print(s); md.append(s)


def paired(P, x, y, a, d, m, n=1000):
    cx, cy = correct(np.nan_to_num(P[x].to_numpy().astype(float)), d), correct(np.nan_to_num(P[y].to_numpy().astype(float)), d)
    lo, hi = ci(a, lambda i: (cx[i] - cy[i])[m[i]].mean() if m[i].sum() else np.nan, n=n)
    return [float((cx - cy)[m].mean()), lo, hi]


R = {}
# 1. history length and the evidence prompt, stronger model, same pairs
say("## History length and prompt variants (gpt-6.1-sol, development pairs, same pairs in every row)\n")
say("| Set | Variant | Pairs | Correct | Pairs 2x apart | Correct | Minus 30 posts, all pairs |"); say("|---|---|---|---|---|---|---|")
R["variants"] = {}
for name, cols, lab in (("dev", ["sol_n0", "sol_n10", "sol", "sol_n60", "sol_n100"], "same-topic, 293 pairs of the fit half"), ("launchpre", ["sol", "sol_n60", "sol_n100", "sol_n30_ev"], "launch before the cutoff, 300 pairs of the fit cell")):
    P = table(name); m = np.ones(P.height, bool)
    for c in cols:
        m &= P[c].is_not_null().to_numpy()
    d, a = P["d"].to_numpy(), P["author_id"].to_numpy(); R["variants"][name] = {"pairs": int(m.sum())}
    nh = P["n_hist"].to_numpy()
    for c in cols:
        s = np.nan_to_num(P[c].to_numpy().astype(float)); r1, r2 = rate(correct(s, d), m, a), rate(correct(s, d), m & (np.abs(d) >= L2), a)
        diff = paired(P, c, "sol", a, d, m) if c != "sol" else None
        nm = {"sol": "30 most recent posts (the configuration scored on the test sets)", "sol_n0": "no history", "sol_n10": "10 posts", "sol_n60": "60 posts", "sol_n100": "100 posts", "sol_n30_ev": "30 posts + the measured launch evidence in the prompt"}[c]
        R["variants"][name][c] = {"all": r1, "2x": r2, "minus_n30": diff}
        say(f"| {lab} | {nm} | {r1[0]} | {fmt_rate(r1)} | {r2[0]} | {fmt_rate(r2)} | " + ("" if diff is None else f"{100 * diff[0]:+.1f} ({100 * diff[1]:+.1f} to {100 * diff[2]:+.1f})") + " |")
    big = m & (nh >= 100)
    if big.sum() >= 40:
        say(f"| {lab}, accounts with 100 or more earlier posts | " + "; ".join(f"{c}: {100 * correct(np.nan_to_num(P[c].to_numpy().astype(float)), d)[big].mean():.1f}%" for c in cols) + f" | {int(big.sum())} | | | | |")
        R["variants"][name]["accounts_with_100_posts"] = {"pairs": int(big.sum()), **{c: float(correct(np.nan_to_num(P[c].to_numpy().astype(float)), d)[big].mean()) for c in cols}}
    say(f"| {lab} | surrogate | {int(m.sum())} | {fmt_rate(rate(correct(P['sg'].to_numpy(), d), m, a))} | | {fmt_rate(rate(correct(P['sg'].to_numpy(), d), m & (np.abs(d) >= L2), a))} | |")

# 2. the stronger model on the pairs the earlier track had scored, and on the ones scored here
say("\n## gpt-6.1-sol on test pairs scored by the earlier track and on those scored here (judge alone, no rule)\n")
say("| Set | Pairs | Count | Surrogate | Judge alone | Judge minus surrogate |"); say("|---|---|---|---|---|---|")
R["old_new"] = {}
for name in ("bench", "launch"):
    P = table(name); d, a = P["d"].to_numpy(), P["author_id"].to_numpy()
    old = load_judge(f"data/tracks/llm_judge/out/{name}__n30__gpt-6.1-sol__json-low.jsonl").select("post_id", "post_id_b", pl.lit(True).alias("old"))
    o = P.join(old, on=["post_id", "post_id_b"], how="left")["old"].fill_null(False).to_numpy()
    ok = P["sol"].is_not_null().to_numpy()
    for lab, m in (("scored by the earlier track (random sample)", ok & o), ("scored in this track (the rest)", ok & ~o), ("all", ok)):
        r_s, r_j = rate(correct(P["sg"].to_numpy(), d), m, a), rate(correct(np.nan_to_num(P["sol"].to_numpy().astype(float)), d), m, a)
        df = paired(P, "sol", "sg", a, d, m)
        R["old_new"][f"{name}|{lab}"] = {"surrogate": r_s, "judge": r_j, "diff": df}
        say(f"| {name} | {lab} | {int(m.sum())} | {fmt_rate(r_s)} | {fmt_rate(r_j)} | {100 * df[0]:+.1f} ({100 * df[1]:+.1f} to {100 * df[2]:+.1f}) |")

# 3. open models on the development validation cell (later period, held-out accounts), next to luna and sol on the same pairs
say("\n## Models compared on development validation pairs (wording differs; later period and held-out accounts)\n")
say("| Kind | Model | Pairs | Judge alone | Judge alone, 2x apart | sol on the same pairs | luna on the same pairs | Surrogate | Blend with surrogate (fitted on half the accounts, read on the other half) | Same side as sol |"); say("|---|---|---|---|---|---|---|---|---|---|")
R["models"] = {}
for kind, name in (("same-topic", "dev"), ("launch", "launchpre")):
    P = table(name).filter((pl.col("sim") < .9) & (pl.col("cell") == "val")); d, a, sg = P["d"].to_numpy(), P["author_id"].to_numpy(), P["sg"].to_numpy()
    y = (d > 0).astype(float)
    for c in ("oss", "dsflash", "qwen", "luna", "sol"):
        if c not in P.columns:
            continue
        m = P[c].is_not_null().to_numpy() & P["sol"].is_not_null().to_numpy()
        if m.sum() < 60:
            continue
        s = np.nan_to_num(P[c].to_numpy().astype(float))
        # blend: two-fold over accounts inside the cell
        ua = np.unique(a[m]); rng = np.random.default_rng(5); half = set(rng.choice(ua, len(ua) // 2, replace=False).tolist()); inh = np.array([x in half for x in a])
        zc = np.zeros(P.height)
        for tr, te in ((m & inh, m & ~inh), (m & ~inh, m & inh)):
            best = None
            for clip in (1.0, 2.0, 4.0, 14.0):
                w = fit(sg[tr], np.clip(s[tr], -clip, clip), y[tr]); z_ = sg[tr] * w[0] + np.clip(s[tr], -clip, clip) * w[1]
                l = float(-(y[tr] * np.log(sig(z_) + 1e-9) + (1 - y[tr]) * np.log(1 - sig(z_) + 1e-9)).mean())
                if best is None or l < best[0]:
                    best = (l, clip, w)
            zc[te] = sg[te] * best[2][0] + np.clip(s[te], -best[1], best[1]) * best[2][1]
        row = {"pairs": int(m.sum()), "judge": rate(correct(s, d), m, a), "judge_2x": rate(correct(s, d), m & (np.abs(d) >= L2), a), "sol": rate(correct(np.nan_to_num(P["sol"].to_numpy().astype(float)), d), m, a),
               "luna": rate(correct(P["luna"].to_numpy(), d), m, a), "surrogate": rate(correct(sg, d), m, a), "blend": rate(correct(zc, d), m, a),
               "same_side_as_sol": float(((s * np.nan_to_num(P["sol"].to_numpy().astype(float))) > 0)[m].mean()), "minus_sol": paired(P, c, "sol", a, d, m), "minus_luna": paired(P, c, "luna", a, d, m)}
        R["models"][f"{kind}|{c}"] = row
        say(f"| {kind} | {c} | {row['pairs']} | {fmt_rate(row['judge'])} | {fmt_rate(row['judge_2x'])} | {fmt_rate(row['sol'])} | {fmt_rate(row['luna'])} | {fmt_rate(row['surrogate'])} | {fmt_rate(row['blend'])} | {100 * row['same_side_as_sol']:.0f}% |")
        say(f"|  | {c} minus sol {100 * row['minus_sol'][0]:+.1f} ({100 * row['minus_sol'][1]:+.1f} to {100 * row['minus_sol'][2]:+.1f}); minus luna {100 * row['minus_luna'][0]:+.1f} ({100 * row['minus_luna'][1]:+.1f} to {100 * row['minus_luna'][2]:+.1f}) | | | | | | | | |")

# 4. latency, tokens and price per comparison (both orders), from the request logs
say("\n## Per comparison of two versions (both orders, 30 posts of history)\n")
say("| Model | Endpoint | Requests logged | Median seconds per call | 90th percentile | Input tokens per comparison | Output tokens per comparison | US dollars per 1,000 comparisons | Failed calls |"); say("|---|---|---|---|---|---|---|---|---|")
from xviral import judge as J
R["per_call"] = {}
for model, files in (("gpt-6-luna", "launchpre__n30__gpt-6-luna__logprob"), ("gpt-6.1-sol", None), ("openai/gpt-oss-120b", "*__n30__openai--gpt-oss-120b__*"), ("deepseek-ai/DeepSeek-V4.1-Flash", "*__n30__deepseek-ai--DeepSeek-V4.1-Flash__*"), ("Qwen/Qwen3.5-397B-A17B", "*__n30__Qwen--Qwen3.5-397B-A17B__*")):
    if files is None:
        continue
    rows = [r for f in glob.glob(f"data/llm/out/{files}.jsonl") for r in map(json.loads, open(f))]
    ok = [r for r in rows if r.get("p") is not None and r.get("s") is not None]
    if not ok:
        continue
    pi, pc, po = J.MODELS[model]["price"]
    tin, tc, tout = np.mean([r["in"] for r in ok]), np.mean([r.get("cached") or 0 for r in ok]), np.mean([r["out"] for r in ok])
    usd = 2 * ((tin - tc) * pi + tc * pc + tout * po) / 1e6 * 1000
    row = {"requests": len(rows), "median_s": float(np.median([r["s"] for r in ok])), "p90_s": float(np.percentile([r["s"] for r in ok], 90)), "tokens_in": 2 * float(tin), "tokens_out": 2 * float(tout), "usd_per_1000": float(usd),
           "failed": int(sum(r.get("p") is None for r in rows)), "errors": {}}
    for r in rows:
        if r.get("p") is None:
            row["errors"][r.get("err")] = row["errors"].get(r.get("err"), 0) + 1
    R["per_call"][model] = row
    say(f"| {model} | {J.MODELS[model]['endpoint']} | {row['requests']} | {row['median_s']:.1f} | {row['p90_s']:.1f} | {row['tokens_in']:,.0f} | {row['tokens_out']:,.0f} | {row['usd_per_1000']:.2f} | {row['failed']} {row['errors']} |")

# 4b. the benchmark's baseline table re-issued: the README's method (confidence scale cross-fitted over accounts inside the set), per slice
from lib import crossfit_conf
say("\n## Baseline benchmark table re-issued (surrogate only; confidence scale cross-fitted inside each slice, as in the README)\n")
say("| Slice | Model | Real gap at least | Pairs | Correct | Confident (60%+) share (pairs) | Correct there |"); say("|---|---|---|---|---|---|---|")
R["baseline_reissued"] = {}
P = table("bench"); d, a = P["d"].to_numpy(), P["author_id"].to_numpy(); sim = P["sim"].to_numpy(); t0 = np.minimum(P["t_a"].to_numpy(), P["t_b"].to_numpy())
for lab, m in (("all pairs (the table in the README)", np.ones(P.height, bool)), ("without the near-identical pairs", sim < .9), ("near-identical pairs only", sim >= .9),
               ("wording differs, both posts on or after 2026-04-23 (test cutoff)", (sim < .9) & (t0 >= CUT["test"])),
               ("wording differs, after 2026-04-30 (gpt-6.1-sol's knowledge cutoff)", (sim < .9) & (t0 >= CUT["gpt-6.1-sol"])), ("wording differs, after 2026-05-18 (gpt-6-luna's knowledge cutoff)", (sim < .9) & (t0 >= CUT["gpt-6-luna"])),
               ("wording differs, both posts before 2026-04-23", (sim < .9) & (np.maximum(P["t_a"].to_numpy(), P["t_b"].to_numpy()) < CUT["test"]))):
    for mn, col in (("baseline three", "sg"), ("ten untouched models", "s10")):
        s = P[col].to_numpy()[m]; dd, aa = d[m], a[m]
        conf, k = crossfit_conf(s, dd, aa); c = correct(s, dd)
        for g in (1.25, 2, 3, 5):
            mm = np.abs(dd) >= np.log(g); cm = mm & (conf >= .6)
            if mm.sum() < 30:
                continue
            r1, r2 = rate(c, mm, aa, 500), rate(c, cm, aa, 500)
            R["baseline_reissued"][f"{lab}|{mn}|{g}x"] = {"pairs": int(mm.sum()), "acc": r1, "conf_share": float(cm.sum() / mm.sum()), "conf_pairs": int(cm.sum()), "conf_acc": r2, "scale": k}
            say(f"| {lab} | {mn} | {g}x | {int(mm.sum()):,} | {fmt_rate(r1)} | {100 * cm.sum() / mm.sum():.0f}% ({int(cm.sum())}) | {fmt_rate(r2)} |")

# 5. dollars per endpoint
say("\n## Tokens and dollars\n"); say("| Run | Endpoint | Requests | Input tokens | Cached | Output tokens | USD |"); say("|---|---|---|---|---|---|---|")
tot, by = {}, {}
for r in map(json.loads, open("data/llm/spend.jsonl")):
    k = (os.path.basename(r["path"]).replace(".jsonl", "") + (" (batch)" if r.get("batch") else ""), r.get("endpoint") or "openai")
    t = tot.setdefault(k, {"n": 0, "in": 0, "cached": 0, "out": 0, "usd": 0.0})
    for f in ("n", "in", "cached", "out", "usd"):
        t[f] += r.get(f) or 0
for k in [k for k in tot if k[1] == "publik"]:
    del tot[k]
for f in sorted(glob.glob("data/llm/out/*--*.jsonl")):          # Publik runs: every request's tokens are in its log line, failed and stopped runs included
    model = os.path.basename(f).split("__")[2].replace("--", "/"); pi, pc, po = J.MODELS[model]["price"]
    t = tot.setdefault((os.path.basename(f).replace(".jsonl", ""), "publik"), {"n": 0, "in": 0, "cached": 0, "out": 0, "usd": 0.0})
    for r in map(json.loads, open(f)):
        i, c, o = r.get("in") or 0, r.get("cached") or 0, r.get("out") or 0
        t["n"] += 1; t["in"] += i; t["cached"] += c; t["out"] += o; t["usd"] += ((i - c) * pi + c * pc + o * po) / 1e6
for (k, ep), t in sorted(tot.items()):
    say(f"| {k} | {ep} | {t['n']:,} | {t['in']:,} | {t['cached']:,} | {t['out']:,} | {t['usd']:.3f} |")
    b = by.setdefault(ep, {"n": 0, "usd": 0.0}); b["n"] += t["n"]; b["usd"] += t["usd"]
for ep, b in by.items():
    say(f"| **Total, {ep}** | | {b['n']:,} | | | | **{b['usd']:.2f}** |")
R["spend"] = {"runs": {f"{k}|{ep}": t for (k, ep), t in tot.items()}, "by_endpoint": by}
json.dump(R, open("data/llm/jp_extras.json", "w"), indent=1, default=float)
open("data/llm/jp_extras.md", "w").write("\n".join(md) + "\n")
