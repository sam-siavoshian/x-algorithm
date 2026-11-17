#!/usr/bin/env python3
"""Tables of the scrape track from data/scrape_track/eval/slices.json (+ .npz): same-account pairs by slice for views and
likes, train / validation / test side by side, differences against the control with intervals from resampling accounts;
and the level of the view forecast.

    python scripts/scrape/summarize.py --families ctl,hist,histsrc,allsrc --out data/scrape_track/eval/tables
"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--families", default="ctl,hist,histsrc,allsrc"); ap.add_argument("--slices", default="data/scrape_track/eval/slices.json")
ap.add_argument("--out", default="data/scrape_track/eval/tables")
args = ap.parse_args()
S = json.load(open(args.slices)); z = np.load(args.slices.replace(".json", ".npz"), allow_pickle=True)
fams = args.families.split(",")
LAB = {"h12src": "+ posts captured 12h+, sampling input", "ctl": "Control", "hist": "+ accounts 20+ (no input)", "histsrc": "+ accounts 20+, sampling input", "allsrc": "+ all posts, sampling input"}
mem = {f: [m for m in (f"{f}_s1", f"{f}_s2") if m in S["models"]] for f in fams}
fams = [f for f in fams if mem[f]]
rng = np.random.default_rng(0)


def acc_ci(ms, sl, t):
    n = z[f"{ms[0]}|{sl}|{t}|n"]; c = sum(z[f"{m}|{sl}|{t}|c"] for m in ms) / len(ms)
    idx = rng.integers(0, len(n), (2000, len(n))); b = c[idx].sum(1) / n[idx].sum(1)
    return float(c.sum() / n.sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def diff(a, b, sl, t):
    k = min(len(a), len(b)); a, b = a[:k], b[:k]
    n = z[f"{a[0]}|{sl}|{t}|n"]; d = (sum(z[f"{m}|{sl}|{t}|c"] for m in a) - sum(z[f"{m}|{sl}|{t}|c"] for m in b)) / k
    idx = rng.integers(0, len(n), (2000, len(n))); bs = d[idx].sum(1) / n[idx].sum(1)
    return float(d.sum() / n.sum()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


md, out = [], {}
for t in ("views", "likes"):
    md.append(f"\n#### Same-account pairs, {t} (every model on the same rows; mean of two seeds [seed 1, seed 2]; differences in points with a 95% interval from resampling accounts)\n")
    head = "| Slice | Rows | Accounts with pairs | Pairs | " + " | ".join(LAB[f] for f in fams) + " | " + " | ".join(f"{LAB[f]} minus Control" for f in fams[1:]) + " |"
    md.append(head); md.append("|" + "---|" * (head.count("|") - 1))
    for sl in S["slices"]:
        first = S["models"][mem[fams[0]][0]][sl]
        if t not in first or first[t]["pairs"] < 30:
            continue
        cells = []
        for f in fams:
            a = [S["models"][m][sl][t]["accuracy"] for m in mem[f]]
            cells.append(f"{100 * np.mean(a):.2f}% [{', '.join(f'{100 * x:.1f}' for x in a)}]")
        ds = []
        for f in fams[1:]:
            d, lo, hi = diff(mem[f], mem[fams[0]], sl, t)
            ds.append(f"{100 * d:+.2f} ({100 * lo:+.1f} to {100 * hi:+.1f})")
            out[f"{sl}|{t}|{f}_minus_ctl"] = {"gap": d, "lo": lo, "hi": hi, "pairs": first[t]["pairs"], "pair_accounts": first[t]["pair_accounts"]}
        md.append(f"| {sl} | {first['rows']:,} | {first[t]['pair_accounts']:,} | {first[t]['pairs']:,} | " + " | ".join(cells) + " | " + " | ".join(ds) + " |")
    if "histsrc" in fams and "hist" in fams:
        md.append(f"\nThe sampling input against the same rows without it ({t}):\n")
        for sl in S["slices"]:
            first = S["models"][mem[fams[0]][0]][sl]
            if t in first and first[t]["pairs"] >= 30:
                d, lo, hi = diff(mem["histsrc"], mem["hist"], sl, t)
                md.append(f"- {sl}: {100 * d:+.2f} ({100 * lo:+.1f} to {100 * hi:+.1f})")
                out[f"{sl}|{t}|histsrc_minus_hist"] = {"gap": d, "lo": lo, "hi": hi}
for t in ("views", "likes"):
    md.append(f"\n#### Level of the {t[:-1]} forecast (median predicted over actual; and, for rows with 5+ visible history posts, predicted and actual over the account's own median)\n")
    md.append("| Slice | Labelled rows | " + " | ".join(f"{LAB[f]}: predicted / actual" for f in fams) + " | Actual / own median | " + " | ".join(f"{LAB[f]}: predicted / own median" for f in fams) + " | " + " | ".join(f"{LAB[f]}: actual above q90" for f in fams) + " |")
    md.append("|" + "---|" * (3 + 3 * len(fams)))
    for sl in S["slices"]:
        first = S["models"][mem[fams[0]][0]][sl]
        if t not in first:
            continue
        pa = [" / ".join(f"{S['models'][m][sl][t]['median_predicted_over_actual']:.2f}x" for m in mem[f]) for f in fams]
        if "level" in first[t]:
            lv = [" / ".join(f"{S['models'][m][sl][t]['level']['predicted_over_own_median']:.2f}x" for m in mem[f]) for f in fams]
            ab = [" / ".join(f"{100 * S['models'][m][sl][t]['level']['above_q90']:.0f}%" for m in mem[f]) for f in fams]
            act = f"{first[t]['level']['actual_over_own_median']:.2f}x"
        else:
            lv, ab, act = ["-"] * len(fams), ["-"] * len(fams), "-"
        md.append(f"| {sl} | {first[t]['labelled_rows']:,} | " + " | ".join(pa) + f" | {act} | " + " | ".join(lv) + " | " + " | ".join(ab) + " |")
        out[f"{sl}|{t}|level"] = {f: [S["models"][m][sl][t]["median_predicted_over_actual"] for m in mem[f]] for f in fams}
Path(args.out + ".md").write_text("\n".join(md) + "\n"); json.dump(out, open(args.out + ".json", "w"), indent=1)
print("\n".join(md))
