#!/usr/bin/env python3
"""Collect every evaluation of the expand track into one markdown file and one JSON.

    python scripts/expand/summarize.py --families ctl,exp,expsrc [--suffix _ft] --out data/expand/eval/results

Reads data/expand/runs/<name>/metrics.json, data/expand/eval/table_slices*.json (+ .npz), the benchmark shards under
data/bench3/<name>/ (through scripts/evidence/gap_table.py) and the launch scores under data/expand/launch/.
A family is a set of seeds <family>_s1, <family>_s2; the first family is the control every other one is compared with.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

E = Path("data/expand")


def pct(x, d=1):
    return "-" if x is None else f"{100 * x:.{d}f}%"


def ci(lo, hi, signed=False):
    f = "+.1f" if signed else ".1f"
    return f"({100 * lo:{f}} to {100 * hi:{f}})"


def boot_diff(z, names_a, names_b, sl, target, n=2000):
    """Paired difference in pair accuracy between two sets of models on one slice, accounts resampled."""
    ca = sum(z[f"{a}|{sl}|{target}|c"] for a in names_a) / len(names_a)
    cb = sum(z[f"{b}|{sl}|{target}|c"] for b in names_b) / len(names_b)
    nn = z[f"{names_a[0]}|{sl}|{target}|n"]
    d = ca - cb
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(nn), (n, len(nn)))
    b = d[idx].sum(1) / nn[idx].sum(1)
    return float(d.sum() / nn.sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="ctl,exp,expsrc")
    ap.add_argument("--seeds", default="s1,s2")
    ap.add_argument("--labels", default="")
    ap.add_argument("--slices", default="data/expand/eval/table_slices.json")
    ap.add_argument("--extra-bench", default="", help="name=glob,... more benchmark runs to list")
    ap.add_argument("--out", default="data/expand/eval/results")
    args = ap.parse_args()
    fams, seeds = args.families.split(","), args.seeds.split(",")
    lab = dict(zip(fams, args.labels.split("|"))) if args.labels else {f: f for f in fams}
    members = {f: [f"{f}_{s}" for s in seeds if (E / "runs" / f"{f}_{s}" / "best.pt").exists()] for f in fams}
    ctl = fams[0]
    md, out = [], {}

    # 1. train_one metrics (original audience test rows)
    md.append("### Ordinary same-account test pairs, each model's own run (existing audience test rows)\n")
    md.append("| Model | Rows | Likes pairs | Likes correct | Views pairs | Views correct | Within-account rho likes | Like NLL | Best step |\n|---|---|---|---|---|---|---|---|---|")
    for f in fams:
        for m in members[f]:
            p = E / "runs" / m / "metrics.json"
            if p.exists():
                j = json.loads(p.read_text())
                md.append(f"| {m} | {j['n']:,} | {j['pairs_likes']:,} | {pct(j['pairwise_likes'], 2)} | {j['pairs_views']:,} | {pct(j['pairwise_views'], 2)} | {j['within_rho_likes']:.4f} | {j['nll_likes']:.4f} | {j.get('best_step', j.get('picked_step'))} |")
                out.setdefault("train_metrics", {})[m] = {k: j[k] for k in ("n", "pairs_likes", "pairwise_likes", "pairs_views", "pairwise_views", "within_rho_likes", "nll_likes")}

    # 2. table slices
    if Path(args.slices).exists():
        S = json.loads(Path(args.slices).read_text())
        z = np.load(args.slices.replace(".json", ".npz"), allow_pickle=True)
        all_fams, all_members = fams, members
        members = {f: [m for m in members[f] if m in S["models"]] for f in fams}
        fams = [f for f in fams if members[f]]
        out["table_slices"] = {}
        for target in (("likes", "views") if members.get(ctl) else ()):
            md.append(f"\n### Same-account pairs by slice, {target} (all models on the same rows of the expanded table)\n")
            head = "| Slice | Rows | Accounts with pairs | Pairs | " + " | ".join(f"{lab[f]} (mean of seeds)" for f in fams) + " | " + " | ".join(f"{lab[f]} minus {lab[ctl]}, points" for f in fams[1:]) + " |"
            md.append(head + "\n|" + "---|" * (head.count("|") - 1))
            for sl in S["slices"]:
                first = S["models"][members[ctl][0]][sl]
                if target not in first or first[target]["pairs"] < 30:
                    continue
                cells = []
                for f in fams:
                    a = [S["models"][m][sl][target]["accuracy"] for m in members[f]]
                    cells.append(f"{pct(float(np.mean(a)), 2)} [{', '.join(pct(x, 1) for x in a)}]")
                diffs = []
                for f in fams[1:]:
                    k = min(len(members[f]), len(members[ctl]))
                    d, lo, hi = boot_diff(z, members[f][:k], members[ctl][:k], sl, target)
                    diffs.append(f"{100 * d:+.2f} {ci(lo, hi, True)}")
                    out["table_slices"][f"{sl}|{target}|{f}_minus_{ctl}"] = {"gap": d, "lo": lo, "hi": hi, "pairs": first[target]["pairs"], "pair_accounts": first[target]["pair_accounts"]}
                md.append(f"| {sl} | {first['rows']:,} | {first[target]['pair_accounts']:,} | {first[target]['pairs']:,} | " + " | ".join(cells) + " | " + " | ".join(diffs) + " |")
        if members.get(ctl):
            md.append("\n### Level of the forecast on table slices, likes (median over rows with 5+ visible history posts; ratio to the account's own median)\n")
            md.append("| Slice | Rows | Accounts | Actual / own median | " + " | ".join(f"{lab[f]}: predicted / own median" for f in fams) + " | " + " | ".join(f"{lab[f]}: actual above q90" for f in fams) + " |")
            md.append("|" + "---|" * (4 + 2 * len(fams)))
            for sl in S["slices"]:
                first = S["models"][members[ctl][0]][sl]
                if "likes" not in first or "level" not in first["likes"]:
                    continue
                L = first["likes"]["level"]
                pr = [" / ".join(f"{S['models'][m][sl]['likes']['level']['predicted_over_own_median']:.2f}x" for m in members[f]) for f in fams]
                ab = [" / ".join(pct(S['models'][m][sl]['likes']['level']['above_q90'], 0) for m in members[f]) for f in fams]
                md.append(f"| {sl} | {L['rows']:,} | {L['accounts']:,} | {L['actual_over_own_median']:.2f}x | " + " | ".join(pr) + " | " + " | ".join(ab) + " |")
        fams, members = all_fams, all_members

    # 3. benchmark
    have = {m: sorted(glob.glob(f"data/bench3/{m}/shard_*.parquet")) for f in fams for m in members[f]}
    have = {m: v for m, v in have.items() if len(v) == 4}
    if have:
        cmd = [sys.executable, "scripts/evidence/gap_table.py", "--out", args.out + "_bench.json"]
        for m in have:
            cmd += ["--run", f"{m}=data/bench3/{m}/shard_*.parquet"]
        for x in [x for x in args.extra_bench.split(",") if x]:
            cmd += ["--run", x]
        groups = {f: [m for m in members[f] if m in have] for f in fams}
        for f, ms in groups.items():
            if len(ms) > 1:
                cmd += ["--group", f"{f}={','.join(ms)}"]
        for f in fams[1:]:
            if len(groups[f]) > 1 and len(groups[ctl]) > 1:
                cmd += ["--versus", f"{f},{ctl}"]
            for a, b in zip(groups[f], groups[ctl]):
                cmd += ["--versus", f"{a},{b}"]
        subprocess.run(cmd, check=True, capture_output=True)
        B = json.loads(Path(args.out + "_bench.json").read_text())
        out["benchmark"] = B
        md.append("\n### The 3,530-pair benchmark (same account, same topic; likes), by size of the real gap\n")
        md.append("| Model | Real gap at least | Pairs | Correct (95% interval) | Confident (60%+) share (pairs) | Correct there (95% interval) |\n|---|---|---|---|---|---|")
        for n, r in B["models"].items():
            for g, x in r["by_gap"].items():
                cc = f"{pct(x['confident_accuracy'])} {ci(*x['confident_accuracy_ci']) if x['confident_accuracy_ci'] else ''}" if x["confident_accuracy"] is not None else "-"
                md.append(f"| {lab.get(n, n)} | {g} | {x['pairs']:,} | {pct(x['accuracy'])} {ci(*x['accuracy_ci'])} | {pct(x['confident_share'], 0)} ({x['confident_pairs']}) | {cc} |")
        md.append("\nPaired differences on the benchmark (points, accounts resampled):\n")
        for v, r in B["versus"].items():
            md.append(f"- {v.replace(',', ' minus ')}: {r['pairs']:,} pairs, all pairs {100 * r['accuracy_gap']:+.1f} {ci(*r['accuracy_gap_ci'], True)}, among confident calls {100 * r['confident_gap']:+.1f} {ci(*r['confident_gap_ci'], True)}")

    # 4. launch pairs and level
    sc = {m: E / "launch" / f"{m}.parquet" for f in fams for m in members[f]}
    sc = {m: p for m, p in sc.items() if p.exists()}
    if sc:
        cmd = [sys.executable, "scripts/expand/launch_check.py", "--out", args.out + "_launch.json"]
        for m, p in sc.items():
            cmd += ["--score", f"{m}={p}"]
            if m in have:
                cmd += ["--bench", f"{m}=data/bench3/{m}/shard_*.parquet"]
        for f in fams:
            ms = [m for m in members[f] if m in sc]
            if len(ms) > 1:
                cmd += ["--group", f"{f}={','.join(ms)}"]
                if all(m in have for m in ms):
                    cmd += ["--bench", f"{f}=" + ",".join(f"data/bench3/{m}/shard_*.parquet" for m in ms)]
        for f in fams[1:]:
            if sum(m in sc for m in members[f]) > 1 and sum(m in sc for m in members[ctl]) > 1:
                cmd += ["--versus", f"{f},{ctl}"]
            for a, b in zip([m for m in members[f] if m in sc], [m for m in members[ctl] if m in sc]):
                cmd += ["--versus", f"{a},{b}"]
        subprocess.run(cmd, check=True, capture_output=True)
        L = json.loads(Path(args.out + "_launch.json").read_text())
        out["launch"] = L
        md.append("\n### The 850 launch pairs (same account, both launch posts, test period)\n")
        cols = (("all", "All"), ("gap_2x", "Gap 2x+"), ("gap_3x", "Gap 3x+"), ("company_accounts", "Company"), ("founder_accounts", "Founder"), ("accounts_under_10k_followers", "Under 10k followers"),
                ("confident_0.6_plus", "Confident 60%+"), ("confident_0.6_plus_gap_2x", "Confident and 2x+"))
        md.append("| Model | " + " | ".join(c for _, c in cols) + " |\n|" + "---|" * (len(cols) + 1))
        for n, o in L["models"].items():
            cells = []
            for k, _ in cols:
                x = o[k]
                if x.get("too_thin"):
                    cells.append(f"too thin ({x['pairs']})")
                else:
                    extra = f", {pct(x['share_of_pairs'], 0)} of pairs" if "share_of_pairs" in x else f", {pct(x['share_of_2x_pairs'], 0)} of 2x pairs" if "share_of_2x_pairs" in x else ""
                    cells.append(f"{pct(x['accuracy'])} {ci(x['lo'], x['hi'])} n={x['pairs']}{extra}")
            md.append(f"| {lab.get(n, n)} | " + " | ".join(cells) + " |")
        md.append("\nPaired differences on the launch pairs (points, accounts resampled):\n")
        for v, r in L["versus"].items():
            md.append(f"- {v.replace(',', ' minus ')}: {r['pairs']} pairs, {r['accounts']} accounts, {100 * r['gap']:+.1f} {ci(r['lo'], r['hi'], True)}")
        md.append("\n### Level of the forecast for launch posts (draft path; likes over the account's trailing median)\n")
        md.append("| Model | Slice | Posts | Accounts | Actual | Predicted (95% interval) | Actual beats own median | Predicted to beat it | Rank correlation of predicted and actual lift | Median predicted / actual likes |\n|---|---|---|---|---|---|---|---|---|---|")
        for n, o in L["models"].items():
            for k, name in (("all", "all"), ("company", "company"), ("founder", "founder"), ("under_10k_followers", "under 10k followers")):
                x = o["level"][k]
                if x.get("too_thin"):
                    md.append(f"| {lab.get(n, n)} | {name} | {x['posts']} | | too thin | | | | | |")
                    continue
                md.append(f"| {lab.get(n, n)} | {name} | {x['posts']} | {x['accounts']} | {x['actual_median_ratio']:.2f}x | {x['predicted_median_ratio']:.2f}x ({x['predicted_median_ratio_ci'][0]:.2f} to {x['predicted_median_ratio_ci'][1]:.2f}) | "
                          f"{pct(x['actual_beat_own_median'], 0)} | {pct(x['predicted_beat_own_median'], 0)} | {x['spearman_pred_vs_actual_lift']:.2f} | {x['median_predicted_over_actual']:.2f} |")
    Path(args.out + ".md").write_text("\n".join(md) + "\n")
    json.dump(out, open(args.out + ".json", "w"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
