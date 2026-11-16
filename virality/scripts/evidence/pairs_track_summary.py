#!/usr/bin/env python3
"""One table for the controlled-pairs experiments: benchmark by gap, held-out controlled pairs, ordinary validation and test.

    python scripts/evidence/pairs_track_summary.py --base base_ts46_s1 --runs ft_w0,ft_w1,... --out results.json
Reads data/bench3/<name>/shard_*.parquet, data/runs/<name>/{metrics.json,start.json} and the stored ensemble members in data/bench2.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="base_ts46_s1")
ap.add_argument("--runs", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
runs = [r for r in args.runs.split(",") if Path(f"data/bench3/{r}").exists()]
cmd = [sys.executable, "scripts/evidence/gap_table.py", "--out", args.out + ".gap.json", "--run", f"{args.base}=data/bench3/{args.base}/shard_*.parquet",
       "--run", "ts46_s2=data/bench2/ts46_s2/shard_*.parquet", "--run", "v2full=data/bench2/compare_v2_full/shard_*.parquet",
       "--group", f"baseline_ensemble_3x46={args.base},ts46_s2,v2full"]
for r in runs:
    cmd += ["--run", f"{r}=data/bench3/{r}/shard_*.parquet", "--versus", f"{r},{args.base}"]
txt = subprocess.run(cmd, capture_output=True, text=True)
print(txt.stdout[-6000:], txt.stderr[-2000:])
gap = json.load(open(args.out + ".gap.json"))
res = {"benchmark": gap, "runs": {}}
base_test = json.load(open("data/runs/ts46_s1/metrics.json"))
keys = ("pairwise_likes", "within_rho_likes", "nll_likes", "pairwise_views", "nll_views", "own_p90_auc_rel_like")
res["baseline_test_metrics"] = {k: base_test[k] for k in keys}
print(f"\n{'run':<16} {'bench acc':>9} {'gap vs base':>20} {'2x acc':>7} {'conf share':>10} {'conf acc':>8} | dev pairs 1.25x  2x  tac | val nll  val pw-likes | test pw-likes nll-likes")
for r in [args.base] + runs:
    g = gap["models"][r]
    row = {"benchmark_accuracy": g["accuracy"], "benchmark_2x": g["by_gap"]["2.0x"]["accuracy"], "confident_share": g["confident"]["share"], "confident_accuracy": g["confident"]["accuracy"]}
    v = gap["versus"].get(f"{r},{args.base}")
    if v:
        row["gap_vs_base"] = [v["accuracy_gap"], *v["accuracy_gap_ci"]]
    mp = Path(f"data/runs/{r}/metrics.json")
    if mp.exists():
        m = json.load(open(mp))
        pick = next(h for h in m["history"] if h["step"] == m["picked_step"])
        row.update(picked_step=m["picked_step"], dev={k: pick[k] for k in pick if k.startswith("ctl_")}, val_nll=pick["val_nll"], val_pairwise_likes=pick.get("pairwise_likes"),
                   test={k: m[k] for k in keys}, args={k: m["args"][k] for k in ("init", "steps", "lr", "w_ctl", "ctl_loss", "ctl_weight", "only", "seed")}, ctl_train_pairs=m["ctl_train_pairs"],
                   history=[{k: h[k] for k in h if k.startswith("ctl_") or k in ("step", "val_nll", "pairwise_likes")} for h in m["history"]])
        sp = Path(f"data/runs/{r}/start.json")
        if sp.exists():
            s = json.load(open(sp))
            row["start"] = {k: s[k] for k in s if k.startswith("ctl_") or k in ("val_nll", "pairwise_likes")}
    res["runs"][r] = row
    gv = f"{100 * row['gap_vs_base'][0]:+.1f} [{100 * row['gap_vs_base'][1]:+.1f},{100 * row['gap_vs_base'][2]:+.1f}]" if "gap_vs_base" in row else "-"
    d = row.get("dev", {})
    t = row.get("test", {})
    print(f"{r:<16} {100 * row['benchmark_accuracy']:8.1f}% {gv:>20} {100 * row['benchmark_2x']:6.1f}% {100 * row['confident_share']:9.0f}% {100 * row['confident_accuracy']:7.1f}% | "
          + (f"{100 * d.get('ctl_acc_1.25x', 0):.1f} {100 * d.get('ctl_acc_2.0x', 0):.1f} {100 * d.get('ctl_acc_tac', 0):.1f} | {row['val_nll']:.4f} {row['val_pairwise_likes']:.4f} | {t['pairwise_likes']:.4f} {t['nll_likes']:.4f}" if d else ""))
json.dump(res, open(args.out, "w"), indent=1)
