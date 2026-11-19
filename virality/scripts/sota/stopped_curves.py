#!/usr/bin/env python3
"""Curves of runs stopped early (no metrics.json), read from their training logs, next to the control's at the same steps.
    python scripts/sota/stopped_curves.py histdrop mask"""
import ast, json, re, sys
def curve(run):
    rows = {}
    for line in open(f"logs/sota_{run}.log"):
        m = re.search(r"curves @(\d+): (\{.*\})", line)
        if m: rows.setdefault(int(m.group(1)), {}).update(ast.literal_eval(m.group(2)))
        m = re.search(r"val @(\d+): nll ([\d.]+)  pairwise_likes ([\d.]+)", line)
        if m: rows.setdefault(int(m.group(1)), {}).update(val_nll=float(m.group(2)), val_pairs=float(m.group(3)))
    return rows
print("| Run | Step | Pairs train | Pairs val | Pairs test | NLL val (all heads) | Dev pairs 2x |\n|---|---|---|---|---|---|---|")
out = {}
for c in ["ctl"] + sys.argv[1:]:
    cs = [curve(f"{c}_s{s}") for s in (1, 2)]
    out[c] = cs
    for step in sorted(set(cs[0]) & set(cs[1])):
        two = lambda k, f=100, d=1: " / ".join(f"{f * x[step][k]:.{d}f}" for x in cs)
        print(f"| {c} | {step} | {two('train_pairwise_likes')} | {two('val_pairs')} | {two('test_pairwise_likes')} | {two('val_nll', 1, 4)} | {two('dev_acc_2.0x')} |")
json.dump(out, open("data/sota/results/stopped_curves.json", "w"), indent=1)
