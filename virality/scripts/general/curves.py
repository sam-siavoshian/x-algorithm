#!/usr/bin/env python3
"""Train, validation and test curves of the general track's runs as one markdown table per run.
    python scripts/general/curves.py data/general/runs/broad_s1 [...] > data/general/eval/curves.md"""
import json, sys
SETS = [("train", "Train"), ("val_aud", "Validation, tech audience"), ("val_broad", "Validation, new sources"), ("test_aud", "Test, tech audience"), ("bed", "Test bed")]
for d in sys.argv[1:]:
    c = json.load(open(d + "/curves.json"))
    print(f"#### {d.split('/')[-1]}: {c['train_rows']:,} training rows, roles {c['args']['roles']}, learning rate {c['args']['lr']}, chosen step {c['best_step']} (on {c['args']['select']})\n")
    print("| Step | " + " | ".join(f"{t}: like NLL, like pairs" for _, t in SETS) + " | Views pairs: validation audience, new sources, test audience, bed |")
    print("|---|" + "---|" * (len(SETS) + 1))
    for r in c["curves"]:
        v = lambda k: "" if r[k].get("pairwise_views") is None else f"{100 * r[k]['pairwise_views']:.1f}%"
        print(f"| {r['step']} | " + " | ".join(f"{r[k]['nll_likes']:.3f}, {100 * r[k]['pairwise_likes']:.1f}%" for k, _ in SETS) + f" | {v('val_aud')}, {v('val_broad')}, {v('test_aud')}, {v('bed')} |")
    print()
