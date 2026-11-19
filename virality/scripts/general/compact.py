#!/usr/bin/env python3
"""Compact level table from a report.py JSON:  python scripts/general/compact.py data/general/eval/bed.json Label [Label ...]"""
import json, sys
r = json.load(open(sys.argv[1])); labels = sys.argv[2:]
slices = [k for k in r["slices"]]
for tg, title in (("level_likes", "likes"), ("level_views", "views")):
    print(f"#### Level, {title}: median account's predicted median over actual median; median absolute factor; share of accounts inside 1.5x\n")
    print("| Slice | Accounts | " + " | ".join(labels) + " |"); print("|---|---|" + "---|" * len(labels))
    for k in slices:
        b = r["models"][labels[0]][k].get(tg)
        if not b:
            continue
        cells = []
        for l in labels:
            v = r["models"][l][k].get(tg)
            cells.append("" if not v else f"{v['median_factor']:.2f}x ({v['ci'][0]:.2f} to {v['ci'][1]:.2f}); {v['median_abs_factor']:.2f}x; {100 * v['within_1.5x']:.0f}%")
        print(f"| {k.strip() if not k.startswith('  ') else '&nbsp;&nbsp;' + k.strip()} | {b['accounts']} | " + " | ".join(cells) + " |")
    print()
