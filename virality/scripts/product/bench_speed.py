#!/usr/bin/env python3
"""Request time of the launch advisor, original path against the fast path, on real accounts and real checkpoints.

    python scripts/product/bench_speed.py --posts data/work/hygiene/posts.parquet --scales research/15_pair_confidence.json \
        --sid data/work/emb/sid_codebook.npz --checkpoint a/best.pt --checkpoint b/best.pt --checkpoint c/best.pt \
        --accounts 12 --out speed.json

Accounts are drawn at random (fixed seed) among those with a follower count and at least 100 posts in the corpus.
Each account gets a cold request (nothing cached for it) with three versions, a warm repeat, and a warm request with
six versions. Both services are in the same process and see the same requests. The reports are compared field by
field; the largest relative difference and any non-numeric difference are written out.
"""
import argparse
import copy
import json
import math
import statistics
import time

import polars as pl

from xviral.app import service

V = [{"text": "Tired of slow deploys? We are launching something today. Details here https://example.com/launch #devtools"},
     {"text": "Introducing Relay 2\n\nDeploys in 9 seconds.\nRollbacks in one click.\nFree for open source.\n\nAvailable today.", "video": True},
     {"text": "Relay 2 is live. 9-second deploys, one-click rollbacks, free for open source.", "image": True},
     {"text": "We shipped Relay 2. Deploys are 9 seconds now, rollbacks take one click, and it is free for open-source projects."},
     {"text": "Relay 2: 9-second deploys.\n\nOne-click rollbacks.\nFree for OSS.", "video": True},
     {"text": "New: Relay 2. https://example.com/relay-2"}]


def diff(a, b, skip=("seconds", "timing_ms", "generated_at")):
    worst, other = 0.0, []

    def walk(x, y, p):
        nonlocal worst
        if isinstance(x, dict):
            for k in x:
                if k not in skip:
                    walk(x[k], y.get(k), f"{p}/{k}")
        elif isinstance(x, list):
            if len(x) != len(y):
                other.append(p)
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{p}[{i}]")
        elif isinstance(x, float) and isinstance(y, float):
            if not (math.isnan(x) and math.isnan(y)):
                worst = max(worst, abs(x - y) / max(abs(x), abs(y), 1e-12))
        elif x != y:
            other.append(p)
    walk(a, b, "")
    return worst, other


ap = argparse.ArgumentParser()
ap.add_argument("--posts", required=True)
ap.add_argument("--scales", required=True)
ap.add_argument("--sid")
ap.add_argument("--checkpoint", action="append", required=True)
ap.add_argument("--accounts", type=int, default=12)
ap.add_argument("--out", default="speed.json")
a = ap.parse_args()
acc = (pl.scan_parquet(a.posts).group_by("author_handle").agg(pl.len().alias("n"), pl.col("author_followers").drop_nulls().last().alias("f"))
       .filter((pl.col("n") >= 100) & pl.col("f").is_not_null()).collect().sort("author_handle").sample(a.accounts, seed=1)["author_handle"].to_list())
orig = service.load(a.checkpoint, a.scales, posts=a.posts, sid_codebook=a.sid, fast=False)
fast = service.load(a.checkpoint, a.scales, posts=a.posts, sid_codebook=a.sid, fast=True)
rows = []
for h in acc:
    r = {"handle": h}
    for name, svc in (("original", orig), ("fast", fast)):
        for case, nv in (("cold_3", 3), ("warm_3", 3), ("warm_6", 6)):
            req = {"handle": h, "versions": V[:nv], "when": "2026-11-18T16:00:00Z"}
            t = time.perf_counter()
            rep = svc.analyze(copy.deepcopy(req))
            r[f"{name}_{case}"] = time.perf_counter() - t
            r[f"{name}_{case}_report"] = rep
            r["history_posts"] = rep["account"]["history_posts"]
    worst, other = 0.0, []
    for case in ("cold_3", "warm_6"):
        w, o = diff(r.pop(f"original_{case}_report"), r.pop(f"fast_{case}_report"))
        worst, other = max(worst, w), other + o
    r.pop("original_warm_3_report"), r.pop("fast_warm_3_report")
    r["max_relative_difference"], r["non_numeric_differences"] = worst, other
    rows.append(r)
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) and k != "max_relative_difference" else v) for k, v in r.items()}), flush=True)
summary = {f"{n}_{c}": {"median_s": statistics.median(r[f"{n}_{c}"] for r in rows), "max_s": max(r[f"{n}_{c}"] for r in rows)}
           for n in ("original", "fast") for c in ("cold_3", "warm_3", "warm_6")}
summary["accounts"] = len(rows)
summary["max_relative_difference"] = max(r["max_relative_difference"] for r in rows)
summary["reports_with_non_numeric_differences"] = sum(bool(r["non_numeric_differences"]) for r in rows)
print(json.dumps(summary, indent=1))
json.dump({"summary": summary, "accounts": rows}, open(a.out, "w"), indent=1)
