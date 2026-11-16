#!/usr/bin/env python3
"""The pair benchmark by size of the real gap: accuracy, share of confident calls and accuracy among them.

    python scripts/evidence/gap_table.py --run name=glob.parquet [--run ...] [--group label=a,b,c] [--versus a,b] --out table.json

Same inputs and confidence as compare_runs.py (scale cross-fitted over accounts on all pairs whose likes differ by
at least 25%); every row is then cut by the real like ratio (1.25x, 2x, 3x, 5x) and by confidence band, with pair
counts. --part test|calibration restricts to one half of the benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import compare_runs as cr  # noqa: E402

GAPS = (1.25, 2.0, 3.0, 5.0)


def table(R: pl.DataFrame) -> dict:
    a = cr.analyse(R)
    P = a.pop("_pairs")
    key = P["key"].join(R.select("post_id", "post_id_b", "likes", "likes_b"), on=["post_id", "post_id_b"], how="left", maintain_order="left")
    d = np.abs(np.log((key["likes"].to_numpy() + 1.0) / (key["likes_b"].to_numpy() + 1.0)))
    correct, conf, auth = P["correct"], P["conf"], P["author"]
    a["by_gap"] = {}
    for g in GAPS:
        m = d >= np.log(g)
        c = m & (conf >= 0.6)
        a["by_gap"][f"{g}x"] = {"pairs": int(m.sum()), "accuracy": float(correct[m].mean()), "accuracy_ci": cr.cluster_ci(auth[m], lambda i: correct[m][i].mean()),
                                "confident_share": float(c.sum() / m.sum()), "confident_pairs": int(c.sum()),
                                "confident_accuracy": float(correct[c].mean()) if c.sum() else None,
                                "confident_accuracy_ci": cr.cluster_ci(auth[c], lambda i: correct[c][i].mean()) if c.sum() > 20 else None,
                                "bands": {f"{lo:.2f}-{min(hi, 1):.2f}": {"pairs": int((m & (conf >= lo) & (conf < hi)).sum()),
                                                                        "accuracy": float(correct[m & (conf >= lo) & (conf < hi)].mean()) if (m & (conf >= lo) & (conf < hi)).sum() else None}
                                          for lo, hi in ((0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01))}}
    a["_pairs"] = P
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--group", action="append", default=[])
    ap.add_argument("--versus", action="append", default=[])
    ap.add_argument("--part", default=None)
    ap.add_argument("--parts-from", default=None, help="parquet with (part, post_id, post_id_b) when --part is given")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = {r.split("=", 1)[0]: cr.load(r.split("=", 1)[1]) for r in args.run}
    if args.part:
        b = pl.read_parquet(args.parts_from).filter(pl.col("part") == args.part)
        b = b.select(pl.min_horizontal("post_id", "post_id_b").alias("post_id"), pl.max_horizontal("post_id", "post_id_b").alias("post_id_b"))
        runs = {n: R.join(b, on=["post_id", "post_id_b"], how="semi") for n, R in runs.items()}
    for g in args.group:
        label, members = g.split("=", 1)
        frames = [runs[m].rename({"s_likes": f"s_{i}"}) for i, m in enumerate(members.split(","))]
        J = frames[0]
        for f in frames[1:]:
            J = J.join(f.select("post_id", "post_id_b", pl.col("^s_\\d+$")), on=["post_id", "post_id_b"])
        runs[label] = J.with_columns(pl.mean_horizontal(pl.col("^s_\\d+$")).alias("s_likes")).drop(pl.col("^s_\\d+$"))
    res = {n: table(R) for n, R in runs.items()}
    vs = {v: cr.versus(res[v.split(",")[0]], res[v.split(",")[1]]) for v in args.versus}
    for r in res.values():
        r.pop("_pairs")
    json.dump({"models": res, "versus": vs}, open(args.out, "w"), indent=1)
    for n, r in res.items():
        print(f"\n{n}: {r['pairs']} pairs, {100 * r['accuracy']:.1f}% [{100 * r['accuracy_ci'][0]:.1f}, {100 * r['accuracy_ci'][1]:.1f}], scale {r['scale']:.2f}")
        print("  gap    pairs  correct           confident share (pairs)  correct there      | by confidence band: " + "  ".join(next(iter(r["by_gap"].values()))["bands"]))
        for g, x in r["by_gap"].items():
            ca = f"{100 * x['confident_accuracy']:.1f}%" if x["confident_accuracy"] is not None else "-"
            ci = f"[{100 * x['confident_accuracy_ci'][0]:.0f},{100 * x['confident_accuracy_ci'][1]:.0f}]" if x["confident_accuracy_ci"] else ""
            bands = "  ".join(f"{100 * b['accuracy']:.0f}% n={b['pairs']}" if b["accuracy"] is not None else "-" for b in x["bands"].values())
            print(f"  {g:>5} {x['pairs']:>6}  {100 * x['accuracy']:.1f}% [{100 * x['accuracy_ci'][0]:.0f},{100 * x['accuracy_ci'][1]:.0f}]   {100 * x['confident_share']:.0f}% ({x['confident_pairs']})   {ca} {ci}   | {bands}")
    for v, r in vs.items():
        print(f"{v}: pairs {r['pairs']}  accuracy gap {100 * r['accuracy_gap']:+.1f} [{100 * r['accuracy_gap_ci'][0]:+.1f},{100 * r['accuracy_gap_ci'][1]:+.1f}]"
              f"  confident-call gap {100 * r['confident_gap']:+.1f} [{100 * r['confident_gap_ci'][0]:+.1f},{100 * r['confident_gap_ci'][1]:+.1f}]")


if __name__ == "__main__":
    main()
