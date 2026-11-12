#!/usr/bin/env python3
"""Fit the scale that turns a score difference between two versions into a probability.

    python scripts/fit_pair_confidence.py --table <work>/model/table --checkpoint a/best.pt [--checkpoint b/best.pt] \
        --tac tac_serve.json --out pair_confidence.json

Likes: the scale fitted on held-out same-account, same-topic pairs (scripts/evidence/tac_serve_eval.py),
which is the comparison the product makes. Views: there are too few such pairs with view counts on
both sides, so the view scale is the one fitted on ordinary same-account pairs from validation
accounts, shrunk by the ratio the like scales show between the two kinds of pair (same-topic pairs
are harder to tell apart than two unrelated posts, and an unshrunk scale would be overconfident).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from xviral.model import evaluate
from xviral.model.features import FeatureTable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--tac", required=True, help="output of tac_serve_eval.py for the same checkpoints")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    table = FeatureTable.load(args.table, args.device)
    if len(args.checkpoint) > 1:
        from xviral.model import ensemble

        model, tcfg, _ = ensemble.load(args.checkpoint, args.device)
        temps = None
    else:
        from xviral.model import train

        model, tcfg, ck = train.load(args.checkpoint[0], args.device)
        temps = ck.get("temperature") or None
    rows = table.rows_of_split("val", audience_only=True)
    pred = evaluate.predict_rows(model, table, rows, tcfg, temperature=temps, with_nll=False)
    author = table.t["author_idx"][rows].cpu().numpy()
    created = table.t["created_s"][rows].cpu().numpy()
    generic = {}
    for target in ("likes", "views"):
        y = pred[f"y_{target}"]
        i, j = evaluate.same_author_pairs(author, created, y, tcfg.pair_window_days * 86400, 0.1)
        if len(i) < 200:
            continue
        s = pred[f"{target}_q50"][i] - pred[f"{target}_q50"][j]
        generic[target] = {"scale": evaluate.fit_pair_scale(s, y[i] > y[j]), "pairs": int(len(i)), "accuracy": float(((s > 0) == (y[i] > y[j])).mean())}
    tac = json.load(open(args.tac))
    k_tac = tac["likes"]["confidence_scale"]
    shrink = k_tac / generic["likes"]["scale"]
    out = {"likes": k_tac, "provenance": {
        "likes": f"same-account, same-topic pairs from accounts not trained on ({args.tac})",
        "generic_same_account_pairs": generic, "same_topic_shrink": shrink, "checkpoints": args.checkpoint}}
    if "views" in generic:
        out["views"] = generic["views"]["scale"] * min(shrink, 1.0)
        out["provenance"]["views"] = "validation-account pairs, shrunk by the same-topic ratio measured on likes"
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
