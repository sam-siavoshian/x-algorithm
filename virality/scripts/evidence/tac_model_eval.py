#!/usr/bin/env python3
"""Does the model pick the better of two versions when the account and the topic are held fixed?

    python scripts/evidence/tac_model_eval.py --rows <work>/dataset/rows.parquet --table <work>/model/table \
        --checkpoint <run>/best.pt [--checkpoint ...] --out tac_eval.json

Pairs: two posts by one account that quote the same post or carry the same link, with different
wording, posted within 24 hours of each other (so the model never sees one side's outcome while
scoring the other: outcomes younger than 24 hours are hidden from the history), in the test period.
Reported: accuracy overall, with posting order balanced, per kind of change (pairs that differ in
that attribute), and by confidence band. Confidence is calibrated on validation-account pairs.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl
import torch

from xviral.model import evaluate
from xviral.model.features import FeatureTable


def build_pairs(rows: pl.DataFrame, max_gap_h: float = 24.0, max_uses: int = 4) -> pl.DataFrame:
    u = rows.select("post_id", pl.col("urls").list.eval(pl.element().struct.field("expanded")).alias("u")).explode("u").filter(pl.col("u").is_not_null())
    u = u.with_columns(pl.col("u").str.to_lowercase().str.replace(r"^https?://(www\.)?", "").str.strip_chars_end("/").alias("key"))
    u = u.filter(~pl.col("key").str.contains(r"^(x\.com|twitter\.com|t\.co)") & (pl.col("key").str.len_chars() > 12) & pl.col("key").str.contains("/"))
    u = u.unique(["post_id", "key"]).select("post_id", pl.concat_str([pl.lit("url:"), pl.col("key")]).alias("key"))
    q = rows.filter(pl.col("quoted_post_id").is_not_null()).select("post_id", pl.concat_str([pl.lit("quote:"), pl.col("quoted_post_id").cast(pl.String)]).alias("key"))
    norm = pl.col("text").fill_null("").str.to_lowercase().str.replace_all(r"https?://\S+", "").str.replace_all(r"\s+", " ").str.strip_chars()
    keep = ["post_id", "author_id", "created_at_ms", "split", "likes", "views", "has_media", "ts_lines", "ts_words", "ts_open_question",
            "ts_url_first_line", "ts_url_at_end", "in_audience"]
    k = pl.concat([u, q]).join(rows.select(*keep, norm.alias("nt")), on="post_id")
    uses = k.group_by("author_id", "key").len().filter(pl.col("len").is_between(2, max_uses))
    k = k.join(uses.select("author_id", "key"), on=["author_id", "key"], how="semi")
    p = k.join(k, on=["author_id", "key"], suffix="_b").filter((pl.col("post_id") < pl.col("post_id_b")) & (pl.col("nt") != pl.col("nt_b")))
    return p.filter(((pl.col("created_at_ms_b") - pl.col("created_at_ms")).abs() <= max_gap_h * 3_600_000) & (pl.col("split") == pl.col("split_b")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    rows = pl.read_parquet(args.rows)
    pairs = build_pairs(rows)
    table = FeatureTable.load(args.table, args.device)
    if len(args.checkpoint) > 1:
        from xviral.model import ensemble

        model, tcfg, _ = ensemble.load(args.checkpoint, args.device)
        temps = None
    else:
        from xviral.model import train

        model, tcfg, ck = train.load(args.checkpoint[0], args.device)
        temps = ck.get("temperature") or None
    pos = {int(p): i for i, p in enumerate(table.t["post_id"].tolist())}
    need = sorted({pos[int(x)] for x in pl.concat([pairs["post_id"], pairs["post_id_b"]]).unique().to_list()})
    pred = evaluate.predict_rows(model, table, torch.tensor(need, device=table.device), tcfg, temperature=temps, with_nll=False)
    at = {r: i for i, r in enumerate(need)}
    res: dict = {"pairs_total": pairs.height}
    for target, key, col in (("likes", "likes_mean", "likes"), ("views", "views_mean", "views")):
        p = pairs.filter(pl.col(col).is_not_null() & pl.col(col + "_b").is_not_null())
        d = ((p[col].cast(pl.Float64) + 1).log() - (p[col + "_b"].cast(pl.Float64) + 1).log()).to_numpy()
        ok = np.abs(d) >= np.log(1.25)                     # near-ties carry no information about which version is better
        p, d = p.filter(pl.Series(ok)), d[ok]
        if p.height < 50:
            continue
        ia = np.array([at[pos[int(x)]] for x in p["post_id"].to_list()])
        ib = np.array([at[pos[int(x)]] for x in p["post_id_b"].to_list()])
        s = pred[key][ia] - pred[key][ib]
        correct = (s * d) > 0
        split = p["split"].to_numpy()
        a_first = (p["created_at_ms"] < p["created_at_ms_b"]).to_numpy()
        test, val = split == "test", split == "val"

        def acc(mask):
            return {"accuracy": float(correct[mask].mean()), "pairs": int(mask.sum())} if mask.sum() >= 30 else None

        # Order-balanced: average of accuracy where the true winner was posted first and where it was posted second.
        win_first = (d > 0) == a_first
        bal = np.mean([correct[test & win_first].mean(), correct[test & ~win_first].mean()]) if (test & win_first).sum() and (test & ~win_first).sum() else float("nan")
        out = {"test": acc(test), "test_order_balanced": float(bal), "later_post_wins_rate": float((~win_first[test]).mean()) if test.any() else None,
               "validation_accounts": acc(val), "train_period_reference": acc(split == "train")}
        # Confidence: a logistic fit of "A wins" on the score difference, fitted on validation-account pairs.
        if val.sum() >= 200:
            scale = np.linspace(0.2, 20, 100)
            y = (d[val] > 0).astype(float)
            nll = [-(y * np.log(1 / (1 + np.exp(-k * s[val])) + 1e-9) + (1 - y) * np.log(1 - 1 / (1 + np.exp(-k * s[val])) + 1e-9)).mean() for k in scale]
            k = float(scale[int(np.argmin(nll))])
            conf = np.maximum(1 / (1 + np.exp(-k * s)), 1 - 1 / (1 + np.exp(-k * s)))
            out["confidence_scale"] = k
            out["by_confidence"] = [{"band": f"{lo:.2f} to {hi:.2f}", "stated": float(conf[test & (conf >= lo) & (conf < hi)].mean()) if (test & (conf >= lo) & (conf < hi)).any() else None,
                                     **(acc(test & (conf >= lo) & (conf < hi)) or {"accuracy": None, "pairs": int((test & (conf >= lo) & (conf < hi)).sum())})}
                                    for lo, hi in ((0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01))]
        # Per kind of change: pairs where exactly one side has the attribute.
        kinds = {"image or video": ("has_media", lambda c: c.fill_null(False)), "three or more lines": ("ts_lines", lambda c: c >= 3),
                 "1.5x more words": None, "opens with a question": ("ts_open_question", lambda c: c > 0), "link in first line": ("ts_url_first_line", lambda c: c > 0)}
        per = {}
        for name, spec in kinds.items():
            if spec is None:
                fa = (p["ts_words"] >= 1.5 * p["ts_words_b"].clip(lower_bound=5)).to_numpy()
                fb = (p["ts_words_b"] >= 1.5 * p["ts_words"].clip(lower_bound=5)).to_numpy()
            else:
                c, f = spec
                fa, fb = f(p[c]).fill_null(False).to_numpy(), f(p[c + "_b"]).fill_null(False).to_numpy()
            m = test & (fa != fb)
            if m.sum() >= 30:
                with_wins = ((d > 0) == fa)[m]
                per[name] = {"pairs": int(m.sum()), "model_accuracy": float(correct[m].mean()), "version_with_it_wins": float(with_wins.mean()),
                             "model_prefers_version_with_it": float(((s > 0) == fa)[m].mean())}
        out["per_change"] = per
        res[target] = out
        print(target, json.dumps({k: v for k, v in out.items() if k not in ("by_confidence", "per_change")}, default=float))
        for b in out.get("by_confidence", []):
            print("   confidence", b)
        for n, v in per.items():
            print("   change", n, v)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=float)


if __name__ == "__main__":
    main()
