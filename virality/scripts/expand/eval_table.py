#!/usr/bin/env python3
"""Ordinary same-account pairs on slices of one table, for any number of checkpoints, with per-account tallies.

    python scripts/expand/eval_table.py --table data/expand/table --run name=path/best.pt [--run ...] --out data/expand/eval/table_slices.json

Every model is scored on the same rows of the same table (a model trained without src_regime reads the table without
that column), each row alone with its own history, as model.evaluate does. Slices:
  orig_test_audience      the test rows the existing models report on (audience accounts, 2026-04-23 on, existing rows)
  orig_test_company / founder / startup   the same, accounts labelled by the launch track or the startup account list
  orig_test_under_10k     the same, accounts with a known follower count under 10,000
  new_val_<source>        validation accounts of the added sources (accounts no expanded model trained on, before the cutoff)
  new_test                added rows from 2026-04-23 on
  *_launch_posts*         launch-type own posts (launch_labels) inside those slices: the level of the forecast for a launch post
Per slice and target (likes, views): pairs, share ranked correctly, mean within-account Spearman, NLL; and the per-account
(pairs, correct) tallies in an npz so differences between models get intervals from resampling accounts.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch

from xviral.model import evaluate
from xviral.model import train as tr
from xviral.model.features import FeatureTable

log = logging.getLogger("eval")


def tallies(pred: dict, table: FeatureTable, tcfg, sel: np.ndarray) -> dict:
    rows = pred["row"][sel]
    author = table.t["author_idx"][rows].numpy()
    created = table.t["created_s"][rows].numpy()
    res = {"rows": int(len(rows)), "accounts": int(len(np.unique(author)))}
    for target in ("likes", "views"):
        y, s = pred[f"y_{target}"][sel], pred[f"{target}_mean"][sel]
        if np.isnan(y).all():
            continue
        i, j = evaluate.same_author_pairs(author, created, y, tcfg.pair_window_days * 86400, tcfg.pair_margin)
        if not len(i):
            continue
        agree = ((s[i] > s[j]) == (y[i] > y[j])).astype(float)
        agree[s[i] == s[j]] = 0.5
        big = np.abs(y[i] - y[j]) >= np.log10(2.0)
        ua, inv = np.unique(author[i], return_inverse=True)
        res[target] = {"pairs": int(len(i)), "accuracy": float(agree.mean()), "pair_accounts": int(len(ua)),
                       "pairs_2x": int(big.sum()), "accuracy_2x": float(agree[big].mean()) if big.any() else None,
                       "nll": float(np.nanmean(pred[f"nll_{target}"][sel])), "labelled_rows": int((~np.isnan(y)).sum()),
                       "_acct": ua, "_n": np.bincount(inv, minlength=len(ua)), "_c": np.bincount(inv, weights=agree, minlength=len(ua))}
        rhos = []
        for a in np.unique(author):
            m = (author == a) & ~np.isnan(y)
            if m.sum() >= 8:
                r = evaluate.spearman(s[m], y[m])
                if not np.isnan(r):
                    rhos.append(r)
        res[target]["within_rho"] = float(np.mean(rhos)) if rhos else None
        res[target]["rho_accounts"] = len(rhos)
        # Level: predicted median against the outcome, and both against the account's own median in the visible history.
        ok = ~np.isnan(y)
        q50 = pred[f"{target}_q50"][sel]
        res[target]["median_predicted_over_actual"] = float(10 ** np.median(q50[ok] - y[ok]))
        med, has = pred[f"own_med_{target}"][sel], pred[f"own_has_{target}"][sel] > 0.5
        m = ok & has
        if m.sum() >= 20:
            res[target]["level"] = {"rows": int(m.sum()), "accounts": int(len(np.unique(author[m]))), "actual_over_own_median": float(10 ** np.median(y[m] - med[m])),
                                    "predicted_over_own_median": float(10 ** np.median(q50[m] - med[m])), "actual_beat_own_median": float((y[m] > med[m]).mean()),
                                    "predicted_beat_own_median": float((q50[m] > med[m]).mean()),
                                    "inside_q10_q90": float(((y[m] >= pred[f"{target}_q10"][sel][m]) & (y[m] <= pred[f"{target}_q90"][sel][m])).mean()),
                                    "above_q90": float((y[m] > pred[f"{target}_q90"][sel][m]).mean())}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/expand/table")
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--merge", action="store_true", help="add these models to the existing --out file")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    full = FeatureTable.load(args.table, "cpu")
    info = pl.read_parquet(Path(args.table) / "row_info.parquet")
    assert (info["post_id"].to_numpy() == full.t["post_id"].numpy()).all()
    # Account kinds: the launch track's labels (by author id) and the startup account list (by handle).
    lt = pl.read_parquet("data/tracks/launch/accounts_labeled.parquet").select("author_id", pl.col("akind").alias("lt_kind"))
    rows_h = pl.read_parquet("data/work/dataset/rows.parquet", columns=["author_id", "author_handle"]).unique("author_id").with_columns(pl.col("author_handle").str.to_lowercase().alias("handle"))
    newr = pl.read_parquet("data/expand/new_rows.parquet", columns=["author_id", "author_handle"]).unique("author_id").with_columns(pl.col("author_handle").str.to_lowercase().alias("handle"))
    hmap = pl.concat([rows_h, newr]).unique("author_id")
    sa = pl.read_parquet("data/tracks/data/startup_accounts.parquet").select("handle", pl.col("kind").alias("sa_kind"))
    info = info.join(lt, on="author_id", how="left").join(hmap.select("author_id", "handle"), on="author_id", how="left").join(sa, on="handle", how="left")
    split, aud = full.t["split"].numpy(), full.t["audience"].numpy()
    fol = full.t["followers"].numpy()
    is_new = info["is_new"].to_numpy()
    src = info["expand_src"].to_numpy()
    kind = info.select(pl.coalesce("lt_kind", "sa_kind").alias("k"))["k"].to_numpy()
    ot = (split == 2) & aud & ~is_new
    slices = {"orig_test_audience": ot, "orig_test_company": ot & (kind == "company"), "orig_test_founder": ot & (kind == "founder"),
              "orig_test_startup": ot & np.isin(kind, ["company", "founder"]), "orig_test_under_10k": ot & (fol > 0) & (fol < 10_000),
              "orig_test_startup_under_10k": ot & np.isin(kind, ["company", "founder"]) & (fol > 0) & (fol < 10_000),
              "new_test": (split == 2) & is_new}
    tp = pl.read_parquet("data/expand/typed_posts.parquet").filter(pl.col("ptype").is_in(["launch", "open_source"]))
    is_launch = np.isin(full.t["post_id"].numpy(), tp["post_id"].to_numpy())
    slices.update({"orig_test_launch_posts": ot & is_launch, "orig_test_launch_posts_startup": ot & is_launch & np.isin(kind, ["company", "founder"]),
                   "orig_test_launch_posts_under_10k": ot & is_launch & (fol > 0) & (fol < 10_000),
                   "new_val_launch_posts": (split == 1) & is_new & is_launch, "new_val_launch_posts_under_10k": (split == 1) & is_new & is_launch & (fol > 0) & (fol < 10_000),
                   "new_val_launch_posts_company": (split == 1) & is_new & is_launch & (kind == "company"), "new_val_launch_posts_founder": (split == 1) & is_new & is_launch & (kind == "founder")})
    for s_ in ("twitter100m", "zenodo_brands", "nitter_2026_replies"):
        m = (split == 1) & is_new & (src == s_)
        slices[f"new_val_{s_}"] = m
        slices[f"new_val_{s_}_under_10k"] = m & (fol > 0) & (fol < 10_000)
    need = np.zeros(full.n, bool)
    for m in slices.values():
        need |= m
    if args.smoke:                   # a quick check of the code path on a few accounts
        keep_a = np.unique(full.t["author_idx"].numpy()[need])[::40]
        need &= np.isin(full.t["author_idx"].numpy(), keep_a)
        slices = {k: v & need for k, v in slices.items()}
    rows = torch.from_numpy(np.flatnonzero(need))
    log.info("rows to score: %d; slices %s", len(rows), {k: int(v.sum()) for k, v in slices.items()})
    out: dict = {"slices": {k: int(v.sum()) for k, v in slices.items()}, "models": {}}
    keep: dict = {}
    if args.merge and Path(args.out).exists():      # add models to an earlier result on the same rows
        prev = json.load(open(args.out))
        assert prev["slices"] == out["slices"]
        out["models"] = prev["models"]
        keep = dict(np.load(args.out.replace(".json", ".npz"), allow_pickle=True))
    for spec in args.run:
        name, path = spec.split("=", 1)
        model, tcfg, ck = tr.load(path, "cpu")
        m = ck["table_meta"]
        table = full.without_num(lambda n: n not in set(m["num_base"]))
        table = table.without_cat([c for c in full.meta["cat_names"] if c not in m["cat_names"]])
        assert table.meta["num_names"] == m["num_names"] and table.meta["cat_names"] == m["cat_names"], name
        pred = evaluate.predict_rows(model, table, rows, tcfg, temperature=ck.get("temperature") or None)
        from xviral.model import episodes
        ctx = np.concatenate([episodes.build(table, rows[i:i + 2048], model.cfg, tcfg, train=False).ctx.numpy() for i in range(0, len(rows), 2048)])
        pred["own_med_likes"], pred["own_med_views"], pred["own_has_likes"], pred["own_has_views"] = ctx[:, 3], ctx[:, 5], ctx[:, 7], ctx[:, 8]
        pos = np.full(full.n, -1)
        pos[pred["row"]] = np.arange(len(pred["row"]))
        out["models"][name] = {}
        for sname, mask in slices.items():
            r = tallies(pred, table, tcfg, pos[np.flatnonzero(mask)])
            for target in ("likes", "views"):
                if target in r:
                    for k in ("_acct", "_n", "_c"):
                        keep[f"{name}|{sname}|{target}|{k[1:]}"] = r[target].pop(k)
            out["models"][name][sname] = r
        log.info("%s: %s", name, {k: (round(v["likes"]["accuracy"], 4), v["likes"]["pairs"]) for k, v in out["models"][name].items() if "likes" in v})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    np.savez(args.out.replace(".json", ".npz"), **keep)


if __name__ == "__main__":
    main()
