#!/usr/bin/env python3
"""Same-account pairs and forecast level on slices of the scrape table, for any number of checkpoints.

    python scripts/scrape/eval_table.py --table data/scrape_track/table --run name=best.pt [...] --out data/scrape_track/eval/slices.json [--merge]

Slices (train, validation and test are reported side by side):
  orig_train_audience     a fixed sample of 600 audience accounts' training rows of the existing table
  orig_val_audience       the existing validation rows (audience)
  orig_test_audience      the existing test rows every model reports (audience, from 2026-04-23)
  orig_test_startup / under_10k   the same for company and founder accounts / accounts under 10,000 followers
  home_train              a fixed sample of 500 scrape accounts that the scrape models trained on
  home_val                scrape validation accounts (used for checkpoint selection where they are audience accounts)
  home_heldout            scrape accounts no model trained on or selected on
  home_heldout_12h        the same, only posts captured 12 hours or more after posting (pairs compare like with like)
  home_heldout_20plus / _under20 / _audience / _startup    by account size in the scrape, audience rule, startup list
Per slice and target the tallies of scripts/expand/eval_table.py, with per-account counts for intervals.
"""
from __future__ import annotations

import argparse, importlib.util, json, logging
from pathlib import Path

import numpy as np, polars as pl, torch

from xviral.model import evaluate, episodes
from xviral.model import train as tr
from xviral.model.features import FeatureTable

spec = importlib.util.spec_from_file_location("expand_eval", Path(__file__).resolve().parents[1] / "expand" / "eval_table.py")
ee = importlib.util.module_from_spec(spec); spec.loader.exec_module(ee)
log = logging.getLogger("eval")

ap = argparse.ArgumentParser()
ap.add_argument("--table", default="data/scrape_track/table"); ap.add_argument("--run", action="append", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--merge", action="store_true"); ap.add_argument("--new-rows", default="data/scrape_track/new_rows.parquet")
args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
full = FeatureTable.load(args.table, "cpu")
info = pl.read_parquet(Path(args.table) / "row_info.parquet")
assert (info["post_id"].to_numpy() == full.t["post_id"].numpy()).all()
lt = pl.read_parquet("data/tracks/launch/accounts_labeled.parquet").select("author_id", pl.col("akind").alias("lt_kind"))
rows_h = pl.read_parquet("data/work/dataset/rows.parquet", columns=["author_id", "author_handle"]).unique("author_id").with_columns(pl.col("author_handle").str.to_lowercase().alias("handle"))
sa = pl.read_parquet("data/tracks/data/startup_accounts.parquet").select("handle", pl.col("kind").alias("sa_kind"))
nr = pl.read_parquet(args.new_rows, columns=["post_id", "acct_n", "age_h", "acct_kind"])
info = info.join(lt, on="author_id", how="left").join(rows_h.select("author_id", "handle"), on="author_id", how="left").join(sa, on="handle", how="left").join(nr, on="post_id", how="left")
split, aud, fol = full.t["split"].numpy(), full.t["audience"].numpy(), full.t["followers"].numpy()
is_new = info["is_new"].to_numpy()
aidx = full.t["author_idx"].numpy()
kind = info.select(pl.coalesce("lt_kind", "sa_kind", "acct_kind").alias("k"))["k"].to_numpy()
acct_n = info["acct_n"].fill_null(0).to_numpy(); age = info["age_h"].fill_null(0).to_numpy()
rng = np.random.default_rng(7)
def sample_accounts(mask, k):
    a = np.unique(aidx[mask]); pick = rng.choice(a, min(k, len(a)), replace=False)
    return mask & np.isin(aidx, pick)
ot = (split == 2) & aud & ~is_new
ho = (split == 2) & is_new
startup = np.isin(kind, ["company", "founder"])
slices = {"orig_train_audience": sample_accounts((split == 0) & aud & ~is_new, 600), "orig_val_audience": (split == 1) & aud & ~is_new, "orig_test_audience": ot,
          "orig_test_startup": ot & startup, "orig_test_under_10k": ot & (fol > 0) & (fol < 10_000),
          "home_train": sample_accounts((split == 0) & is_new & (acct_n >= 5), 500), "home_val": (split == 1) & is_new, "home_heldout": ho, "home_heldout_12h": ho & (age >= 12),
          "home_heldout_20plus": ho & (acct_n >= 20), "home_heldout_under20": ho & (acct_n < 20), "home_heldout_audience": ho & aud, "home_heldout_startup": ho & startup,
          "home_heldout_under_10k": ho & (fol > 0) & (fol < 10_000), "home_heldout_20plus_12h": ho & (acct_n >= 20) & (age >= 12)}
need = np.zeros(full.n, bool)
for m in slices.values():
    need |= m
rows = torch.from_numpy(np.flatnonzero(need))
log.info("rows to score: %d; slices %s", len(rows), {k: int(v.sum()) for k, v in slices.items()})
out = {"slices": {k: int(v.sum()) for k, v in slices.items()}, "models": {}}
keep = {}
if args.merge and Path(args.out).exists():
    prev = json.load(open(args.out)); assert prev["slices"] == out["slices"]; out["models"] = prev["models"]
    keep = dict(np.load(args.out.replace(".json", ".npz"), allow_pickle=True))
for spec_ in args.run:
    name, path = spec_.split("=", 1)
    model, tcfg, ck = tr.load(path, "cpu")
    m = ck["table_meta"]
    table = full.without_num(lambda n: n not in set(m["num_base"]))
    table = table.without_cat([c for c in full.meta["cat_names"] if c not in m["cat_names"]])
    assert table.meta["num_names"] == m["num_names"] and table.meta["cat_names"] == m["cat_names"], name
    pred = evaluate.predict_rows(model, table, rows, tcfg, temperature=ck.get("temperature") or None)
    ctx = np.concatenate([episodes.build(table, rows[i:i + 2048], model.cfg, tcfg, train=False).ctx.numpy() for i in range(0, len(rows), 2048)])
    pred["own_med_likes"], pred["own_med_views"], pred["own_has_likes"], pred["own_has_views"] = ctx[:, 3], ctx[:, 5], ctx[:, 7], ctx[:, 8]
    pos = np.full(full.n, -1); pos[pred["row"]] = np.arange(len(pred["row"]))
    out["models"][name] = {}
    for sname, mask in slices.items():
        r = ee.tallies(pred, table, tcfg, pos[np.flatnonzero(mask)])
        for target in ("likes", "views"):
            if target in r:
                for k in ("_acct", "_n", "_c"):
                    keep[f"{name}|{sname}|{target}|{k[1:]}"] = r[target].pop(k)
        out["models"][name][sname] = r
    # per-row predictions for the level analysis (views): post id, actual, predicted median
    np.savez(Path(args.out).parent / f"pred_{name}.npz", post_id=full.t["post_id"].numpy()[pred["row"]], y_views=pred["y_views"], q50_views=pred["views_q50"], y_likes=pred["y_likes"], q50_likes=pred["likes_q50"],
             own_med_views=pred["own_med_views"], own_has_views=pred["own_has_views"])
    log.info("%s: %s", name, {k: (round(v["views"]["accuracy"], 4), v["views"]["pairs"]) for k, v in out["models"][name].items() if "views" in v})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    np.savez(args.out.replace(".json", ".npz"), **keep)
