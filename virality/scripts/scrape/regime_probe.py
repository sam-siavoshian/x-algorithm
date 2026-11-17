#!/usr/bin/env python3
"""What a model trained with the sampling input has learned about it: the same post scored under each value of
`src_regime`. One row per account (its last row, so the changed row is in nobody's history).

    python scripts/scrape/regime_probe.py --run histsrc_s1=... --run allsrc_s1=... --out data/scrape_track/eval/regime_probe.json

For held-out scrape accounts: predicted views under each capture-age value over the prediction under home_24h_plus (the
model's maturity curve, to set against the measured growth of likes), and under search_sample / stream_sample.
For existing test accounts with views (search-sampled): the same ratios, which is the level shift a forecast would take
if the row were declared home-sampled.
"""
import argparse, json
from pathlib import Path
import numpy as np, polars as pl, torch
from xviral.model import evaluate, features
from xviral.model import train as tr
from xviral.model.features import FeatureTable

ap = argparse.ArgumentParser(); ap.add_argument("--table", default="data/scrape_track/table"); ap.add_argument("--run", action="append", required=True); ap.add_argument("--out", required=True)
args = ap.parse_args()
full = FeatureTable.load(args.table, "cpu")
info = pl.read_parquet(Path(args.table) / "row_info.parquet")
names = dict(features.CAT_SPEC)["src_regime"]
ci = full.meta["cat_names"].index("src_regime")
split, aidx, new = full.t["split"].numpy(), full.t["author_idx"].numpy(), info["is_new"].to_numpy()
yv = full.t["y_views"].numpy()
a_end = full.t["a_end"].numpy()
last = np.zeros(full.n, bool); last[np.unique(a_end) - 1] = True           # the last row of every account
a_start = full.t["a_start"].numpy()
long_hist = (a_end - a_start) >= 6
sets = {"home_heldout_last_post_5plus_history": np.flatnonzero(last & new & (split == 2) & long_hist),
        "orig_test_last_post_with_views": np.flatnonzero(last & ~new & (split == 2) & ~np.isnan(yv) & full.t["audience"].numpy())}
out = {}
for spec in args.run:
    name, path = spec.split("=", 1)
    model, tcfg, ck = tr.load(path, "cpu")
    m = ck["table_meta"]
    table = full.without_num(lambda n: n not in set(m["num_base"]))
    assert "src_regime" in m["cat_names"]
    out[name] = {}
    for sname, rows in sets.items():
        res = {}
        base_cat = table.t["cat"].clone()
        for k, rg in enumerate(names):
            if rg in ("unknown", "full_timeline"):      # no training row carries these in this table
                continue
            c = base_cat.clone(); c[rows, ci] = k
            t2 = FeatureTable(meta=table.meta, t={**table.t, "cat": c}, texts=table.texts)
            p = evaluate.predict_rows(model, t2, torch.from_numpy(rows), tcfg, temperature=ck.get("temperature") or None, with_nll=False)
            res[rg] = {"views_q50": p["views_q50"], "likes_q50": p["likes_q50"]}
        true = evaluate.predict_rows(model, table, torch.from_numpy(rows), tcfg, temperature=ck.get("temperature") or None, with_nll=False)
        o = {"rows": int(len(rows)), "true_regime_counts": {names[k]: int(c) for k, c in enumerate(np.bincount(base_cat[rows, ci].numpy(), minlength=len(names))) if c}}
        for ref in ("home_24h_plus", "search_sample"):
            o[f"median_predicted_views_over_{ref}"] = {rg: float(10 ** np.median(res[rg]["views_q50"] - res[ref]["views_q50"])) for rg in res}
            o[f"median_predicted_likes_over_{ref}"] = {rg: float(10 ** np.median(res[rg]["likes_q50"] - res[ref]["likes_q50"])) for rg in res}
        y = yv[rows]; ok = ~np.isnan(y)
        o["median_predicted_over_actual_views_true_regime"] = float(10 ** np.median(true["views_q50"][ok] - y[ok]))
        out[name][sname] = o
    json.dump(out, open(args.out, "w"), indent=1)
print(json.dumps(out, indent=1))
