#!/usr/bin/env python3
"""The post-type level correction (scripts/posttype/fit_correction.py) refitted for a model set trained on the combine
table: a ridge regression of the ensemble's error in log10 likes (actual minus predicted median) on the seven post-type
columns, on the accounts' own announcements from before 2026-04-23 only (train and validation splits of the original rows;
types from tracks/posttype/post_type.parquet). Half of the accounts are held out for the honesty check.

    python scripts/combine/fit_type_correction.py --table data/combine/table_te3l_broad --run a.pt,b.pt --name te3l_broad2 --out data/combine/type_correction.json
"""
import argparse, json
import numpy as np, polars as pl, torch
from xviral.features import post_type as PTF
from xviral.model import evaluate, train
from xviral.model.features import FeatureTable

ap = argparse.ArgumentParser(); ap.add_argument("--table", required=True); ap.add_argument("--run", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--types", default="data/tracks/posttype/post_type.parquet"); ap.add_argument("--ridge", type=float, default=20.0)
a = ap.parse_args(); torch.set_grad_enabled(False)
full = FeatureTable.load(a.table)
ids, split = full.t["post_id"].numpy(), full.t["split"].numpy()
PT = pl.DataFrame({"post_id": ids}).join(pl.read_parquet(a.types).select("post_id", "author_id", "pt_type", *PTF.TYPE_COLS).unique("post_id"), on="post_id", how="left", maintain_order="left")
ty = PT["pt_type"].fill_null("other").to_numpy()
rows = torch.from_numpy(np.flatnonzero((ty != "other") & (split < 2)))
preds = []
for ck in a.run.split(","):
    model, tcfg, c = train.load(ck)
    m = c["table_meta"]
    table = full.without_num(lambda n: n not in set(m["num_base"])).without_cat([x for x in full.meta["cat_names"] if x not in m["cat_names"]])
    p = evaluate.predict_rows(model, table, rows, tcfg, temperature=c.get("temperature") or {}, with_nll=False); preds.append(p["likes_q50"]); y = p["y_likes"]
res = y - np.mean(preds, 0); ok = ~np.isnan(res)
X = PT[rows.numpy()].select([pl.col(c).cast(pl.Float64) for c in PTF.TYPE_COLS]).to_numpy()[ok]; r = res[ok]
auth = PT["author_id"].to_numpy()[rows.numpy()][ok]; types = ty[rows.numpy()][ok]
fit = lambda X, r: np.linalg.solve(X.T @ X + a.ridge * np.eye(X.shape[1]), X.T @ r)      # noqa: E731
w = fit(X, r)
rng = np.random.default_rng(0); ua = np.unique(auth); half = np.isin(auth, rng.choice(ua, len(ua) // 2, replace=False))
wh = fit(X[half], r[half]); h = ~half
out = {"model": a.name, "checkpoints": a.run.split(","), "fitted_on": {"posts": int(len(r)), "accounts": int(len(ua)), "period": "before 2026-04-23 (train and val splits)", "ridge": a.ridge,
                                                                            "by_type": {t: int((types == t).sum()) for t in sorted(set(types))}},
       "held_out_accounts_check": {"posts": int(h.sum()), "mean_abs_log10_error_before": round(float(np.abs(r[h]).mean()), 4), "after": round(float(np.abs(r[h] - X[h] @ wh).mean()), 4),
                                   "mean_error_before": round(float(r[h].mean()), 4), "mean_error_after": round(float((r[h] - X[h] @ wh).mean()), 4),
                                   "by_type_mean_error_before_after": {t: [round(float(r[h & (types == t)].mean()), 4), round(float((r[h] - X[h] @ wh)[types[h] == t].mean()), 4), int((h & (types == t)).sum())] for t in sorted(set(types))}},
       "weights_log10": {c: round(float(v), 4) for c, v in zip(PTF.TYPE_COLS, w)}}
print(json.dumps(out, indent=1)); json.dump(out, open(a.out, "w"), indent=1)
