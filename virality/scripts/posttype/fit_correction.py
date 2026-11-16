#!/usr/bin/env python3
"""A small type-conditioned correction on top of a trained model: a ridge regression of the model's error in
log10 likes (actual minus predicted median) on the post type and, for launch-type posts, the launch traits.
Fitted on the account's own announcements in the training period only (train and val splits of the feature table).

    python scripts/posttype/fit_correction.py --run name=ckpt[,ckpt...] --out data/posttype/correction_<name>.json
The JSON holds one weight per column of xviral/features/post_type.py; the correction of a post is the dot product
with its raw (0/1) columns, added to the predicted log10(likes + 1). Ordinary posts get exactly zero.
"""
import argparse, json, numpy as np, polars as pl, torch
from xviral.features import post_type as PTF
from xviral.model import evaluate, train
from xviral.model.features import FeatureTable
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True); ap.add_argument("--table", default="data/work/model/table_pt"); ap.add_argument("--ridge", type=float, default=20.0); ap.add_argument("--with-traits", action="store_true", help="also fit the launch traits (default: the post type only)")
args = ap.parse_args(); torch.set_grad_enabled(False)
name, cks = args.run.split("=", 1)
PT = pl.read_parquet(f"{args.table}/post_type.parquet")
full = FeatureTable.load(args.table)
COLS = PTF.TYPE_COLS + (PTF.LAUNCH_TRAIT_COLS if args.with_traits else [])
ty = PT["pt_type"].to_numpy(); split = full.t["split"].numpy()
rows = torch.from_numpy(np.flatnonzero((ty != "other") & (split < 2)))
preds = []
for ck in cks.split(","):
    model, tcfg, c = train.load(ck)
    keep = [full.meta["num_names"].index(n) for n in c["table_meta"]["num_names"]]
    table = FeatureTable(meta={**c["table_meta"], "n": full.n}, t={**full.t, "num": full.t["num"][:, keep]}, texts=None)
    p = evaluate.predict_rows(model, table, rows, tcfg, temperature=c.get("temperature") or {}, with_nll=False); preds.append(p["likes_q50"]); y = p["y_likes"]
res = y - np.mean(preds, 0); ok = ~np.isnan(res)
X = PT[rows.numpy()].select([pl.col(c).cast(pl.Float64) for c in COLS]).to_numpy()[ok]; r = res[ok]
auth = PT["author_id"].to_numpy()[rows.numpy()][ok]
def fit(X, r): return np.linalg.solve(X.T @ X + args.ridge * np.eye(X.shape[1]), X.T @ r)
w = fit(X, r)
# honesty check inside the training period: fit on half the accounts, measure on the other half
rng = np.random.default_rng(0); ua = np.unique(auth); half = np.isin(auth, rng.choice(ua, len(ua) // 2, replace=False))
w_h = fit(X[half], r[half]); before = float(np.abs(r[~half]).mean()); after = float(np.abs(r[~half] - X[~half] @ w_h).mean())
out = {"model": name, "checkpoints": cks.split(","), "fitted_on": {"posts": int(len(r)), "accounts": int(len(ua)), "period": "before 2026-04-23 (train and val splits)", "ridge": args.ridge},
       "held_out_accounts_check": {"mean_abs_log10_error_before": round(before, 4), "after": round(after, 4), "mean_error_before": round(float(r[~half].mean()), 4), "mean_error_after": round(float((r[~half] - X[~half] @ w_h).mean()), 4)},
       "weights_log10": {c: round(float(v), 4) for c, v in zip(COLS, w)}}
print(json.dumps(out, indent=1)); json.dump(out, open(args.out, "w"), indent=1)
