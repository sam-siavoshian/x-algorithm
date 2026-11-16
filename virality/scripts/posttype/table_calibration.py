#!/usr/bin/env python3
"""Predicted against actual level by post type on the test split of the feature table (no draft path: fast, for
comparing variants). Per type: the median of likes over the account's own median in the visible history, actual and
predicted, and same-account pair accuracy among launch-type posts.

    python scripts/posttype/table_calibration.py --run name=ckpt:table [--run ...] --out data/posttype/table_calibration.json
"""
import argparse, json, numpy as np, polars as pl, torch
from xviral.model import evaluate, train
from xviral.model.features import FeatureTable
ap = argparse.ArgumentParser(); ap.add_argument("--run", action="append", required=True); ap.add_argument("--out", required=True); ap.add_argument("--split", default="test")
ap.add_argument("--types", default="data/work/model/table_pt/post_type.parquet"); ap.add_argument("--other-sample", type=int, default=20000)
args = ap.parse_args()
torch.set_grad_enabled(False)
PT = pl.read_parquet(args.types).select("post_id", "pt_type", "author_id")
tables, out = {}, {}
for spec in args.run:
    name, rest = spec.split("=", 1); ck, tdir = rest.split(":")
    if tdir not in tables: tables[tdir] = FeatureTable.load(tdir)
    full = tables[tdir]
    model, tcfg, c = train.load(ck)
    names = c["table_meta"]["num_names"]; keep = [full.meta["num_names"].index(n) for n in names]
    table = FeatureTable(meta={**c["table_meta"], "n": full.n}, t={**full.t, "num": full.t["num"][:, keep]}, texts=None)
    ids = table.t["post_id"].numpy()
    assert np.array_equal(ids, PT["post_id"].to_numpy())
    ty = PT["pt_type"].to_numpy(); sp = table.t["split"].numpy() == {"train": 0, "val": 1, "test": 2}[args.split]
    rng = np.random.default_rng(0)
    typed = np.flatnonzero(sp & (ty != "other")); oth = np.flatnonzero(sp & (ty == "other")); oth = rng.choice(oth, min(args.other_sample, len(oth)), replace=False)
    rows = torch.from_numpy(np.sort(np.r_[typed, oth]))
    pred = evaluate.predict_rows(model, table, rows, tcfg, temperature=c.get("temperature") or {}, with_nll=False)
    r = pred["row"]; y = pred["y_rel_like"]; p = pred["rel_like_q50"]; ok = ~np.isnan(y)
    res = {}
    for k in ["launch", "open_source", "funding", "feature", "hiring", "milestone", "demo", "other"]:
        m = ok & (ty[r] == k)
        if m.sum() < 5: continue
        res[k] = {"posts": int(m.sum()), "actual_x": round(float(10 ** np.median(y[m])), 3), "predicted_x": round(float(10 ** np.median(p[m])), 3),
                  "actual_mean_log10": round(float(y[m].mean()), 4), "predicted_mean_log10": round(float(p[m].mean()), 4),
                  "spearman": round(float(evaluate.spearman(y[m], p[m])), 3)}
    # same-account pairs of launch-type posts, absolute likes head
    m = np.isin(ty[r], ["launch", "open_source"]) & ~np.isnan(pred["y_likes"])
    a = PT["author_id"].to_numpy()[r[m]]; yy = pred["y_likes"][m]; ss = pred["likes_q50"][m]
    c_, n_ = 0, 0; big_c, big_n = 0, 0
    for u in np.unique(a):
        i = np.flatnonzero(a == u)
        if len(i) < 2: continue
        dy = yy[i][:, None] - yy[i][None]; ds = ss[i][:, None] - ss[i][None]; up = np.triu(np.ones_like(dy, bool), 1)
        g = up & (np.abs(dy) >= np.log10(1.25)); c_ += ((dy * ds) > 0)[g].sum(); n_ += g.sum()
        g = up & (np.abs(dy) >= np.log10(2)); big_c += ((dy * ds) > 0)[g].sum(); big_n += g.sum()
    res["launch_pairs_in_table"] = {"pairs": int(n_), "accuracy": round(float(c_ / max(n_, 1)), 4), "pairs_2x": int(big_n), "accuracy_2x": round(float(big_c / max(big_n, 1)), 4)}
    out[name] = res
    print(name); [print("  ", k, v) for k, v in res.items()]
json.dump(out, open(args.out, "w"), indent=1)
