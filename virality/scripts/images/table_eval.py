#!/usr/bin/env python3
"""Ordinary same-account test pairs, in-table: does the image input help the model rank an account's own posts?

    python scripts/images/table_eval.py --run img=data/images/runs/img_s1 --run ctl=data/images/runs/ctl_s1 ... --out data/images/table_eval.json

Each run is scored on its own table (<run>/table.txt names it); both tables hold the same rows in the same order.
Test-split audience rows; pairs of posts by one account within the training config's pair window whose likes differ
by 1.25x or more (log10 0.097); split by whether both posts, exactly one, or neither has a fetched image. Members of the
same kind (img_*, ctl_*) are averaged into an ensemble. Intervals: accounts resampled, paired difference img - ctl.
"""
import argparse, json, os
import numpy as np, torch
from xviral.model import evaluate, train
from xviral.model.features import FeatureTable
ap = argparse.ArgumentParser(); ap.add_argument("--run", action="append", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
preds, tables = {}, {}
for spec in a.run:
    name, path = spec.split("=", 1)
    tp = open(os.path.join(path, "table.txt")).read().strip()
    if tp not in tables: tables[tp] = FeatureTable.load(tp)
    T = tables[tp]
    cache = os.path.join(path, "test_pred.npz")
    if os.path.exists(cache):
        z = np.load(cache); preds[name] = (z["row"], z["s"], z["y"]); continue
    model, tcfg, ck = train.load(os.path.join(path, "best.pt"), "cpu")
    rows = T.rows_of_split("test", audience_only=True)
    with torch.no_grad():
        p = evaluate.predict_rows(model, T, rows, tcfg, temperature=ck.get("temperature"), with_nll=False)
    np.savez(cache, row=p["row"], s=p["likes_mean"], y=p["y_likes"]); preds[name] = (p["row"], p["likes_mean"], p["y_likes"])
    print("scored", name, len(p["row"]), flush=True)
T = next(iter(tables.values()))
row = next(iter(preds.values()))[0]
assert all((v[0] == row).all() for v in preds.values())
y = next(iter(preds.values()))[2]
author = T.t["author_idx"][row].numpy(); created = T.t["created_s"][row].numpy()
has_img = T.t["has_aux_img"][row].numpy() if "has_aux_img" in T.t else np.load(os.path.join(open(os.path.join(a.run[0].split("=", 1)[1], "table.txt")).read().strip(), "has_aux_img.npy"))[row]
if "has_aux_img" not in T.t:
    for tp in tables:
        if os.path.exists(os.path.join(tp, "has_aux_img.npy")): has_img = np.load(os.path.join(tp, "has_aux_img.npy"))[row]
i, j = evaluate.same_author_pairs(author, created, y, 30 * 86400, np.log10(1.25))
d = y[i] - y[j]
groups = {}
for name, (_, s, _) in preds.items():
    groups.setdefault(name.split("_")[0], []).append(s)
S = {g: np.mean(v, 0) for g, v in groups.items()} | {n: v[1] for n, v in preds.items()}
rng = np.random.default_rng(0)
def acc(mask, key):
    c = ((S[key][i] - S[key][j]) * d > 0)[mask].astype(float); au = author[i][mask]
    u, inv = np.unique(au, return_inverse=True); sm = np.bincount(inv, c); n = np.bincount(inv)
    idx = rng.integers(0, len(u), (1000, len(u))); b = sm[idx].sum(1) / n[idx].sum(1)
    return {"pairs": int(mask.sum()), "accounts": int(len(u)), "accuracy": round(float(c.mean()), 4), "lo": round(float(np.quantile(b, .025)), 4), "hi": round(float(np.quantile(b, .975)), 4)}
def diff(mask, k1, k2):
    c1 = ((S[k1][i] - S[k1][j]) * d > 0)[mask].astype(float); c2 = ((S[k2][i] - S[k2][j]) * d > 0)[mask].astype(float); au = author[i][mask]
    u, inv = np.unique(au, return_inverse=True); s1 = np.bincount(inv, c1); s2 = np.bincount(inv, c2); n = np.bincount(inv)
    idx = rng.integers(0, len(u), (1000, len(u))); b = (s1[idx].sum(1) - s2[idx].sum(1)) / n[idx].sum(1)
    return {"diff": round(float(c1.mean() - c2.mean()), 4), "lo": round(float(np.quantile(b, .025)), 4), "hi": round(float(np.quantile(b, .975)), 4)}
hi, hj = has_img[i], has_img[j]
masks = {"all_pairs": np.ones(len(i), bool), "both_with_image": hi & hj, "one_with_image": hi ^ hj, "at_least_one_image": hi | hj, "no_image": ~hi & ~hj,
         "gap_2x_at_least_one_image": (hi | hj) & (np.abs(d) >= np.log10(2))}
out = {"design": "test-split audience rows of the image-subset table, same account, within 30 days, likes differ 1.25x+", "models": list(preds)}
for mk, m in masks.items():
    out[mk] = {k: acc(m, k) for k in S}
    if "img" in S and "ctl" in S: out[mk]["img_minus_ctl"] = diff(m, "img", "ctl")
    print(mk, {k: v.get("accuracy", v.get("diff")) for k, v in out[mk].items()}, out[mk].get("img_minus_ctl"))
json.dump(out, open(a.out, "w"), indent=1)
