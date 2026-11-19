#!/usr/bin/env python3
"""Test-bed tables from saved per-row predictions (scripts/general/predict_rows.py).

    python scripts/general/report.py --model Label=a,b [--model ...] [--customer Label] --base Label --out data/general/eval/bed

A model is one prediction file or the mean of several members' log10 predictions (an ensemble). For each slice of the test bed:
  pairs   ordinary same-account pairs (two posts within 7 days, outcomes 0.1 apart in log10), likes and views, share ranked right
  level   per account, log10(median predicted) - log10(median actual) over its scored posts. Reported: the median account's
          factor (1.00x is honest), the median absolute factor, and the share of accounts inside 1.5x
Intervals resample accounts (2,000 draws). Differences against --base are paired on accounts.
--customer Label reads that model's "_customer" predictions on the bed (source unknown), named "<Label> as customer".
Level correction: one log10 offset per sampling regime, the median account error on the general VALIDATION accounts, subtracted
on the bed ("<Label> + correction"). Written to <out>_level_correction.json.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, polars as pl
from xviral.model import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--table", default="data/general/table"); ap.add_argument("--pred", default="data/general/pred")
ap.add_argument("--model", action="append", required=True); ap.add_argument("--customer", action="append", default=[])
ap.add_argument("--base", required=True); ap.add_argument("--out", default="data/general/eval/bed")
args = ap.parse_args()
T = Path(args.table)
info = pl.read_parquet(T / "row_info.parquet")
author, created, fol = np.load(T / "author_idx.npy"), np.load(T / "created_s.npy"), info["followers"].to_numpy()
cat = np.load(T / "cat.npy"); meta = json.loads((T / "meta.json").read_text())
regime = cat[:, meta["cat_names"].index("src_regime")]
src = info["expand_src"].fill_null("orig").to_numpy(); tier = info["g_tier"].fill_null("").to_numpy(); seen = info["seen_by_expand"].to_numpy()
rng = np.random.default_rng(0)


def load(members: list[str], sname: str, suffix: str = "") -> dict:
    ps = [np.load(Path(args.pred) / f"{m}{suffix}__{sname}.npz") for m in members]
    out = {k: ps[0][k] for k in ps[0].files}
    for k in ("likes_mean", "likes_q50", "views_mean", "views_q50"):
        out[k] = np.mean([p[k] for p in ps], axis=0)
    assert all((p["row"] == ps[0]["row"]).all() for p in ps)
    return out


def boot(acc_ids: np.ndarray, fn, n=2000):
    ua, inv = np.unique(acc_ids, return_inverse=True)
    return ua, inv


def pair_tally(p: dict, sel: np.ndarray, target: str):
    rows = p["row"][sel]; y, s = p[f"y_{target}"][sel], p[f"{target}_mean"][sel]
    if np.isnan(y).all():
        return None
    i, j = evaluate.same_author_pairs(author[rows], created[rows], y, 7 * 86400, 0.1)
    if len(i) < 50:
        return None
    agree = ((s[i] > s[j]) == (y[i] > y[j])).astype(float); agree[s[i] == s[j]] = 0.5
    ua, inv = np.unique(author[rows][i], return_inverse=True)
    return {"acct": ua, "n": np.bincount(inv).astype(float), "c": np.bincount(inv, weights=agree)}


def acct_level(p: dict, sel: np.ndarray, target: str, offset=0.0):
    rows = p["row"][sel]; y, q = p[f"y_{target}"][sel], p[f"{target}_q50"][sel]
    a = author[rows]; ok = ~np.isnan(y)
    ua = np.unique(a[ok]); res = {}
    for u in ua:
        m = ok & (a == u)
        if m.sum() >= 10:
            off = offset.get(int(u), 0.0) if isinstance(offset, dict) else offset
            res[int(u)] = float(np.median(q[m]) - np.median(y[m]) - off)
    return res


def acct_feats(p: dict, sel: np.ndarray) -> dict:
    """Per account: log10 followers and the median of its visible-history median likes (log10), both known at serving time."""
    rows = p["row"][sel]; a = author[rows]; out = {}
    for u in np.unique(a):
        m = a == u
        out[int(u)] = (float(np.log10(fol[rows[m][0]] + 1)), float(np.median(p["own_med_likes"][sel][m])), int(regime[rows[m][0]]))
    return out


def ci(vals_fn, k, n=2000):
    idx = rng.integers(0, k, (n, k))
    b = np.array([vals_fn(i) for i in idx])
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def summarize(p: dict, mask_rows: np.ndarray, offset=0.0) -> dict:
    sel = np.flatnonzero(mask_rows[p["row"]])
    out = {"rows": int(len(sel)), "accounts": int(len(np.unique(author[p["row"][sel]])))}
    for target in ("likes", "views"):
        t = pair_tally(p, sel, target)
        if t is not None:
            out[f"pairs_{target}"] = {"pairs": int(t["n"].sum()), "accounts": int(len(t["n"])), "acc": float(t["c"].sum() / t["n"].sum()),
                                      "ci": ci(lambda i: t["c"][i].sum() / t["n"][i].sum(), len(t["n"])), "_t": t}
        lv = acct_level(p, sel, target, offset)
        if len(lv) >= 5:
            v = np.array(list(lv.values()))
            out[f"level_{target}"] = {"accounts": int(len(v)), "median_factor": float(10 ** np.median(v)), "ci": [10 ** x for x in ci(lambda i: np.median(v[i]), len(v))],
                                      "median_abs_factor": float(10 ** np.median(np.abs(v))), "within_1.5x": float((np.abs(v) <= np.log10(1.5)).mean()),
                                      "within_2x": float((np.abs(v) <= np.log10(2)).mean()), "_v": lv}
    return out


def diff(a: dict, b: dict):
    """Paired difference in pair accuracy, a minus b, accounts resampled."""
    ta, tb = a["_t"], b["_t"]
    assert (ta["acct"] == tb["acct"]).all()
    f = lambda i: ta["c"][i].sum() / ta["n"][i].sum() - tb["c"][i].sum() / tb["n"][i].sum()
    return {"diff": float(f(np.arange(len(ta["n"])))), "ci": ci(f, len(ta["n"]))}


bedm = info["bed_eval"].to_numpy()
small = (fol > 0) & (fol < 5000)
S = {"twitter100m full timelines (likes, to April 2023)": bedm & (src == "twitter100m"),
     "  under 1k followers": bedm & (src == "twitter100m") & (tier == "under_1k"), "  1k to 5k": bedm & (src == "twitter100m") & (tier == "1k_5k"),
     "  5k to 10k": bedm & (src == "twitter100m") & (tier == "5k_10k"), "  10k to 100k": bedm & (src == "twitter100m") & (tier == "10k_100k"),
     "  100k and over": bedm & (src == "twitter100m") & (tier == "100k_plus"),
     "2026 view corpus (likes and views)": bedm & (src == "nitter_2026_replies"),
     "brand accounts, full timelines": bedm & (src == "zenodo_brands"), "  brands no model has seen": bedm & (src == "zenodo_brands") & ~seen,
     "small accounts, under 5k followers (all sources)": bedm & small, "  small, 2026 view corpus": bedm & small & (src == "nitter_2026_replies"),
     "all full-timeline accounts (twitter100m and brands)": bedm & np.isin(src, ["twitter100m", "zenodo_brands"]),
     "whole test bed": bedm}
res: dict = {"slices": {k: int(v.sum()) for k, v in S.items()}, "models": {}}
corr: dict = {}
models = [(spec.split("=", 1)[0], spec.split("=", 1)[1].split(","), "") for spec in args.model]
models += [(f"{lab} as customer", dict((m[0], m[1]) for m in models)[lab], "_customer") for lab in args.customer]
for label, members, suffix in models:
    p = load(members, "bed", suffix)
    res["models"][label] = {k: summarize(p, m) for k, m in S.items()}
    # the correction: fitted on validation accounts only
    pv = load(members, "val", suffix)
    corr[label] = {}; off: dict = {}
    selv, selb = np.arange(len(pv["row"])), np.flatnonzero(bedm[p["row"]])
    lv, fv, fb = acct_level(pv, selv, "likes"), acct_feats(pv, selv), acct_feats(p, selb)
    # full timelines: error = a + b*log10(followers) + c*log10(own median likes), least squares over validation accounts
    ids = [u for u in lv if fv[u][2] == 3]
    X = np.array([[1.0, fv[u][0], fv[u][1]] for u in ids]); yv = np.array([lv[u] for u in ids])
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    corr[label]["full_timeline"] = {"form": "log10_error = a + b*log10(followers+1) + c*log10(median likes of visible history + 1)", "a": float(beta[0]), "b": float(beta[1]), "c": float(beta[2]),
                                    "validation_accounts": len(ids), "median_error_before": float(10 ** np.median(yv)), "median_abs_before": float(10 ** np.median(np.abs(yv))),
                                    "median_abs_after": float(10 ** np.median(np.abs(yv - X @ beta)))}
    ids2 = [u for u in lv if fv[u][2] == 2]
    c2 = float(np.median([lv[u] for u in ids2])) if len(ids2) >= 20 else 0.0
    corr[label]["search_sample"] = {"log10_offset": c2, "factor": 10 ** c2, "validation_accounts": len(ids2)}
    for u, (lf, om, rg) in fb.items():
        off[u] = float(beta[0] + beta[1] * lf + beta[2] * om) if rg == 3 else (c2 if rg == 2 else 0.0)
    res["models"][f"{label} + correction"] = {k: summarize(p, m, off) for k, m in S.items()}
    if not suffix and (Path(args.pred) / f"{members[0]}__test_aud.npz").exists():
        pt = load(members, "test_aud")
        res["models"][label]["existing audience test rows"] = summarize(pt, np.ones(len(author), bool))
base = res["models"][args.base]
for label, r in res["models"].items():
    for k, v in r.items():
        for tg in ("pairs_likes", "pairs_views"):
            if tg in v and label != args.base and k in base and tg in base[k] and not label.endswith("+ correction"):
                v[tg]["vs_base"] = diff(v[tg], base[k][tg])


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
    return o


Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(strip(res), open(args.out + ".json", "w"), indent=1)
json.dump(corr, open(args.out + "_level_correction.json", "w"), indent=1)
labels = [l for l in res["models"]]
plain = [l for l in labels if not l.endswith("+ correction")]
L = []
pct = lambda x: f"{100 * x:.1f}%"
for tg, title in (("pairs_likes", "Ordinary same-account pairs, likes"), ("pairs_views", "Ordinary same-account pairs, views")):
    L += [f"#### {title}", "", "| Slice | Accounts | Pairs | " + " | ".join(plain) + " |", "|---|---|---|" + "---|" * len(plain)]
    for k in list(S) + ["existing audience test rows"]:
        if k not in base or tg not in base[k]:
            continue
        cells = []
        for l in plain:
            v = res["models"][l].get(k, {}).get(tg)
            if v is None:
                cells.append("")
            else:
                d = v.get("vs_base")
                cells.append(f"{pct(v['acc'])}" + (f" ({100 * d['diff']:+.1f}, {100 * d['ci'][0]:+.1f} to {100 * d['ci'][1]:+.1f})" if d else f" ({pct(v['ci'][0])} to {pct(v['ci'][1])})"))
        L.append(f"| {k} | {base[k][tg]['accounts']:,} | {base[k][tg]['pairs']:,} | " + " | ".join(cells) + " |")
    L.append("")
for tg, title in (("level_likes", "Level of the forecast, likes: median account's predicted median over its actual median (1.00x is honest); in brackets the 95% interval; then the median absolute factor; then the share of accounts inside 1.5x"),
                  ("level_views", "Level of the forecast, views")):
    use = labels if tg == "level_likes" else plain
    L += [f"#### {title}", "", "| Slice | Accounts | " + " | ".join(use) + " |", "|---|---|" + "---|" * len(use)]
    for k in S:
        if tg not in base[k]:
            continue
        cells = []
        for l in use:
            v = res["models"][l].get(k, {}).get(tg)
            cells.append("" if v is None else f"{v['median_factor']:.2f}x ({v['ci'][0]:.2f} to {v['ci'][1]:.2f}); {v['median_abs_factor']:.2f}x; {pct(v['within_1.5x'])}")
        L.append(f"| {k} | {base[k][tg]['accounts']:,} | " + " | ".join(cells) + " |")
    L.append("")
L += ["#### Level correction fitted on validation accounts", "", "Full timelines: log10(error) = a + b*log10(followers+1) + c*log10(median likes of the visible history+1), subtracted from the forecast. Search sample: one offset.", "",
      "| Model | a | b | c | Validation accounts | Median absolute error before | after | Search-sample offset |", "|---|---|---|---|---|---|---|---|"]
for l, c in corr.items():
    f = c["full_timeline"]
    L.append(f"| {l} | {f['a']:+.3f} | {f['b']:+.3f} | {f['c']:+.3f} | {f['validation_accounts']} | {f['median_abs_before']:.2f}x | {f['median_abs_after']:.2f}x | {c['search_sample']['factor']:.2f}x |")
open(args.out + ".md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
