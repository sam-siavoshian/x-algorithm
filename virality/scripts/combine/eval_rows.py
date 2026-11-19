#!/usr/bin/env python3
"""Ordinary same-account pairs and forecast level from saved per-row predictions (scripts/combine/predict_rows.py).

    python scripts/combine/eval_rows.py --model "Label=table_dir:pred_dir:run1,run2" [--model ...] --base Label --out data/combine/eval/rows

Pairs: two posts of one account within 7 days whose outcomes differ by 0.1 in log10 (the definition of every earlier
track), ranked by the ensemble mean of the members' predicted log10 mean. Three slices of the same pairs: all; real gap
2x or more; both posts with 3 or more likes (for view pairs: both with 100 or more views). Sets: train_aud, val_aud,
val_broad, test_aud, bed (and bed, val_broad as a customer when the model has the source inputs).
Level (bed and val_broad): per account, log10(median predicted) - log10(median actual) over its scored posts (10 or more);
the median account's factor, the median absolute factor, the share of accounts inside 1.5x.
Intervals resample accounts (1,000 draws); differences to --base are paired on accounts.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, polars as pl
from xviral.model import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--model", action="append", required=True); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
models = {}
for spec in a.model:
    label, rest = spec.split("=", 1)
    tdir, pdir, runs = rest.split(":")
    models[label] = (tdir, pdir, runs.split(","))
T0 = Path(next(iter(models.values()))[0])
info = pl.read_parquet(T0 / "row_info.parquet")
author, created = np.load(T0 / "author_idx.npy"), np.load(T0 / "created_s.npy")
src = info["expand_src"].fill_null("orig").to_numpy(); tier = info["g_tier"].fill_null("").to_numpy(); fol = info["followers"].to_numpy()
rng = np.random.default_rng(0)
_cache: dict = {}


def load(label: str, sname: str, suffix: str = "") -> dict | None:
    key = (label, sname, suffix)
    if key not in _cache:
        _, pdir, runs = models[label]
        fs = [Path(pdir) / f"{r}{suffix}__{sname}.npz" for r in runs]
        if not all(f.exists() for f in fs):
            _cache[key] = None
        else:
            ps = [np.load(f) for f in fs]
            assert all((p["row"] == ps[0]["row"]).all() for p in ps)
            out = {k: ps[0][k] for k in ps[0].files}
            for k in ("likes_mean", "likes_q50", "views_mean", "views_q50"):
                out[k] = np.mean([p[k] for p in ps], axis=0)
            out["nll_likes"] = np.mean([p["nll_likes"] for p in ps], axis=0)       # mean of members' losses (not the mixture's)
            _cache[key] = out
    return _cache[key]


_pairs: dict = {}


def pairs_of(rows, y, target):
    key = (target, len(rows), int(rows[0]), int(rows[-1]))
    if key not in _pairs:
        i, j = evaluate.same_author_pairs(author[rows], created[rows], y, 7 * 86400, 0.1)
        floor = np.log10(4.0) if target == "likes" else np.log10(101.0)
        _pairs[key] = (i, j, np.abs(y[i] - y[j]) >= np.log10(2.0), (y[i] >= floor) & (y[j] >= floor))
    return _pairs[key]


def tally(p, sel, target):
    """Per account: pairs and correct calls, for the three slices."""
    rows = p["row"][sel]; y, s = p[f"y_{target}"][sel], p[f"{target}_mean"][sel]
    if len(rows) < 2 or np.isnan(y).all():
        return None
    i, j, m2, m3 = pairs_of(rows, y, target)
    if len(i) < 50:
        return None
    agree = ((s[i] > s[j]) == (y[i] > y[j])).astype(float); agree[s[i] == s[j]] = 0.5
    ua, inv = np.unique(author[rows][i], return_inverse=True)
    out = {"acct": ua}
    for nm, m in (("all", np.ones(len(i), bool)), ("2x", m2), ("3plus", m3)):
        out[nm] = (np.bincount(inv, weights=m.astype(float), minlength=len(ua)), np.bincount(inv, weights=agree * m, minlength=len(ua)))
    return out


def acc_ci(n, c, draws=1000):
    if n.sum() < 50:
        return None
    b = rng.integers(0, len(n), (draws, len(n)))
    v = c[b].sum(1) / np.maximum(n[b].sum(1), 1)
    return {"acc": float(c.sum() / n.sum()), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)), "pairs": int(n.sum()), "accounts": int((n > 0).sum())}


def diff_ci(n, c, c0, draws=1000):
    b = rng.integers(0, len(n), (draws, len(n)))
    v = (c[b].sum(1) - c0[b].sum(1)) / np.maximum(n[b].sum(1), 1)
    return {"diff": float((c.sum() - c0.sum()) / n.sum()), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}


def level(p, sel, target="likes"):
    rows = p["row"][sel]; y, q = p[f"y_{target}"][sel], p[f"{target}_q50"][sel]
    au = author[rows]; ok = ~np.isnan(y)
    e = []
    for u in np.unique(au[ok]):
        m = ok & (au == u)
        if m.sum() >= 10:
            e.append(float(np.median(q[m]) - np.median(y[m])))
    e = np.array(e)
    if len(e) < 5:
        return None
    b = np.median(e[rng.integers(0, len(e), (1000, len(e)))], axis=1)
    return {"accounts": int(len(e)), "median_factor": float(10 ** np.median(e)), "lo": float(10 ** np.percentile(b, 2.5)), "hi": float(10 ** np.percentile(b, 97.5)),
            "median_abs_factor": float(10 ** np.median(np.abs(e))), "inside_1_5x": float((np.abs(e) <= np.log10(1.5)).mean())}


def slices(sname, rows):
    s = {"all": np.ones(len(rows), bool)}
    if sname == "bed":
        sr, tr_, fo = src[rows], tier[rows], fol[rows]
        s.update({"twitter100m full timelines": sr == "twitter100m", "2026 view corpus": sr == "nitter_2026_replies", "brand accounts": sr == "zenodo_brands",
                  "full timelines (twitter100m and brands)": np.isin(sr, ["twitter100m", "zenodo_brands"]),
                  "under 1k followers (twitter100m)": (sr == "twitter100m") & (tr_ == "under_1k"), "100k and over (twitter100m)": (sr == "twitter100m") & (tr_ == "100k_plus"),
                  "small accounts under 5k": np.isin(tr_, ["under_1k", "1k_5k"])})
    return s


res: dict = {"models": {k: v[2] for k, v in models.items()}, "pairs": {}, "level": {}, "nll": {}}
for sname, suffix in (("train_aud", ""), ("val_aud", ""), ("val_broad", ""), ("test_aud", ""), ("bed", ""), ("bed", "_customer"), ("val_broad", "_customer")):
    tag = sname + suffix
    P = {lab: (load(lab, sname, suffix) or (load(lab, sname) if suffix else None)) for lab in models}       # a model without the source inputs is its own customer mode
    P = {k: v for k, v in P.items() if v is not None}
    if not P:
        continue
    rows = next(iter(P.values()))["row"]
    assert all((p["row"] == rows).all() for p in P.values()), tag
    for sl, sel in slices(sname, rows).items():
        if sel.sum() < 100:
            continue
        for target in ("likes", "views"):
            tl = {lab: tally(p, sel, target) for lab, p in P.items()}
            if any(v is None for v in tl.values()):
                continue
            for nm in ("all", "2x", "3plus"):
                rec = {}
                for lab, t in tl.items():
                    r = acc_ci(*t[nm])
                    if r is None:
                        continue
                    if lab != a.base and a.base in tl:
                        r["vs_base"] = diff_ci(t[nm][0], t[nm][1], tl[a.base][nm][1])
                    rec[lab] = r
                res["pairs"][f"{tag}|{sl}|{target}|{nm}"] = rec
        if sname in ("bed", "val_broad"):
            res["level"][f"{tag}|{sl}"] = {lab: level(p, sel) for lab, p in P.items()}
    res["nll"][tag] = {lab: float(np.nanmean(p["nll_likes"])) for lab, p in P.items()}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out + ".json").write_text(json.dumps(res, indent=1))
L = list(models)
md = ["| Set | Slice | Outcome | Pairs | " + " | ".join(L) + " |", "|---|---|---|---|" + "---|" * len(L)]
for key, rec in res["pairs"].items():
    tag, sl, target, nm = key.split("|")
    cells = []
    for lab in L:
        r = rec.get(lab)
        cells.append("" if r is None else f"{100 * r['acc']:.1f}%" + (f" ({100 * r['vs_base']['diff']:+.1f}, {100 * r['vs_base']['lo']:+.1f} to {100 * r['vs_base']['hi']:+.1f})" if "vs_base" in r else f" ({100 * r['lo']:.1f} to {100 * r['hi']:.1f})"))
    n = next(iter(rec.values()))["pairs"] if rec else 0
    md.append(f"| {tag} | {sl} | {target} | {nm} | {n:,} | " + " | ".join(cells) + " |")
md += ["", "| Set | Slice | " + " | ".join(L) + " |", "|---|---|" + "---|" * len(L)]
for key, rec in res["level"].items():
    tag, sl = key.split("|")
    md.append(f"| {tag} | {sl} | " + " | ".join("" if rec.get(lab) is None else f"{rec[lab]['median_factor']:.2f}x ({rec[lab]['lo']:.2f} to {rec[lab]['hi']:.2f}); {rec[lab]['median_abs_factor']:.2f}x; {100 * rec[lab]['inside_1_5x']:.0f}% of {rec[lab]['accounts']}" for lab in L) + " |")
Path(a.out + ".md").write_text("\n".join(md) + "\n")
print("\n".join(md))
