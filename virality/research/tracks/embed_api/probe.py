#!/usr/bin/env python3
"""(Copied from the encoder track, unchanged apart from paths.) Step 1b of the encoder track: how well does each encoder's embedding difference predict which version did better?

    python research/tracks/embed_api/probe.py --dir data/probe --after 2026-04-23 --bench 'data/bench2/*/shard_*.parquet' \
        --set small=BAAI/bge-small-en-v1.5 --set base=BAAI/bge-base-en-v1.5 --set both=BAAI/bge-small-en-v1.5+BAAI/bge-base-en-v1.5 \
        --out probe.json

An antisymmetric logistic model (no intercept, each pair in both orders) on the standardised embedding
difference, trained on pre-cutoff pairs with a 25%+ like gap. The regularisation strength is chosen by 5-fold
cross-validation over pre-cutoff accounts; the test period is never used for fitting or selection.
Reported on the benchmark's test-period pairs: accuracy overall, by size of real gap, and by confidence band
(scale fitted on out-of-fold pre-cutoff scores). `+struct` adds pair_model.py's structure, media and link columns.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from xviral.features import text_structure as ts
from xviral.model import evaluate
from xviral.util.core import date_to_ms

GAPS = (1.25, 2.0, 3.0, 5.0)
BANDS = ((0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 1.01))
CS = (0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01)


def cluster_ci(auth, stat, n=400, seed=1):
    rng = np.random.default_rng(seed)
    ua, inv = np.unique(auth, return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(ua))]
    vals = [stat(np.concatenate([members[j] for j in rng.integers(0, len(ua), len(ua))])) for _ in range(n)]
    return [float(x) for x in np.nanpercentile(vals, [2.5, 97.5])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--set", action="append", required=True, help="name=encoder[+encoder...][@view]")
    ap.add_argument("--baseline", default=None, help="name of the set the others are compared with (paired, accounts resampled)")
    ap.add_argument("--no-struct", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pairs = pl.read_parquet(os.path.join(args.dir, "pairs.parquet"))
    X = pl.read_parquet(os.path.join(args.dir, "texts.parquet"))
    P = pl.read_parquet(os.path.join(args.dir, "posts.parquet"))
    assert (P["post_id"] == X["post_id"]).all()
    pos = {p: i for i, p in enumerate(X["post_id"].to_list())}
    ia = np.array([pos[x] for x in pairs["post_id"].to_list()])
    ib = np.array([pos[x] for x in pairs["post_id_b"].to_list()])
    y = np.log1p(pairs["likes"].to_numpy().astype(float)) - np.log1p(pairs["likes_b"].to_numpy().astype(float))
    auth = pairs["author_id"].to_numpy()
    cut = date_to_ms(args.after)
    test = pairs["created_at_ms"].to_numpy() >= cut
    B = pl.concat([pl.read_parquet(f, columns=["post_id", "post_id_b", "part"]) for f in glob.glob(args.bench)]).unique(["post_id", "post_id_b"])
    bk = {(min(a, b), max(a, b)): p for a, b, p in B.iter_rows()}
    part = np.array([bk.get((min(a, b), max(a, b)), "") for a, b in zip(pairs["post_id"].to_list(), pairs["post_id_b"].to_list())])
    bench_test = test & (part == "test")
    clear = np.abs(y) >= np.log(1.25)
    print(f"pairs {len(y)}: pre-cutoff {int((~test).sum())} ({int((~test & clear).sum())} clear), test-period {int(test.sum())}, "
          f"benchmark test {int(bench_test.sum())} ({int((bench_test & clear).sum())} clear), benchmark calibration {int((part == 'calibration').sum())}", flush=True)

    F = ts.text_structure(P)
    tcols = [c for c in F.columns if c.startswith("ts_") and F.schema[c].is_numeric()]
    F = F.select("post_id", *tcols).join(P.select(
        "post_id", (pl.col("media").list.len() > 0).cast(pl.Float64).alias("x_media"),
        pl.col("media").list.eval(pl.element().struct.field("type") == "video").list.any().fill_null(False).cast(pl.Float64).alias("x_video"),
        (pl.col("urls").list.len() > 0).cast(pl.Float64).alias("x_link"),
        pl.col("text").fill_null("").str.len_chars().log1p().alias("x_loglen")), on="post_id")
    cols = [c for c in F.columns if c != "post_id"]
    Fm = F.join(X.select("post_id"), on="post_id", how="right").select(cols).fill_null(0).to_numpy().astype(np.float64)
    S = Fm[ia] - Fm[ib]
    S = S / (S[~test].std(0) + 1e-9)

    rng = np.random.default_rng(0)
    ua = np.unique(auth[~test])
    fold_of = dict(zip(ua, rng.integers(0, 5, len(ua))))
    fold = np.array([fold_of.get(a, -1) for a in auth])
    tr_all = ~test & clear

    def emb(spec: str) -> np.ndarray:
        spec, _, view = spec.partition("@")
        parts = [np.load(os.path.join(args.dir, f"emb_{e.replace('/', '__')}_{view or 'rendered'}.npy")) for e in spec.split("+")]
        D = np.hstack([p[ia] - p[ib] for p in parts]).astype(np.float64)
        return D / (D[~test].std(0) + 1e-9)

    def fit(Z, tr, C):
        m = LogisticRegression(C=C, fit_intercept=False, max_iter=3000).fit(np.vstack([Z[tr], -Z[tr]]), np.concatenate([y[tr] > 0, y[tr] < 0]).astype(int))
        return m.coef_[0]

    def run(Z) -> dict:
        best, oof_best = None, None
        cv = {}
        for C in CS:
            s = np.zeros(len(y))
            for k in range(5):
                s[(fold == k) & ~test] = Z[(fold == k) & ~test] @ fit(Z, tr_all & (fold != k), C)
            acc = float(((s[tr_all] * y[tr_all]) > 0).mean())
            cv[str(C)] = acc
            if best is None or acc > cv[str(best)]:
                best, oof_best = C, s
        s = oof_best.copy()
        s[test] = Z[test] @ fit(Z, tr_all, best)
        k_ = evaluate.fit_pair_scale(s[tr_all], y[tr_all] > 0)
        te = bench_test & clear
        c = (s * y) > 0
        p = 1 / (1 + np.exp(-k_ * s))
        conf = np.maximum(p, 1 - p)
        out = {"C": best, "cv_accuracy_by_C": cv, "oof_accuracy_pre_cutoff": cv[str(best)], "dim": int(Z.shape[1]),
               "test_pairs": int(te.sum()), "test_accuracy": float(c[te].mean()), "test_accuracy_ci": cluster_ci(auth[te], lambda i: c[te][i].mean()),
               "by_gap": {}, "bands": {}}
        for g in GAPS:
            m = bench_test & (np.abs(y) >= np.log(g))
            mc = m & (conf >= 0.6)
            out["by_gap"][f"{g}x"] = {"pairs": int(m.sum()), "accuracy": float(c[m].mean()), "confident_share": float(mc.sum() / m.sum()),
                                      "confident_pairs": int(mc.sum()), "confident_accuracy": float(c[mc].mean()) if mc.sum() else None}
        for lo, hi in BANDS:
            m = te & (conf >= lo) & (conf < hi)
            out["bands"][f"{lo:.2f}-{min(hi, 1):.2f}"] = {"pairs": int(m.sum()), "accuracy": float(c[m].mean()) if m.sum() else None}
        out["_c"], out["_s"] = c, s
        return out

    res: dict = {"train_pairs": int(tr_all.sum()), "test_pairs": int((bench_test & clear).sum()), "sets": {}}
    te = bench_test & clear
    variants = [("emb", False)] if args.no_struct else [("emb", False), ("emb+struct", True)]
    if not args.no_struct:
        r = run(S)
        res["sets"]["struct_only"] = r
        print(f"{'struct_only':<34} dim {r['dim']:>4} C {r['C']:<6} oof {100 * r['oof_accuracy_pre_cutoff']:.1f}  test {100 * r['test_accuracy']:.1f}", flush=True)
    for spec in args.set:
        name, enc = spec.split("=", 1)
        E = emb(enc)
        for vn, with_s in variants:
            r = run(np.hstack([S, E]) if with_s else E)
            res["sets"][f"{name}/{vn}"] = r | {"encoder": enc}
            g = " ".join(f"{k}:{100 * v['accuracy']:.1f}(n={v['pairs']})" for k, v in r["by_gap"].items())
            b = " ".join(f"{k}:{100 * v['accuracy']:.1f}(n={v['pairs']})" if v["pairs"] else f"{k}:-" for k, v in r["bands"].items())
            print(f"{name + '/' + vn:<34} dim {r['dim']:>4} C {r['C']:<6} oof {100 * r['oof_accuracy_pre_cutoff']:.1f}  test {100 * r['test_accuracy']:.1f} "
                  f"[{100 * r['test_accuracy_ci'][0]:.1f},{100 * r['test_accuracy_ci'][1]:.1f}] | {g} | {b}", flush=True)
    if args.baseline:
        for k, r in res["sets"].items():
            vn = k.split("/")[-1] if "/" in k else None
            base = res["sets"].get(f"{args.baseline}/{vn}")
            if base is None or base is r:
                continue
            ca, cb = r["_c"][te].astype(float), base["_c"][te].astype(float)
            r["vs_baseline"] = {"gap": float(ca.mean() - cb.mean()), "ci": cluster_ci(auth[te], lambda i: ca[i].mean() - cb[i].mean())}
            print(f"{k:<34} vs {args.baseline}: {100 * r['vs_baseline']['gap']:+.1f} [{100 * r['vs_baseline']['ci'][0]:+.1f},{100 * r['vs_baseline']['ci'][1]:+.1f}]")
    scores = pl.DataFrame({"post_id": pairs["post_id"], "post_id_b": pairs["post_id_b"], "author_id": auth, "test": test, "bench_part": part, "y": y,
                           **{"s_" + k: r["_s"] for k, r in res["sets"].items()}})
    scores.write_parquet(args.out.replace(".json", "_scores.parquet"))
    for r in res["sets"].values():
        r.pop("_c"), r.pop("_s")
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
