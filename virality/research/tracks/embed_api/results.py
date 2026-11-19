#!/usr/bin/env python3
"""The track's results: every arm next to the bge-small control on ordinary pairs (train / validation / test), the
3,530 same-topic benchmark pairs and the 850 launch pairs.

    python research/tracks/embed_api/results.py --arm label=run1,run2 [--arm ...] --control control --out data/api/results.json

A run name maps to data/api/runs/<run>/{metrics,splits}.json, data/api/bench_out/<run>/shard_*.parquet and
data/api/launch/<run>.parquet. Arms are score-level ensembles of their runs (two seeds). Benchmark confidence: a
logistic function of the score gap, scale cross-fitted over accounts (as compare_runs.py); launch confidence the same,
cross-fitted on the launch pairs. Intervals and paired differences resample accounts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "evidence"))
import compare_runs as C  # noqa: E402
from xviral.model import evaluate  # noqa: E402

GAPS = (1.25, 2.0, 3.0, 5.0)


def boot(auth, cols, f, n=400, seed=1):
    rng = np.random.default_rng(seed)
    ua, inv = np.unique(auth, return_inverse=True)
    S = [np.bincount(inv, weights=np.asarray(c, float), minlength=len(ua)) for c in cols]
    v = []
    for _ in range(n):
        s = rng.integers(0, len(ua), len(ua))
        with np.errstate(all="ignore"):
            v.append(f(*[x[s].sum() for x in S]))
    return [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]


def confidence(s, d, auth, seed=0):
    clear = np.abs(d) >= np.log(1.25)
    rng = np.random.default_rng(seed)
    ua = np.unique(auth)
    fold = dict(zip(ua, rng.integers(0, 5, len(ua))))
    fo = np.array([fold[a] for a in auth])
    conf = np.zeros(len(s))
    for k_ in range(5):
        k = evaluate.fit_pair_scale(s[(fo != k_) & clear], (d > 0)[(fo != k_) & clear])
        p = 1 / (1 + np.exp(-k * s[fo == k_]))
        conf[fo == k_] = np.maximum(p, 1 - p)
    return conf


def table(s, d, auth):
    conf, correct = confidence(s, d, auth), ((s * d) > 0).astype(float)
    out = {}
    for g in GAPS:
        m = np.abs(d) >= np.log(g)
        mc = m & (conf >= 0.6)
        one = np.ones(int(m.sum()))
        n40 = int(round(0.4 * m.sum()))
        top = np.zeros(int(m.sum()))
        top[np.argsort(-np.abs(s[m]), kind="stable")[:n40]] = 1
        out[f"{g}x"] = {"pairs": int(m.sum()), "accuracy": float(correct[m].mean()), "ci": boot(auth[m], [correct[m], one], lambda a, b: a / b),
                        "confident_share": float(mc.sum() / m.sum()), "confident_pairs": int(mc.sum()),
                        "confident_accuracy": float(correct[mc].mean()) if mc.sum() else None,
                        "surest_40pct_accuracy": float((correct[m] * top).sum() / max(top.sum(), 1))}
    return out, correct, conf


def versus(a, b, d, auth):
    out = {}
    for g in (1.25, 2.0):
        m = np.abs(d) >= np.log(g)
        ca, cb = a[1][m], b[1][m]
        ma, mb = (a[2][m] >= 0.6).astype(float), (b[2][m] >= 0.6).astype(float)
        one = np.ones(int(m.sum()))
        out[f"{g}x"] = {"pairs": int(m.sum()), "accuracy_gap": float(ca.mean() - cb.mean()), "accuracy_gap_ci": boot(auth[m], [ca, cb, one], lambda x, y, n: (x - y) / n),
                        "confident_share_gap": float(ma.mean() - mb.mean()), "confident_share_gap_ci": boot(auth[m], [ma, mb, one], lambda x, y, n: (x - y) / n),
                        "confident_accuracy_gap": float((ca * ma).sum() / max(ma.sum(), 1) - (cb * mb).sum() / max(mb.sum(), 1)),
                        "confident_accuracy_gap_ci": boot(auth[m], [ca * ma, ma, cb * mb, mb], lambda p, q, r, s: p / q - r / s)}
    return out


def fmt(t):
    return " | ".join(f"{g}: {100 * v['accuracy']:.1f}% of {v['pairs']}, conf {100 * v['confident_share']:.0f}% at {100 * (v['confident_accuracy'] or 0):.1f}%, top40 {100 * v['surest_40pct_accuracy']:.1f}%" for g, v in t.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--versus", action="append", default=[], help="a,b: also report a minus b (besides every arm minus the control)")
    ap.add_argument("--name", action="append", default=[], help="label=text for the tables")
    ap.add_argument("--root", default="data/api")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = {x.split("=")[0]: x.split("=")[1].split(",") for x in a.arm}
    res: dict = {"arms": arms, "ordinary": {}, "benchmark": {}, "benchmark_test_period": {}, "launch": {}, "versus_control": {}, "versus": {}}
    nm = dict(x.split("=", 1) for x in a.name)
    N = lambda k: nm.get(k, k)  # noqa: E731
    extra = [tuple(v.split(",")) for v in a.versus]
    TT: dict = {}

    # ordinary same-account pairs: train / validation / test per run
    print("== ordinary same-account pairs (audience accounts): pairwise accuracy on likes, train (700 fitted accounts) / validation (held-out accounts) / test (after the cutoff)")
    for label, runs in arms.items():
        rows = []
        for r in runs:
            f = f"{a.root}/runs/{r}/splits.json"
            if os.path.exists(f):
                rows.append(json.load(open(f)))
        res["ordinary"][label] = {"runs": dict(zip(runs, rows))}
        if rows:
            for s in ("train", "val", "test"):
                res["ordinary"][label][s] = {k: float(np.mean([x[s][k] for x in rows])) for k in ("pairwise_likes", "within_rho_likes", "nll_likes", "pairwise_views")} | {"pairs_likes": rows[0][s]["pairs_likes"]}
            print(f"{label:<26} " + " | ".join(f"{s} {100 * res['ordinary'][label][s]['pairwise_likes']:.2f}% rho {res['ordinary'][label][s]['within_rho_likes']:.3f} nll {res['ordinary'][label][s]['nll_likes']:.3f}"
                                               for s in ("train", "val", "test")) + "  seeds test: " + ", ".join(f"{100 * x['test']['pairwise_likes']:.2f}" for x in rows))

    # benchmark
    part = pl.read_parquet("data/tracks/pairs/benchmark_pairs.parquet").select(pl.min_horizontal("post_id", "post_id_b").alias("post_id"), pl.max_horizontal("post_id", "post_id_b").alias("post_id_b"), "part")
    J = None
    for label, runs in arms.items():
        fs = [C.load(f"{a.root}/bench_out/{r}/shard_*.parquet").rename({"s_likes": f"s_{r}"}) for r in runs]
        R = fs[0]
        for x in fs[1:]:
            R = R.join(x.select("post_id", "post_id_b", x.columns[-1]), on=["post_id", "post_id_b"])
        R = R.with_columns(pl.mean_horizontal([pl.col(f"s_{r}") for r in runs]).alias("S_" + label)).select("post_id", "post_id_b", "author_id", "likes", "likes_b", "S_" + label)
        J = R if J is None else J.join(R.select("post_id", "post_id_b", "S_" + label), on=["post_id", "post_id_b"])
    J = J.join(part, on=["post_id", "post_id_b"]).sort("post_id", "post_id_b")
    d = ((J["likes"].cast(pl.Float64) + 1).log() - (J["likes_b"].cast(pl.Float64) + 1).log()).to_numpy()
    auth = J["author_id"].to_numpy()
    for name, m in (("benchmark", np.ones(len(d), bool)), ("benchmark_test_period", (J["part"] == "test").to_numpy())):
        print(f"== {name}: {int(m.sum())} pairs scored by every arm")
        T = {label: table(J["S_" + label].to_numpy()[m], d[m], auth[m]) for label in arms}
        TT[name] = (T, d[m], auth[m])
        for label, t in T.items():
            res[name][label] = t[0]
            print(f"{label:<26} {fmt(t[0])}")
        for label in arms:
            if label != a.control:
                v = versus(T[label], T[a.control], d[m], auth[m])
                res["versus_control"].setdefault(label, {})[name] = v
                print(f"   {label} minus {a.control}: " + " | ".join(f"{g}: {100 * x['accuracy_gap']:+.1f} [{100 * x['accuracy_gap_ci'][0]:+.1f},{100 * x['accuracy_gap_ci'][1]:+.1f}], conf share {100 * x['confident_share_gap']:+.1f} "
                                                                    f"[{100 * x['confident_share_gap_ci'][0]:+.1f},{100 * x['confident_share_gap_ci'][1]:+.1f}], conf acc {100 * x['confident_accuracy_gap']:+.1f} "
                                                                    f"[{100 * x['confident_accuracy_gap_ci'][0]:+.1f},{100 * x['confident_accuracy_gap_ci'][1]:+.1f}]" for g, x in v.items()))

    # launch pairs, built exactly as the launch and replicate tracks build them
    Tl = pl.read_parquet("data/tracks/launch/launch_posts.parquet")
    S0 = pl.read_parquet("data/tracks/launch/launch_scores.parquet").select("post_id")
    P = Tl.join(S0, on="post_id")
    p = P.join(P, on="author_id", suffix="_b").filter(pl.col("post_id") < pl.col("post_id_b"))
    p = p.with_columns(((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log()).alias("d")).filter(pl.col("d").abs() >= np.log(1.25))
    p = p.with_columns(pl.int_range(pl.len()).shuffle(seed=1).over("author_id").alias("r")).filter(pl.col("r") < 30).select("post_id", "post_id_b", "author_id", "d")
    d, auth = p["d"].to_numpy(), p["author_id"].to_numpy()
    print(f"== launch pairs: {p.height} pairs, {p['author_id'].n_unique()} accounts")

    def member(r):
        s = pl.read_parquet(f"{a.root}/launch/{r}.parquet").with_columns((pl.col("pred_likes") + 1).log10().alias("q")).select("post_id", "q")
        j = p.join(s, on="post_id", how="left").join(s.rename({"post_id": "post_id_b", "q": "q_b"}), on="post_id_b", how="left")
        assert j["q"].null_count() == 0 and j["q_b"].null_count() == 0
        return (j["q"] - j["q_b"]).to_numpy()

    T = {}
    for label, runs in arms.items():
        if all(os.path.exists(f"{a.root}/launch/{r}.parquet") for r in runs):
            T[label] = table(np.mean([member(r) for r in runs], 0), d, auth)
            res["launch"][label] = T[label][0]
            print(f"{label:<26} {fmt(T[label][0])}")
    TT["launch"] = (T, d, auth)
    for label in T:
        if label != a.control and a.control in T:
            v = versus(T[label], T[a.control], d, auth)
            res["versus_control"].setdefault(label, {})["launch"] = v
            print(f"   {label} minus {a.control}: " + " | ".join(f"{g}: {100 * x['accuracy_gap']:+.1f} [{100 * x['accuracy_gap_ci'][0]:+.1f},{100 * x['accuracy_gap_ci'][1]:+.1f}]" for g, x in v.items()))
    for x, y in extra:
        for name, (T, d_, au_) in TT.items():
            if x in T and y in T:
                res["versus"].setdefault(f"{x} minus {y}", {})[name] = versus(T[x], T[y], d_, au_)
    json.dump(res, open(a.out, "w"), indent=1)

    # markdown
    pc = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    pm = lambda g, ci: f"{100 * g:+.1f} ({100 * ci[0]:+.1f} to {100 * ci[1]:+.1f})"  # noqa: E731
    L = ["| Model | Train | Validation | Test | Test, each seed | Rank correlation train / validation / test | Likes NLL train / validation / test | Views pairs, test |", "|---|---|---|---|---|---|---|---|"]
    for label in arms:
        o = res["ordinary"][label]
        if "test" not in o:
            continue
        L.append(f"| {N(label)} | {100 * o['train']['pairwise_likes']:.1f}% | {100 * o['val']['pairwise_likes']:.1f}% | **{100 * o['test']['pairwise_likes']:.1f}%** | "
                 + ", ".join(f"{100 * x['test']['pairwise_likes']:.1f}" for x in o["runs"].values()) + f" | {o['train']['within_rho_likes']:.3f} / {o['val']['within_rho_likes']:.3f} / {o['test']['within_rho_likes']:.3f} | "
                 f"{o['train']['nll_likes']:.3f} / {o['val']['nll_likes']:.3f} / {o['test']['nll_likes']:.3f} | {100 * o['test']['pairwise_views']:.1f}% |")
    for name, title in (("benchmark", "Same-topic benchmark, all 3,530 pairs"), ("benchmark_test_period", "Same-topic benchmark, test-period pairs only"), ("launch", "Launch pairs")):
        L += ["", f"{title}:", "", "| Model | Real gap at least | Pairs | Correct | Confident share (pairs) | Correct there | Surest 40% correct |", "|---|---|---|---|---|---|---|"]
        for label in arms:
            if label in res[name]:
                for g, v in res[name][label].items():
                    L.append(f"| {N(label)} | {g} | {v['pairs']:,} | {pc(v['accuracy'])} ({100 * v['ci'][0]:.1f} to {100 * v['ci'][1]:.1f}) | {100 * v['confident_share']:.0f}% ({v['confident_pairs']:,}) | {pc(v['confident_accuracy'] or 0)} | {pc(v['surest_40pct_accuracy'])} |")
    L += ["", "Paired differences:", "", "| A minus B | Test set | Real gap at least | Pairs | Correct, points | Confident share, points | Correct among confident, points |", "|---|---|---|---|---|---|---|"]
    allv = [(f"{N(k)} minus {N(a.control)}", v) for k, v in res["versus_control"].items()] + [(" minus ".join(N(z) for z in k.split(" minus ")), v) for k, v in res["versus"].items()]
    for title, v in allv:
        for name in ("benchmark", "benchmark_test_period", "launch"):
            for g, x in v.get(name, {}).items():
                L.append(f"| {title} | {name.replace('_', ' ')} | {g} | {x['pairs']:,} | {pm(x['accuracy_gap'], x['accuracy_gap_ci'])} | {pm(x['confident_share_gap'], x['confident_share_gap_ci'])} | {pm(x['confident_accuracy_gap'], x['confident_accuracy_gap_ci'])} |")
    open(a.out.replace(".json", ".md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
