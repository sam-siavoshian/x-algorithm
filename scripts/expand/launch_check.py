#!/usr/bin/env python3
"""The 850 launch pairs and the calibration of the predicted level of launch posts, for any scored model.

    python scripts/expand/launch_check.py --score name=scores.parquet [--score ...] [--group label=a,b] [--bench name=glob_of_benchmark_shards]
                                          [--versus a,b] --out data/expand/eval/launch.json

Pairs are built exactly as scripts/launch/model_check.py builds them (same account, both launch-type, both from
2026-04-23 on, likes 1.25x or more apart, at most 30 per account, same shuffle seed). A group averages its members'
log10 predictions. Confidence is sigmoid(K * |difference in log10 predicted likes|) with K = 1.5156, the scale the launch
track used (so "confident" matches its report); with --bench the same cuts are also given under a scale fitted on the
model's own benchmark calibration pairs (accounts no model trained on, before the cutoff), under "own_scale".
Level: predicted and actual likes over the account's trailing median of 30 posts (launch track's `med_likes`).
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import polars as pl

from xviral.model import evaluate

rng = np.random.default_rng(0)


def acc(q: pl.DataFrame, col: str = "s") -> dict:
    q = q.filter(pl.col(col) != 0)
    if q.height < 20:
        return {"pairs": q.height, "too_thin": True}
    c = ((q[col].to_numpy() * q["d"].to_numpy()) > 0).astype(float)
    uu, iv = np.unique(q["author_id"].to_numpy(), return_inverse=True)
    Sx, C = np.bincount(iv, weights=c), np.bincount(iv).astype(float)
    idx = np.random.default_rng(0).integers(0, len(uu), (2000, len(uu)))
    b = Sx[idx].sum(1) / C[idx].sum(1)
    return {"pairs": int(len(c)), "accounts": int(len(uu)), "accuracy": round(float(c.mean()), 4), "lo": round(float(np.quantile(b, .025)), 4), "hi": round(float(np.quantile(b, .975)), 4)}


def level(z: pl.DataFrame) -> dict:
    if z.height < 20:
        return {"posts": z.height, "too_thin": True}
    a = ((z["likes"] + 1) / (z["med_likes"] + 1)).log(10).to_numpy()
    p = ((z["pred_likes"] + 1) / (z["med_likes"] + 1)).log(10).to_numpy()
    err = np.log10(z["pred_likes"].to_numpy() + 1) - np.log10(z["likes"].to_numpy() + 1)
    uu, iv = np.unique(z["author_id"].to_numpy(), return_inverse=True)
    boot = []
    r = np.random.default_rng(0)
    members = [np.flatnonzero(iv == j) for j in range(len(uu))]
    for _ in range(500):
        i = np.concatenate([members[j] for j in r.integers(0, len(uu), len(uu))])
        boot.append(10 ** np.median(p[i]))
    out = {"posts": z.height, "accounts": int(len(uu)), "actual_median_ratio": round(float(10 ** np.median(a)), 3), "predicted_median_ratio": round(float(10 ** np.median(p)), 3),
           "predicted_median_ratio_ci": [round(float(np.quantile(boot, .025)), 3), round(float(np.quantile(boot, .975)), 3)],
           "median_predicted_over_actual": round(float(10 ** np.median(err)), 3),
           "actual_beat_own_median": round(float((z["likes"] > z["med_likes"]).mean()), 4), "predicted_beat_own_median": round(float((z["pred_likes"] > z["med_likes"]).mean()), 4),
           "spearman_pred_vs_actual_lift": round(float(evaluate.spearman(p, a)), 3), "mean_abs_log10_error": round(float(np.mean(np.abs(err))), 4)}
    if "pred_likes_q10" in z.columns and z["pred_likes_q10"].null_count() == 0:
        out["actual_inside_q10_q90"] = round(float(((z["likes"] >= z["pred_likes_q10"]) & (z["likes"] <= z["pred_likes_q90"])).mean()), 4)
        out["actual_above_q90"] = round(float((z["likes"] > z["pred_likes_q90"]).mean()), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="append", required=True)
    ap.add_argument("--group", action="append", default=[])
    ap.add_argument("--bench", action="append", default=[])
    ap.add_argument("--versus", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    T = pl.read_parquet("data/launch/table.parquet")
    S = {}
    for s in args.score:
        n, p = s.split("=", 1)
        S[n] = pl.read_parquet(p).select("post_id", "pred_likes", *[c for c in ("pred_likes_q10", "pred_likes_q90") if c in pl.read_parquet_schema(p)])
    for g in args.group:
        label, mem = g.split("=", 1)
        fr = [S[m].select("post_id", (pl.col("pred_likes") + 1).log(10).alias(f"l{i}")) for i, m in enumerate(mem.split(","))]
        J = fr[0]
        for f in fr[1:]:
            J = J.join(f, on="post_id")
        S[label] = J.select("post_id", (10 ** pl.mean_horizontal(pl.col("^l\\d+$")) - 1).alias("pred_likes"))
    # Confidence scale from each model's benchmark calibration pairs.
    K = {}
    parts = pl.read_parquet("data/pairs/benchmark_pairs.parquet").select("part", pl.min_horizontal("post_id", "post_id_b").alias("post_id"), pl.max_horizontal("post_id", "post_id_b").alias("post_id_b"))
    for b in args.bench:
        n, spec = b.split("=", 1)
        fr = []
        for i, g in enumerate(spec.split(",")):
            R = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(g))])
            flip = pl.col("post_id") > pl.col("post_id_b")
            fr.append(R.select(pl.min_horizontal("post_id", "post_id_b").alias("post_id"), pl.max_horizontal("post_id", "post_id_b").alias("post_id_b"),
                               pl.when(flip).then("likes_b").otherwise("likes").alias("likes"), pl.when(flip).then("likes").otherwise("likes_b").alias("likes_b"),
                               pl.when(flip).then(-pl.col("s_likes")).otherwise("s_likes").alias(f"s{i}")).unique(["post_id", "post_id_b"]))
        J = fr[0]
        for f in fr[1:]:
            J = J.join(f.select("post_id", "post_id_b", pl.col("^s\\d+$")), on=["post_id", "post_id_b"])
        J = J.join(parts, on=["post_id", "post_id_b"]).filter(pl.col("part") == "calibration").with_columns(pl.mean_horizontal(pl.col("^s\\d+$")).alias("s"))
        d = np.log((J["likes"].to_numpy() + 1.0) / (J["likes_b"].to_numpy() + 1.0))
        m = np.abs(d) >= np.log(1.25)
        K[n] = evaluate.fit_pair_scale(J["s"].to_numpy()[m], (d > 0)[m])
    res: dict = {"models": {}, "versus": {}}
    keep = {}
    for n, sc in S.items():
        P = T.join(sc, on="post_id")
        p = P.join(P, on="author_id", suffix="_b").filter(pl.col("post_id") < pl.col("post_id_b"))
        p = p.with_columns(((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log()).alias("d"), (pl.col("pred_likes") + 1).log10().sub((pl.col("pred_likes_b") + 1).log10()).alias("s"),
                           ((pl.col("created_at_ms") - pl.col("created_at_ms_b")).abs() / 86_400_000).alias("gap_d"))
        p = p.filter(pl.col("d").abs() >= np.log(1.25))
        p = p.with_columns(pl.int_range(pl.len()).shuffle(seed=1).over("author_id").alias("r")).filter(pl.col("r") < 30)
        k = 1.5156                      # the launch track's scale, so "confident" means what it means in its report
        conf = 1 / (1 + np.exp(-k * np.abs(p["s"].to_numpy())))
        p = p.with_columns(pl.Series("conf", conf))
        if n in K:
            p = p.with_columns(pl.Series("conf_own", 1 / (1 + np.exp(-K[n] * np.abs(p["s"].to_numpy())))))
        big = pl.col("d").abs() >= np.log(2)
        small = pl.col("followers_max") < 10_000
        o = {"confidence_scale": round(float(k), 4), "scale_from": "the launch track's scale (baseline ensemble on its calibration pairs)",
             "all": acc(p), "gap_2x": acc(p.filter(big)), "gap_3x": acc(p.filter(pl.col("d").abs() >= np.log(3))),
             "company_accounts": acc(p.filter(pl.col("akind") == "company")), "founder_accounts": acc(p.filter(pl.col("akind") == "founder")),
             "company_or_founder": acc(p.filter(pl.col("akind").is_in(["company", "founder"]))), "other_accounts": acc(p.filter(pl.col("akind") == "other")),
             "accounts_under_10k_followers": acc(p.filter(small)),
             "confident_0.6_plus": acc(p.filter(pl.col("conf") >= 0.6)) | {"share_of_pairs": round(float((conf >= 0.6).mean()), 4)},
             "confident_0.6_plus_gap_2x": acc(p.filter((pl.col("conf") >= 0.6) & big)) | {"share_of_2x_pairs": round(float(p.filter((pl.col("conf") >= 0.6) & big).height / max(p.filter(big).height, 1)), 4)}}
        if n in K:
            own = pl.col("conf_own") >= 0.6
            o["own_scale"] = {"scale": round(float(K[n]), 4), "from": "this model's benchmark calibration pairs",
                              "confident_0.6_plus": acc(p.filter(own)) | {"share_of_pairs": round(float(p.filter(own).height / p.height), 4)},
                              "confident_0.6_plus_gap_2x": acc(p.filter(own & big)) | {"share_of_2x_pairs": round(float(p.filter(own & big).height / max(p.filter(big).height, 1)), 4)}}
        z = P.filter(pl.col("med_likes").is_not_null())
        o["level"] = {"all": level(z), "company": level(z.filter(pl.col("akind") == "company")), "founder": level(z.filter(pl.col("akind") == "founder")),
                      "company_or_founder": level(z.filter(pl.col("akind").is_in(["company", "founder"]))), "under_10k_followers": level(z.filter(small)),
                      "posts_without_trailing_median": P.height - z.height}
        res["models"][n] = o
        keep[n] = p.select("post_id", "post_id_b", "author_id", "d", "s", "conf")
    for v in args.versus:
        a, b = v.split(",")
        J = keep[a].join(keep[b].select("post_id", "post_id_b", pl.col("s").alias("s_b")), on=["post_id", "post_id_b"])
        ca, cb = ((J["s"].to_numpy() * J["d"].to_numpy()) > 0).astype(float), ((J["s_b"].to_numpy() * J["d"].to_numpy()) > 0).astype(float)
        uu, iv = np.unique(J["author_id"].to_numpy(), return_inverse=True)
        Sd, C = np.bincount(iv, weights=ca - cb), np.bincount(iv).astype(float)
        idx = np.random.default_rng(0).integers(0, len(uu), (2000, len(uu)))
        bt = Sd[idx].sum(1) / C[idx].sum(1)
        res["versus"][v] = {"pairs": J.height, "accounts": int(len(uu)), "gap": round(float((ca - cb).mean()), 4), "lo": round(float(np.quantile(bt, .025)), 4), "hi": round(float(np.quantile(bt, .975)), 4)}
    json.dump(res, open(args.out, "w"), indent=1)
    for n, o in res["models"].items():
        L = o["level"]["all"]
        print(f"{n}: all {o['all']}  2x {o['gap_2x'].get('accuracy')} ({o['gap_2x'].get('pairs')})  confident {o['confident_0.6_plus'].get('accuracy')} share {o['confident_0.6_plus'].get('share_of_pairs')}"
              f"  level pred {L.get('predicted_median_ratio')} actual {L.get('actual_median_ratio')} K {o['confidence_scale']}")
    for v, r in res["versus"].items():
        print(v, r)


if __name__ == "__main__":
    main()
