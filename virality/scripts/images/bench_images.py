#!/usr/bin/env python3
"""Benchmarks restricted to pairs where an image is involved.

    python scripts/images/bench_images.py select     # writes data/images/bench_pairs_img.parquet, data/images/launch_posts_img.parquet
    python scripts/images/bench_images.py report --tac name=path.parquet ... --launch name=path.parquet ...   # writes data/images/bench_images.json

Same-topic benchmark (tracks/pairs/benchmark_pairs.parquet): pairs where at least one side has a fetched image, test
part, likes differing by 1.25x or more; by size of real gap and for confident calls (scale 1.5156, as in the launch
track). Launch pairs: the launch track's design (same account, both launch-type posts on or after 2026-04-23, likes
1.25x apart, at most 30 pairs per account) restricted to pairs where at least one side has a fetched image.
Paired differences between models with accounts resampled.
"""
import json, sys, argparse
import numpy as np, polars as pl
K = 1.5156
ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--tac", action="append", default=[]); ap.add_argument("--launch", action="append", default=[])
a = ap.parse_args()
img = set(np.load("data/images/aux_img.npz")["post_id"].tolist())
if a.cmd == "select":
    B = pl.read_parquet("data/tracks/pairs/benchmark_pairs.parquet")
    B = B.filter(pl.col("post_id").is_in(list(img)) | pl.col("post_id_b").is_in(list(img)))
    print("benchmark pairs with an image:", B.group_by("part").len().to_dicts())
    L = pl.read_parquet("data/tracks/launch/launch_scores.parquet").select("post_id", "author_id")
    acc = L.filter(pl.col("post_id").is_in(list(img)))["author_id"].unique()
    S = L.filter(pl.col("author_id").is_in(acc.implode()))
    S.write_parquet("data/images/launch_posts_img.parquet"); print("launch posts to score (accounts with an image launch post):", S.height, S["author_id"].n_unique()); sys.exit()
rng = np.random.default_rng(0)
def acc(c, au):
    if len(c) < 20: return {"pairs": int(len(c)), "too_thin": True}
    u, inv = np.unique(au, return_inverse=True); s = np.bincount(inv, c); n = np.bincount(inv); idx = rng.integers(0, len(u), (2000, len(u))); b = s[idx].sum(1) / n[idx].sum(1)
    return {"pairs": int(len(c)), "accounts": int(len(u)), "accuracy": round(float(c.mean()), 4), "lo": round(float(np.quantile(b, .025)), 4), "hi": round(float(np.quantile(b, .975)), 4)}
def pdiff(c1, c2, au):
    if len(c1) < 20: return None
    u, inv = np.unique(au, return_inverse=True); s1 = np.bincount(inv, c1); s2 = np.bincount(inv, c2); n = np.bincount(inv); idx = rng.integers(0, len(u), (2000, len(u))); b = (s1[idx].sum(1) - s2[idx].sum(1)) / n[idx].sum(1)
    return {"diff": round(float(c1.mean() - c2.mean()), 4), "lo": round(float(np.quantile(b, .025)), 4), "hi": round(float(np.quantile(b, .975)), 4)}
def slices(P, models):
    d = P["d"].to_numpy(); au = P["author_id"].to_numpy(); res = {}
    sl = {"all_1.25x": np.abs(d) >= np.log(1.25), "gap_2x": np.abs(d) >= np.log(2), "gap_3x": np.abs(d) >= np.log(3), "gap_5x": np.abs(d) >= np.log(5),
          "both_sides_image": (np.abs(d) >= np.log(1.25)) & P["img_a"].to_numpy() & P["img_b"].to_numpy(), "one_side_image": (np.abs(d) >= np.log(1.25)) & (P["img_a"].to_numpy() ^ P["img_b"].to_numpy())}
    for nm, m in sl.items():
        res[nm] = {}
        for mo in models:
            s = P[f"s_{mo}"].to_numpy(); c = ((s * d) > 0).astype(float); conf = 1 / (1 + np.exp(-K * np.abs(s)))
            res[nm][mo] = acc(c[m], au[m]) | {"confident_0.6": acc(c[m & (conf >= 0.6)], au[m & (conf >= 0.6)]) | {"share": round(float((conf[m] >= 0.6).mean()), 4) if m.sum() else None}}
        if "img" in models and "ctl" in models:
            c1 = ((P["s_img"].to_numpy() * d) > 0).astype(float); c2 = ((P["s_ctl"].to_numpy() * d) > 0).astype(float); res[nm]["img_minus_ctl"] = pdiff(c1[m], c2[m], au[m])
    return res
out = {}
if a.tac:
    J = None
    for spec in a.tac:
        nm, path = spec.split("=", 1); R = pl.read_parquet(path).filter(pl.col("part") == "test").select("post_id", "post_id_b", "author_id", "likes", "likes_b", pl.col("s_likes").alias(f"s_{nm}"))
        J = R if J is None else J.join(R.select("post_id", "post_id_b", f"s_{nm}"), on=["post_id", "post_id_b"])
    J = J.with_columns(((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log()).alias("d"), pl.col("post_id").is_in(list(img)).alias("img_a"), pl.col("post_id_b").is_in(list(img)).alias("img_b"))
    out["same_topic_benchmark_pairs_with_an_image"] = slices(J, [s.split("=")[0] for s in a.tac])
if a.launch:
    T = pl.read_parquet("data/launch/table.parquet").select("post_id", "created_at_ms", "likes")
    P = None
    for spec in a.launch:
        nm, path = spec.split("=", 1); S = pl.read_parquet(path).select("post_id", "author_id", pl.col("pred_likes").alias(f"p_{nm}"))
        P = S if P is None else P.join(S.select("post_id", f"p_{nm}"), on="post_id")
    P = P.join(T, on="post_id")
    p = P.join(P, on="author_id", suffix="_b").filter(pl.col("post_id") < pl.col("post_id_b"))
    p = p.with_columns(((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log()).alias("d")).filter(pl.col("d").abs() >= np.log(1.25))
    p = p.with_columns(pl.col("post_id").is_in(list(img)).alias("img_a"), pl.col("post_id_b").is_in(list(img)).alias("img_b")).filter(pl.col("img_a") | pl.col("img_b"))
    p = p.with_columns(pl.int_range(pl.len()).shuffle(seed=1).over("author_id").alias("r")).filter(pl.col("r") < 30)
    p = p.with_columns([((pl.col(f"p_{nm}") + 1).log10() - (pl.col(f"p_{nm}_b") + 1).log10()).alias(f"s_{nm}") for nm in [s.split("=")[0] for s in a.launch]])
    out["launch_pairs_with_an_image"] = slices(p, [s.split("=")[0] for s in a.launch])
json.dump(out, open("data/images/bench_images.json", "w"), indent=1)
for k, v in out.items():
    print("==", k)
    for sl, d in v.items(): print(f"  {sl:<18}", {m: (x.get("accuracy"), x.get("pairs"), x.get("confident_0.6", {}).get("accuracy"), x.get("confident_0.6", {}).get("share")) if isinstance(x, dict) and "diff" not in x else x for m, x in d.items()})
