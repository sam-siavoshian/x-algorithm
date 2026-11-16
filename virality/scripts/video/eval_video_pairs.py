#!/usr/bin/env python3
"""Pair accuracy of the arms of train_video.py on video-involved pairs, table path, likes.

    python scripts/video/eval_video_pairs.py baseline=data/video/runs/baseline control=data/video/runs/control_s1,data/video/runs/control_s2 thumb=...

Several run folders after one name are averaged (an ensemble of seeds). Pair sets, all same account, both posts on or
after the cutoff (test split), real like counts differing by 1.25x or more:
  video_video        both posts are video posts with poster-frame features, at most 365 days apart, at most 50 per account
  bench_video        pairs of the 3,530 same-topic benchmark with a video post on either side and both posts in the table
  launch_video       pairs of test-period launch posts by one account (the launch benchmark's construction) with a video
                     post with features on either side and both posts in the table
Each: accuracy overall and at a 2x gap, with an interval from resampling accounts, and the difference to `control` on
the same pairs (paired bootstrap over accounts). Writes data/video/model_pairs.json.
"""
import json, sys, numpy as np, polars as pl
rng = np.random.default_rng(0)
arms = {}
for a in sys.argv[1:]:
    k, v = a.split("="); P = [pl.read_parquet(f"{d}/pred.parquet") for d in v.split(",")]
    J = P[0].select("post_id", "y_likes", pl.col("likes_mean").alias("s0"))
    for i, p in enumerate(P[1:], 1): J = J.join(p.select("post_id", pl.col("likes_mean").alias(f"s{i}")), on="post_id")
    arms[k] = J.select("post_id", pl.mean_horizontal(pl.col(r"^s\d+$")).alias("s"))
posts = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id", "author_id", "created_at_ms", "likes").collect()
feat = set(pl.read_parquet("data/video/labels.parquet")["post_id"].to_list())
first = next(iter(arms.values()))
base = first.select("post_id").join(posts, on="post_id").with_columns(pl.col("post_id").is_in(list(feat)).alias("v"))
CUT = 1776902400000
def pairs_within(df, cap=50):
    p = df.join(df, on="author_id", suffix="_b").filter((pl.col("post_id") < pl.col("post_id_b")) & ((pl.col("created_at_ms") - pl.col("created_at_ms_b")).abs() <= 365 * 86_400_000))
    return p.with_columns(pl.int_range(pl.len()).shuffle(seed=1).over("author_id").alias("r")).filter(pl.col("r") < cap).select("author_id", "post_id", "post_id_b", "likes", "likes_b")
vv = pairs_within(base.filter(pl.col("v") & (pl.col("created_at_ms") >= CUT)))
B = pl.read_parquet("data/tracks/pairs/benchmark_pairs.parquet")
allv = set(pl.scan_parquet("data/work/filtered/posts.parquet").filter(pl.col("media").list.eval(pl.element().struct.field("type").is_in(["video", "gif"])).list.any().fill_null(False)).select("post_id").collect()["post_id"].to_list())
inpred = set(first["post_id"].to_list())
bv = B.filter(pl.col("post_id").is_in(list(inpred)) & pl.col("post_id_b").is_in(list(inpred)) & (pl.col("post_id").is_in(list(allv)) | pl.col("post_id_b").is_in(list(allv)))).select("author_id", "post_id", "post_id_b", "likes", "likes_b", "part")
LS = pl.read_parquet("data/tracks/launch/launch_scores.parquet").select("post_id").join(base, on="post_id")
lp = pairs_within(LS, cap=10_000)
lpv = lp.filter(pl.col("post_id").is_in(list(feat)) | pl.col("post_id_b").is_in(list(feat)))
sets = {"video_video": vv, "bench_video_all_parts": bv.drop("part"), "bench_video_test_part": bv.filter(pl.col("part") == "test").drop("part"), "launch_pairs_in_table": lp, "launch_video": lpv}
def score(P, arm):
    s = arms[arm]
    P = P.join(s, on="post_id").join(s.rename({"post_id": "post_id_b", "s": "s_b"}), on="post_id_b")
    d = ((P["likes"].cast(pl.Float64) + 1).log() - (P["likes_b"].cast(pl.Float64) + 1).log()).to_numpy()
    return P["author_id"].to_numpy(), d, (((P["s"] - P["s_b"]).to_numpy() * d) > 0).astype(float)
def acc(a, c, ref=None):
    if len(c) == 0: return None
    u, inv = np.unique(a, return_inverse=True); S = np.zeros(len(u)); C = np.zeros(len(u)); np.add.at(S, inv, c); np.add.at(C, inv, 1)
    idx = rng.integers(0, len(u), (2000, len(u))); b = S[idx].sum(1) / C[idx].sum(1)
    r = {"pairs": int(len(c)), "accounts": int(len(u)), "accuracy": round(float(c.mean()), 4), "lo": round(float(np.quantile(b, .025)), 4), "hi": round(float(np.quantile(b, .975)), 4)}
    if ref is not None:
        R = np.zeros(len(u)); np.add.at(R, inv, ref); db = (S[idx].sum(1) - R[idx].sum(1)) / C[idx].sum(1)
        r |= {"diff_vs_control": round(float(c.mean() - ref.mean()), 4), "diff_lo": round(float(np.quantile(db, .025)), 4), "diff_hi": round(float(np.quantile(db, .975)), 4)}
    return r
out = {}
for name, P in sets.items():
    out[name] = {}
    for gap in (1.25, 2.0):
        ref = None
        if "control" in arms:
            a, d, c = score(P, "control"); ref = c[np.abs(d) >= np.log(gap)]
        for arm in arms:
            a, d, c = score(P, arm); m = np.abs(d) >= np.log(gap)
            r = acc(a[m], c[m], ref if arm != "control" else None); out[name].setdefault(arm, {})[f"gap_{gap}x"] = r
            if r: print(f"{name:<24}{arm:<10} gap {gap:<5} {100*r['accuracy']:5.1f}% ({100*r['lo']:.1f}-{100*r['hi']:.1f}) n={r['pairs']:<5} acc={r['accounts']:<4}" + (f" diff vs control {100*r['diff_vs_control']:+.1f} ({100*r['diff_lo']:+.1f} to {100*r['diff_hi']:+.1f})" if "diff_vs_control" in r else ""))
json.dump(out, open("data/video/model_pairs.json", "w"), indent=1)
