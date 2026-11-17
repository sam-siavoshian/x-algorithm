#!/usr/bin/env python3
"""Pair accuracy of the arms of train_vf.py on video-involved pairs, table path, likes; train, val and test side by side.

    python scripts/video_files/eval_vf_pairs.py baseline=data/vf/runs/baseline control=data/vf/runs/control_s1,data/vf/runs/control_s2 vf=...

Several run folders after one name are averaged (an ensemble of seeds); each folder is also scored alone. Pair sets, all
same account, real like counts differing by 1.25x or more (and 2x), at most 365 days apart, at most 50 an account:
  video_video_<split>   both posts are videos with features, both in that split (train / val / test)
  video_other_test      one video with features and one other post of the account, both in the test split
  launch_video_test     both are launch-type posts (tracks/launch), test split, a video with features on either side
Accuracy with an interval from resampling accounts, and the difference to `control` on the same pairs (paired).
Writes data/vf/model_pairs.json.
"""
import json, sys, numpy as np, polars as pl
rng = np.random.default_rng(0)
arms = {}
for a in sys.argv[1:]:
    k, v = a.split("="); dirs = v.split(","); P = [pl.read_parquet(f"{d}/pred.parquet") for d in dirs]
    J = P[0].select("post_id", pl.col("likes_mean").alias("s0"))
    for i, p in enumerate(P[1:], 1): J = J.join(p.select("post_id", pl.col("likes_mean").alias(f"s{i}")), on="post_id")
    arms[k] = J.select("post_id", pl.mean_horizontal(pl.col(r"^s\d+$")).alias("s"))
    if len(dirs) > 1:
        for i, p in enumerate(P): arms[f"{k}_seed{i+1}"] = p.select("post_id", pl.col("likes_mean").alias("s"))
base = pl.read_parquet(sys.argv[1].split("=")[1].split(",")[0] + "/pred.parquet").select("post_id", "author_idx", "has_vf", "split")
posts = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id", "created_at_ms", "likes").collect()
base = base.join(posts, on="post_id")
L = set(pl.read_parquet("data/tracks/launch/launch_posts.parquet").filter(pl.col("ptype").is_in(["launch", "open_source"]))["post_id"].to_list())
base = base.with_columns(pl.col("post_id").is_in(list(L)).alias("launch"))
def pairs(df, cond=None, cap=50):
    p = df.join(df, on="author_idx", suffix="_b").filter((pl.col("post_id") < pl.col("post_id_b")) & ((pl.col("created_at_ms") - pl.col("created_at_ms_b")).abs() <= 365 * 86_400_000))
    if cond is not None: p = p.filter(cond)
    return p.with_columns(pl.int_range(pl.len()).shuffle(seed=1).over("author_idx").alias("r")).filter(pl.col("r") < cap).select("author_idx", "post_id", "post_id_b", "likes", "likes_b")
sets = {f"video_video_{s}": pairs(base.filter(pl.col("has_vf") & (pl.col("split") == s))) for s in ("train", "val", "test")}
T = base.filter(pl.col("split") == "test")
sets["video_other_test"] = pairs(T, pl.col("has_vf") != pl.col("has_vf_b"))
sets["launch_video_test"] = pairs(T.filter(pl.col("launch")), pl.col("has_vf") | pl.col("has_vf_b"), cap=10_000)
def score(P, arm):
    s = arms[arm]; P = P.join(s, on="post_id").join(s.rename({"post_id": "post_id_b", "s": "s_b"}), on="post_id_b")
    d = ((P["likes"].cast(pl.Float64) + 1).log() - (P["likes_b"].cast(pl.Float64) + 1).log()).to_numpy()
    return P["author_idx"].to_numpy(), d, (((P["s"] - P["s_b"]).to_numpy() * d) > 0).astype(float)
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
            if r: print(f"{name:<22}{arm:<14} gap {gap:<5} {100*r['accuracy']:5.1f}% ({100*r['lo']:.1f}-{100*r['hi']:.1f}) n={r['pairs']:<5} acc={r['accounts']:<4}" + (f" diff vs control {100*r['diff_vs_control']:+.1f} ({100*r['diff_lo']:+.1f} to {100*r['diff_hi']:+.1f})" if "diff_vs_control" in r else ""))
json.dump(out, open("data/vf/model_pairs.json", "w"), indent=1)
