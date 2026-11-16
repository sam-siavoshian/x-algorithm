#!/usr/bin/env python3
"""Accuracy of every benchmarked model (bench2/<model>/shard_*.parquet) on the same-topic benchmark pairs in which at
least one side is a launch-type post / an own announcement. Test part only, likes differing by 1.25x or more.
Writes data/launch/bench_launch.json."""
import glob, json, os, numpy as np, polars as pl
T = pl.read_parquet("data/launch/table.parquet").select("post_id", "ptype", "akind")
ANN = ["launch", "open_source", "feature", "funding", "milestone", "hiring"]
def load(m):
    R = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"data/bench2/{m}/shard_*.parquet"))])
    flip = pl.col("post_id") > pl.col("post_id_b")
    return R.select("part", pl.when(flip).then("post_id_b").otherwise("post_id").alias("post_id"), pl.when(flip).then("post_id").otherwise("post_id_b").alias("post_id_b"), "author_id",
                    pl.when(flip).then("likes_b").otherwise("likes").alias("likes"), pl.when(flip).then("likes").otherwise("likes_b").alias("likes_b"),
                    pl.when(flip).then(-pl.col("s_likes")).otherwise("s_likes").alias("s_likes")).unique(["post_id", "post_id_b"])
models = sorted(os.path.basename(p) for p in glob.glob("data/bench2/*") if os.path.isdir(p))
runs = {m: load(m) for m in models}
groups = {"ensemble_ts46 (ts46_s1, ts46_s2, compare_v2_full)": ["ts46_s1", "ts46_s2", "compare_v2_full"], "ensemble_all_ten": models}
for g, ms in groups.items():
    J = runs[ms[0]].rename({"s_likes": "s_0"})
    for i, m in enumerate(ms[1:], 1):
        J = J.join(runs[m].select("post_id", "post_id_b", pl.col("s_likes").alias(f"s_{i}")), on=["post_id", "post_id_b"])
    runs[g] = J.with_columns(pl.mean_horizontal(pl.col(r"^s_\d+$")).alias("s_likes"))
rng = np.random.default_rng(1)
def acc(R):
    if R.height == 0: return None
    d = ((R["likes"].cast(pl.Float64) + 1).log() - (R["likes_b"].cast(pl.Float64) + 1).log()).to_numpy(); s = R["s_likes"].to_numpy(); c = ((s * d) > 0).astype(float); a = R["author_id"].to_numpy()
    u, inv = np.unique(a, return_inverse=True); S = np.zeros(len(u)); C = np.zeros(len(u)); np.add.at(S, inv, c); np.add.at(C, inv, 1)
    idx = rng.integers(0, len(u), (2000, len(u))); b = S[idx].sum(1) / C[idx].sum(1)
    return {"pairs": int(len(c)), "accounts": int(len(u)), "accuracy": round(float(c.mean()), 4), "lo": round(float(np.quantile(b, .025)), 4), "hi": round(float(np.quantile(b, .975)), 4)}
out = {}
for m, R in runs.items():
    R = R.filter(pl.col("part") == "test").join(T.rename({"ptype": "pt_a", "akind": "akind"}), on="post_id", how="left").join(T.select(pl.col("post_id").alias("post_id_b"), pl.col("ptype").alias("pt_b")), on="post_id_b", how="left")
    R = R.with_columns(pl.col("pt_a").fill_null("other"), pl.col("pt_b").fill_null("other"), pl.col("akind").fill_null("other"))
    R = R.filter(((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log()).abs() >= np.log(1.25))
    L = pl.col("pt_a").is_in(["launch", "open_source"]) | pl.col("pt_b").is_in(["launch", "open_source"]); A = pl.col("pt_a").is_in(ANN) | pl.col("pt_b").is_in(ANN)
    big = ((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log()).abs() >= np.log(2)
    out[m] = {"all_pairs": acc(R), "one_side_launch": acc(R.filter(L)), "one_side_announcement": acc(R.filter(A)), "one_side_announcement_gap_2x": acc(R.filter(A & big)), "company_or_founder_account_any_pair": acc(R.filter(pl.col("akind") != "other")),
              "both_sides_announcement": acc(R.filter(pl.col("pt_a").is_in(ANN) & pl.col("pt_b").is_in(ANN)))}
    f = lambda x: f"{100*x['accuracy']:5.1f}% ({100*x['lo']:.0f}-{100*x['hi']:.0f}) n={x['pairs']:<4}" if x else "-"
    print(f"{m[:34]:<34} all {f(out[m]['all_pairs'])} | launch {f(out[m]['one_side_launch'])} | announce {f(out[m]['one_side_announcement'])} | co/founder acct {f(out[m]['company_or_founder_account_any_pair'])}")
json.dump(out, open("data/launch/bench_launch.json", "w"), indent=1)
