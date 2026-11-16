#!/usr/bin/env python3
"""The launch track's 850 held-out launch pairs and predicted-against-actual lift by post type, for any set of models
scored with scripts/posttype/score_posts.py. Pairs are built as scripts/launch/model_check.py builds them.

    python scripts/posttype/eval_launch.py --scores data/posttype/scores --model label=name1,name2,... [--model ...] --out data/posttype/launch_eval.json
A model with several names is a score-level ensemble: the members' log10(likes + 1) averaged.
`--scale label=k` sets the confidence scale of a model (default: the baseline's 1.5156 for every model).
"""
import argparse, datetime as dt, json, numpy as np, polars as pl
ap = argparse.ArgumentParser(); ap.add_argument("--scores", default="data/posttype/scores"); ap.add_argument("--model", action="append", required=True); ap.add_argument("--scale", action="append", default=[])
ap.add_argument("--out", required=True)
args = ap.parse_args()
CUT = int(dt.datetime(2026, 4, 23, tzinfo=dt.timezone.utc).timestamp() * 1000)
K0 = 1.5156
scales = {s.split("=")[0]: float(s.split("=")[1]) for s in args.scale}
posts = pl.read_parquet(f"{args.scores}/posts.parquet")
T = pl.read_parquet("data/tracks/launch/launch_posts.parquet")
L = T.filter(pl.col("ptype").is_in(["launch", "open_source"]) & (pl.col("created_at_ms") >= CUT)).filter(pl.len().over("author_id") >= 2)
ref = pl.read_parquet("data/tracks/launch/launch_scores.parquet").select("post_id", pl.col("pred_likes").alias("pred_ref"))
rng = np.random.default_rng(0)
def acc(c, a):
    if len(c) < 20: return {"pairs": int(len(c))}
    uu, iv = np.unique(a, return_inverse=True); Sx = np.zeros(len(uu)); C = np.zeros(len(uu)); np.add.at(Sx, iv, c); np.add.at(C, iv, 1)
    idx = rng.integers(0, len(uu), (2000, len(uu))); b = Sx[idx].sum(1) / C[idx].sum(1)
    return {"pairs": int(len(c)), "accounts": int(len(uu)), "accuracy": round(float(c.mean()), 4), "lo": round(float(np.quantile(b, .025)), 4), "hi": round(float(np.quantile(b, .975)), 4)}
def load(names):
    S = None
    for i, n in enumerate(names):
        if n == "launch_track_file": x = ref.select("post_id", (pl.col("pred_ref") + 1).log10().alias(f"m{i}"))
        else: x = pl.read_parquet(f"{args.scores}/{n}.*.parquet").select("post_id", (pl.col("pred_likes") + 1).log10().alias(f"m{i}"))
        S = x if S is None else S.join(x, on="post_id")
    return S.select("post_id", pl.mean_horizontal(pl.col(r"^m\d+$")).alias("lp"))
out = {}; keep_pairs = {}
for spec in args.model:
    label, names = spec.split("=", 1); S = load(names.split(","))
    # --- pairs (model_check.py) ---
    P = T.join(ref.select("post_id"), on="post_id").join(S, on="post_id")           # the 582 posts the launch track scored, in its order
    p = P.join(P, on="author_id", suffix="_b").filter(pl.col("post_id") < pl.col("post_id_b"))
    p = p.with_columns(((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log()).alias("d"), (pl.col("lp") - pl.col("lp_b")).alias("s"))
    p = p.filter(pl.col("d").abs() >= np.log(1.25))
    p = p.with_columns(pl.int_range(pl.len()).shuffle(seed=1).over("author_id").alias("r")).filter(pl.col("r") < 30)
    d, s, a = p["d"].to_numpy(), p["s"].to_numpy(), p["author_id"].to_numpy(); c = ((s * d) > 0).astype(float)
    k = scales.get(label, K0); conf = 1 / (1 + np.exp(-k * np.abs(s)))
    res = {"members": names.split(","), "confidence_scale": k, "all": acc(c, a)}
    keep_pairs[label] = p.select("post_id", "post_id_b", "author_id").with_columns(pl.Series("c", c), pl.Series("conf", conf), pl.Series("d", d))
    for g, nm in ((1.25, "gap_1.25x"), (2, "gap_2x"), (3, "gap_3x"), (5, "gap_5x")):
        m = np.abs(d) >= np.log(g); mc = m & (conf >= 0.6)
        res[nm] = acc(c[m], a[m]) | {"confident_share": round(float(mc.sum() / max(m.sum(), 1)), 4), "confident": acc(c[mc], a[mc])}
    res["by_confidence"] = {f"{lo:.2f}-{min(hi, 1):.2f}": acc(c[(conf >= lo) & (conf < hi)], a[(conf >= lo) & (conf < hi)]) for lo, hi in ((0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 1.01))}
    for f in ("names_product_first_line", "open_problem", "is_quote"):
        q = p.filter(pl.col(f) != pl.col(f + "_b")); w = np.where(q[f].to_numpy(), 1, -1)
        res[f"pairs_differing_in_{f}"] = {"pairs": q.height, "version_with_it_wins": round(float(((q["d"].to_numpy() * w) > 0).mean()), 4), "model_prefers_version_with_it": round(float(((q["s"].to_numpy() * w) > 0).mean()), 4)}
    q = p.filter((pl.col("media_kind") == "video") != (pl.col("media_kind_b") == "video")); w = np.where((q["media_kind"] == "video").to_numpy(), 1, -1)
    res["pairs_differing_in_video"] = {"pairs": q.height, "version_with_it_wins": round(float(((q["d"].to_numpy() * w) > 0).mean()), 4), "model_prefers_version_with_it": round(float(((q["s"].to_numpy() * w) > 0).mean()), 4)}
    # --- level by type ---
    if names.split(",")[0] != "launch_track_file":
        z = posts.filter(pl.col("med_likes").is_not_null()).join(S, on="post_id").with_columns(((pl.col("likes") + 1) / (pl.col("med_likes") + 1)).log().alias("act"), (pl.col("lp") * np.log(10) - (pl.col("med_likes") + 1).log()).alias("pre"))
        lv = {}
        for (ty,), g in z.group_by("ptype"):
            if g.height < 15: continue
            act, pre, au = g["act"].to_numpy(), g["pre"].to_numpy(), g["author_id"].to_numpy()
            uu, iv = np.unique(au, return_inverse=True); mem = [np.flatnonzero(iv == j) for j in range(len(uu))]
            bs = np.array([[np.median(act[ii]), np.median(pre[ii])] for ii in (np.concatenate([mem[j] for j in rng.integers(0, len(uu), len(uu))]) for _ in range(400))])
            lv[ty] = {"posts": g.height, "accounts": int(len(uu)), "actual_x": round(float(np.exp(np.median(act))), 3), "actual_ci": [round(float(np.exp(x)), 2) for x in np.quantile(bs[:, 0], [.025, .975])],
                      "predicted_x": round(float(np.exp(np.median(pre))), 3), "predicted_ci": [round(float(np.exp(x)), 2) for x in np.quantile(bs[:, 1], [.025, .975])],
                      "actual_beats_median": round(float((act > 0).mean()), 3), "predicted_beats_median": round(float((pre > 0).mean()), 3),
                      "spearman": round(float(pl.DataFrame({"a": act, "b": pre}).select(pl.corr("a", "b", method="spearman")).item()), 3),
                      "mean_abs_log_error": round(float(np.abs(act - pre).mean()), 3)}
        res["level_by_type"] = lv
    out[label] = res
    print(f"\n== {label} ({names})  pairs {res['all']}")
    for nm in ("gap_1.25x", "gap_2x", "gap_3x", "gap_5x"):
        r = res[nm]; cf = r["confident"]
        print(f"  {nm:<10} n={r['pairs']:<4} acc {100*r['accuracy']:.1f}  confident share {100*r['confident_share']:.0f}% n={cf['pairs']:<4} acc {100*cf.get('accuracy', float('nan')):.1f}")
    print("  bands", {b: (v["pairs"], v.get("accuracy")) for b, v in res["by_confidence"].items()})
    for f in ("names_product_first_line", "video"): print("  ", f, res[f"pairs_differing_in_{f}"])
    for ty, v in sorted(res.get("level_by_type", {}).items(), key=lambda kv: -kv[1]["posts"]):
        print(f"  {ty:<12} n={v['posts']:<5} actual {v['actual_x']:.2f}x  predicted {v['predicted_x']:.2f}x  beats median {v['actual_beats_median']:.2f}/{v['predicted_beats_median']:.2f}  rho {v['spearman']:.2f}  abs log err {v['mean_abs_log_error']:.2f}")
# paired differences against the first model, intervals from resampling accounts
labels = list(keep_pairs); base = keep_pairs[labels[0]]
out["_versus_" + labels[0]] = {}
for lb in labels[1:]:
    J = base.join(keep_pairs[lb].select("post_id", "post_id_b", pl.col("c").alias("c2"), pl.col("conf").alias("conf2")), on=["post_id", "post_id_b"])
    a = J["author_id"].to_numpy(); uu, iv = np.unique(a, return_inverse=True); mem = [np.flatnonzero(iv == j) for j in range(len(uu))]
    c1, c2, f1, f2, big = J["c"].to_numpy(), J["c2"].to_numpy(), J["conf"].to_numpy() >= 0.6, J["conf2"].to_numpy() >= 0.6, np.abs(J["d"].to_numpy()) >= np.log(2)
    def stats(i):
        g = lambda c, f: c[i][f[i]].mean() if f[i].sum() else np.nan   # noqa: E731
        return [c2[i].mean() - c1[i].mean(), g(c2, f2) - g(c1, f1), c2[i][big[i]].mean() - c1[i][big[i]].mean(), g(c2, f2 & big) - g(c1, f1 & big)]
    bs = np.array([stats(np.concatenate([mem[j] for j in rng.integers(0, len(uu), len(uu))])) for _ in range(1000)]); pt = stats(np.arange(len(a)))
    r = {nm: {"gap": round(float(pt[i]), 4), "ci": [round(float(x), 4) for x in np.nanquantile(bs[:, i], [.025, .975])]} for i, nm in enumerate(("all", "confident", "gap_2x", "confident_gap_2x"))} | {"pairs": J.height}
    out["_versus_" + labels[0]][lb] = r
    print(f"{lb} minus {labels[0]}: " + "  ".join(f"{nm} {100*v['gap']:+.1f} [{100*v['ci'][0]:+.1f},{100*v['ci'][1]:+.1f}]" for nm, v in r.items() if nm != "pairs"))
json.dump(out, open(args.out, "w"), indent=1)
