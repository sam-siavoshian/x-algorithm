#!/usr/bin/env python3
"""How well the current model ranks launch posts: same-account pairs of launch-type posts from the test period
(created on or after 2026-04-23), each scored through the draft path (scripts/launch/score_launch.py).
Also scores the same pairs with a rule score fitted on launch posts from before the cutoff only (a held-out test of the
evidence table). Writes data/launch/model_check.json."""
import datetime as dt, json, numpy as np, polars as pl
rng = np.random.default_rng(0)
CUT = int(dt.datetime(2026, 4, 23, tzinfo=dt.timezone.utc).timestamp() * 1000)
K = 1.5156          # confidence scale of the likes comparison, research/13_same_topic_pair_benchmark_ensemble.json
T = pl.read_parquet("data/launch/table.parquet")
T = T.with_columns((pl.col("chars") >= 280).alias("long280"), (pl.col("lines") >= 3).alias("lines3"), (pl.col("media_kind") == "video").alias("video"), (pl.col("media_kind") == "image").alias("image"),
                   (pl.col("ext_link") & ~pl.col("points_to_reply")).alias("link_in_post"), (pl.col("dow") >= 6).alias("weekend"), pl.col("hour_utc").is_between(18, 23).alias("h18_23"),
                   (pl.col("gap_prev_h") < 2).fill_null(False).alias("prev2h"))
F = ["video", "image", "link_in_post", "thread_marker", "open_announce", "open_problem", "open_number", "names_product_first_line", "long280", "lines3", "tags", "hashtag", "is_quote", "weekend", "h18_23", "prev2h"]
CA = pl.col("source") == "community_archive"
# rule score: account-fixed-effects regression of lift on the attributes, launch posts before the cutoff only
tr = T.filter(pl.col("ptype").is_in(["launch", "open_source"]) & (pl.col("created_at_ms") < CUT) & pl.col("lift_likes").is_not_null() & CA).filter(pl.len().over("author_id") >= 2)
X = tr.select([pl.col(c).cast(pl.Float64) for c in F]).to_numpy(); Y = tr["lift_likes"].to_numpy(); u, inv = np.unique(tr["author_id"].to_numpy(), return_inverse=True)
def dm(M):
    s = np.zeros((len(u),) + M.shape[1:]); np.add.at(s, inv, M); c = np.bincount(inv).astype(float); return M - (s / c.reshape(-1, *([1] * (M.ndim - 1))))[inv]
beta = np.linalg.lstsq(dm(X), dm(Y), rcond=None)[0]
print("rule score fitted on", len(Y), "pre-cutoff launch posts,", len(u), "accounts:", {f: round(float(np.exp(b)), 2) for f, b in zip(F, beta)})
S = pl.read_parquet("data/launch/launch_scores.parquet").select("post_id", "pred_likes", "pred_views", "history", "members_likes")
P = T.join(S, on="post_id").with_columns(pl.sum_horizontal([pl.col(f).cast(pl.Float64) * float(b) for f, b in zip(F, beta)]).alias("rule"))
SURV = {"video": 1, "names_product_first_line": 1, "open_number": 1, "open_problem": -1, "is_quote": -1, "link_in_post": -1}
P = P.with_columns(pl.sum_horizontal([pl.col(f).cast(pl.Float64) * w for f, w in SURV.items()]).alias("count"))
p = P.join(P, on="author_id", suffix="_b").filter(pl.col("post_id") < pl.col("post_id_b"))
p = p.with_columns(((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log()).alias("d"), (pl.col("pred_likes") + 1).log10().sub((pl.col("pred_likes_b") + 1).log10()).alias("s"),
                   (pl.col("rule") - pl.col("rule_b")).alias("sr"), (pl.col("count") - pl.col("count_b")).alias("sc"), ((pl.col("created_at_ms") - pl.col("created_at_ms_b")).abs() / 86_400_000).alias("gap_d"))
p = p.filter(pl.col("d").abs() >= np.log(1.25))
p = p.with_columns(pl.int_range(pl.len()).shuffle(seed=1).over("author_id").alias("r")).filter(pl.col("r") < 30)
mem = np.array([[np.log10(a + 1) - np.log10(b + 1) for a, b in zip(x, y)] for x, y in zip(p["members_likes"].to_list(), p["members_likes_b"].to_list())])
p = p.with_columns(pl.Series("agree", (np.sign(mem) == np.sign(p["s"].to_numpy())[:, None]).all(1)))
def acc(q, col):
    q = q.filter(pl.col(col) != 0)
    if q.height < 20: return {"pairs": q.height, "too_thin": True}
    c = ((q[col].to_numpy() * q["d"].to_numpy()) > 0).astype(float); a = q["author_id"].to_numpy(); uu, iv = np.unique(a, return_inverse=True)
    Sx = np.zeros(len(uu)); C = np.zeros(len(uu)); np.add.at(Sx, iv, c); np.add.at(C, iv, 1); idx = rng.integers(0, len(uu), (2000, len(uu))); b = Sx[idx].sum(1) / C[idx].sum(1)
    return {"pairs": int(len(c)), "accounts": int(len(uu)), "accuracy": round(float(c.mean()), 4), "lo": round(float(np.quantile(b, .025)), 4), "hi": round(float(np.quantile(b, .975)), 4)}
conf = 1 / (1 + np.exp(-K * np.abs(p["s"].to_numpy()))); p = p.with_columns(pl.Series("conf", conf))
big = pl.col("d").abs() >= np.log(2)
out = {"pairs": p.height, "accounts": p["author_id"].n_unique(), "posts_scored": S.height, "design": "same account, both launch-type posts, both posted on or after 2026-04-23, likes differ by 1.25x or more, at most 30 pairs per account",
       "model": "ensemble of ts46_s1, ts46_s2, compare_v2_full (the three 46-column small models), draft path, 56 posts of history",
       "model_all": acc(p, "s"), "model_gap_2x": acc(p.filter(big), "s"), "model_gap_3x": acc(p.filter(pl.col("d").abs() >= np.log(3)), "s"),
       "model_within_30_days": acc(p.filter(pl.col("gap_d") <= 30), "s"),
       "model_company_accounts": acc(p.filter(pl.col("akind") == "company"), "s"), "model_founder_accounts": acc(p.filter(pl.col("akind") == "founder"), "s"), "model_other_accounts": acc(p.filter(pl.col("akind") == "other"), "s"),
       "model_confident_0.6_plus": acc(p.filter(pl.col("conf") >= 0.6), "s") | {"share_of_pairs": round(float((conf >= 0.6).mean()), 4)},
       "model_confident_0.6_plus_gap_2x": acc(p.filter((pl.col("conf") >= 0.6) & big), "s"),
       "model_members_agree": acc(p.filter(pl.col("agree")), "s") | {"share_of_pairs": round(float(p["agree"].mean()), 4)},
       "rule_score_fitted_before_cutoff": acc(p, "sr"), "rule_score_gap_2x": acc(p.filter(big), "sr"), "survivor_count_score": acc(p, "sc"),
       "model_when_rule_agrees": acc(p.filter((pl.col("s") * pl.col("sr")) > 0), "s"), "model_when_rule_disagrees": acc(p.filter((pl.col("s") * pl.col("sr")) < 0), "s"),
       "later_post_wins": round(float(((p["d"] > 0) == (p["created_at_ms"] > p["created_at_ms_b"])).mean()), 4)}
out["rule_score_weights_likes_ratio"] = {f: round(float(np.exp(b)), 3) for f, b in zip(F, beta)}; out["rule_score_fit"] = {"posts": int(len(Y)), "accounts": int(len(u))}
# video pairs: does the model know video?
for nm, f in (("video", "video"), ("names_product_first_line", "names_product_first_line"), ("open_problem", "open_problem"), ("is_quote", "is_quote")):
    q = p.filter(pl.col(f) != pl.col(f + "_b"))
    if q.height >= 30:
        w = np.where(q[f].to_numpy(), 1, -1)
        out[f"pairs_differing_in_{nm}"] = {"pairs": q.height, "version_with_it_wins": round(float(((q["d"].to_numpy() * w) > 0).mean()), 4), "model_prefers_version_with_it": round(float(((q["s"].to_numpy() * w) > 0).mean()), 4), "model_accuracy": acc(q, "s").get("accuracy")}
# does the model expect a launch to beat the account's usual? predicted vs actual ratio to the trailing median
z = P.filter(pl.col("med_likes").is_not_null())
out["launch_vs_own_median"] = {"posts": z.height, "actual_median_ratio": round(float(((z["likes"] + 1) / (z["med_likes"] + 1)).median()), 3), "predicted_median_ratio": round(float(((z["pred_likes"] + 1) / (z["med_likes"] + 1)).median()), 3),
                               "actual_beat_own_median": round(float((z["likes"] > z["med_likes"]).mean()), 4), "predicted_beat_own_median": round(float((z["pred_likes"] > z["med_likes"]).mean()), 4),
                               "spearman_pred_vs_actual_lift": round(float(pl.DataFrame({"a": ((z["likes"] + 1) / (z["med_likes"] + 1)).log(), "b": ((z["pred_likes"] + 1) / (z["med_likes"] + 1)).log()}).select(pl.corr("a", "b", method="spearman")).item()), 3)}
print(json.dumps(out, indent=1))
json.dump(out, open("data/launch/model_check.json", "w"), indent=1)
