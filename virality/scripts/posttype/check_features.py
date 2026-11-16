#!/usr/bin/env python3
"""Checks on the post-type inputs: agreement with the launch track's labels, no leakage from the post itself or
later posts into trailing values, and equality of the table values with what serving computes from a 56-post history.
Writes data/posttype/feature_checks.json."""
import json, os, numpy as np, polars as pl
from xviral.features import post_type as PTF
from xviral import serve
os.makedirs("data/posttype", exist_ok=True)
out = {}
T = pl.read_parquet("data/work/model/table_pt/post_type.parquet")
# 1. agreement with the launch track (types from whole-timeline account labels)
LP = pl.read_parquet("data/tracks/launch/launch_posts.parquet").select("post_id", pl.col("ptype").alias("track"), "akind")
j = T.select("post_id", "pt_type").join(LP, on="post_id", how="left").with_columns(pl.col("track").fill_null("other"))
out["agreement_with_launch_track"] = {"table_rows": j.height, "same_type": round(float((j["pt_type"] == j["track"]).mean()), 5),
    "track_typed_rows": int((j["track"] != "other").sum()), "of_those_same_here": round(float(j.filter(pl.col("track") != "other").select((pl.col("pt_type") == pl.col("track")).mean()).item()), 4),
    "typed_here": int((j["pt_type"] != "other").sum()), "of_those_same_in_track": round(float(j.filter(pl.col("pt_type") != "other").select((pl.col("pt_type") == pl.col("track")).mean()).item()), 4),
    "track_launch_rows": int(j["track"].is_in(["launch", "open_source"]).sum()),
    "track_launch_also_launch_here": round(float(j.filter(pl.col("track").is_in(["launch", "open_source"])).select(pl.col("pt_type").is_in(["launch", "open_source"]).mean()).item()), 4)}
A = pl.read_parquet("data/tracks/launch/accounts_labeled.parquet").select("author_id", "akind")
v = T.join(pl.read_parquet("data/work/dataset/rows.parquet", columns=["post_id", "author_id"]), on="post_id").join(A, on="author_id", how="left").with_columns(pl.col("akind").fill_null("other"))
out["company_voice_from_previous_posts"] = {k: {"rows": n, "share_with_company_voice": round(m, 4)} for k, n, m in v.group_by("akind").agg(pl.len(), pl.col("pt_company_voice").mean()).iter_rows()}
print(json.dumps(out, indent=1))
# 2. leakage: change the likes of a post and of every later post; the post's row must not change
rows = pl.read_parquet("data/work/dataset/rows.parquet", columns=["post_id", "author_id", "created_at_ms"])
rng = np.random.default_rng(0)
authors = rng.choice(T.filter(pl.col("pt_type") != "other").join(rows, on="post_id")["author_id"].unique().to_numpy(), 300, replace=False)
d = (pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id", "author_id", "created_at_ms", "text", "kind", "likes", "media", "source")
     .filter(pl.col("author_id").is_in(list(authors))).collect())
base = PTF.post_type_features(d).sort("post_id")
# per account, pick a cut post; scramble likes at and after it
cut = d.group_by("author_id").agg(pl.col("created_at_ms").sample(1, seed=1).first().alias("cut"))
d2 = d.join(cut, on="author_id").with_columns(pl.when(pl.col("created_at_ms") >= pl.col("cut")).then(pl.col("likes") * 7 + 1000).otherwise(pl.col("likes")).alias("likes"))
pert = PTF.post_type_features(d2.drop("cut")).sort("post_id")
chk = base.join(d.select("post_id", "author_id", "created_at_ms").join(cut, on="author_id"), on="post_id").sort("post_id")
at_or_before = (chk["created_at_ms"] <= chk["cut"]).to_numpy()
A = base.select(PTF.COLUMNS).cast(pl.Float64).fill_null(-99).to_numpy(); B = pert.select(PTF.COLUMNS).cast(pl.Float64).fill_null(-99).to_numpy()
same = np.isclose(A, B, atol=1e-9).all(1)
out["leakage_test"] = {"accounts": 300, "posts": int(len(same)), "rows_at_or_before_the_changed_post": int(at_or_before.sum()), "of_those_unchanged": int(same[at_or_before].sum()),
                       "later_rows_changed (expected, they see the changed likes)": int((~same[~at_or_before]).sum())}
# also: dropping all later posts does not change a row
keep = d.join(cut, on="author_id").filter(pl.col("created_at_ms") <= pl.col("cut")).drop("cut")
trunc = PTF.post_type_features(keep).sort("post_id")
bb = base.join(trunc.select("post_id"), on="post_id", how="semi").sort("post_id")
out["leakage_test"]["rows_unchanged_when_later_posts_are_removed"] = f"{int(np.isclose(bb.select(PTF.COLUMNS).cast(pl.Float64).fill_null(-99).to_numpy(), trunc.select(PTF.COLUMNS).cast(pl.Float64).fill_null(-99).to_numpy(), atol=1e-9).all(1).sum())} of {trunc.height}"
print(json.dumps(out["leakage_test"], indent=1))
# 3. serving equality: the post as a draft against its previous 56 posts, through serve.feature_frame
from xviral.schema import conform
full = pl.scan_parquet("data/work/filtered/posts.parquet").filter(pl.col("author_id").is_in(list(authors[:120]))).collect()
by = {a: g.sort("created_at_ms") for (a,), g in full.group_by("author_id")}
tt = T.join(rows, on="post_id").filter(pl.col("author_id").is_in(list(authors[:120])))
samp = pl.concat([tt.filter(pl.col("pt_type") != "other").sample(200, seed=0), tt.filter(pl.col("pt_type") == "other").sample(200, seed=0)])
diffs = {c: 0 for c in PTF.COLUMNS}; n = 0; short = 0
for r in samp.iter_rows(named=True):
    g = by[r["author_id"]]; x = g.filter(pl.col("post_id") == r["post_id"])
    hist = g.filter(pl.col("created_at_ms") < r["created_at_ms"]).tail(56)
    draft = x.with_columns(pl.lit("draft").alias("source"), pl.lit(None, dtype=pl.Int64).alias("likes"), pl.lit(None, dtype=pl.Int64).alias("views"))
    f = serve.feature_frame(conform(pl.concat([hist, draft], how="vertical_relaxed")), None, {}).filter(pl.col("source") == "draft")
    n += 1; short += hist.height < 50
    for c in PTF.COLUMNS:
        a, b = r[c], f[c][0]
        a = -99 if a is None else float(a); b = -99 if b is None else float(b)
        if abs(a - b) > 1e-6: diffs[c] += 1
out["serving_equality"] = {"posts": n, "with_under_50_previous_posts": int(short), "columns_differing (count of posts)": {c: v for c, v in diffs.items() if v}, "columns_identical": sum(v == 0 for v in diffs.values()), "columns": len(diffs)}
print(json.dumps(out["serving_equality"], indent=1))
json.dump(out, open("data/posttype/feature_checks.json", "w"), indent=1)
