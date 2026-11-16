#!/usr/bin/env python3
"""What a type correction does to the general pair benchmark: the members' per-pair scores are averaged and the
correction of each side's post type is added (types computed for each post at its own time from the account's earlier
posts). Writes a folder bench_table.py reads like any model.

    python scripts/posttype/bench_correction.py --members dir1,dir2,dir3 --correction corr.json --out data/bench2/baseline_corr
"""
import argparse, json, os, sys, polars as pl
sys.path.insert(0, "scripts/evidence")
import compare_runs as CR
from xviral.features import post_type as PTF
ap = argparse.ArgumentParser(); ap.add_argument("--members", required=True); ap.add_argument("--correction", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--posts", default="data/bench2/posts_pair_accounts.parquet")
args = ap.parse_args()
J = None
for i, d in enumerate(args.members.split(",")):
    R = CR.load(f"{d}/shard_*.parquet").rename({"s_likes": f"s_{i}"})
    J = R if J is None else J.join(R.select("post_id", "post_id_b", f"s_{i}"), on=["post_id", "post_id_b"])
J = J.with_columns(pl.mean_horizontal(pl.col(r"^s_\d+$")).alias("s"))
d = (pl.scan_parquet(args.posts).select("post_id", "author_id", "created_at_ms", "text", "kind", "likes", "media", "source").filter(pl.col("author_id").is_in(J["author_id"].unique().implode())).collect())
F = PTF.post_type_features(d, only=pl.concat([J["post_id"], J["post_id_b"]]).unique())
w = json.load(open(args.correction))["weights_log10"]
F = F.select("post_id", "pt_type", pl.sum_horizontal([pl.col(c).cast(pl.Float64) * v for c, v in w.items()]).alias("corr"))
J = J.join(F, on="post_id").join(F.rename({"post_id": "post_id_b", "pt_type": "pt_type_b", "corr": "corr_b"}), on="post_id_b")
J = J.with_columns((pl.col("s") + pl.col("corr") - pl.col("corr_b")).alias("s_likes"))
os.makedirs(args.out, exist_ok=True)
J.select("post_id", "post_id_b", "author_id", "likes", "likes_b", "s_likes").write_parquet(f"{args.out}/shard_0.parquet")
typed = J.filter((pl.col("pt_type") != "other") | (pl.col("pt_type_b") != "other"))
print("pairs", J.height, "with an own announcement on either side", typed.height, "whose score changes", J.filter(pl.col("corr") != pl.col("corr_b")).height,
      "direction flips", J.filter((pl.col("s") * pl.col("s_likes")) < 0).height)
