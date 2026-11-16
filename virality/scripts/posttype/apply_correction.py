#!/usr/bin/env python3
"""Apply a fitted type correction (fit_correction.py) to draft-path scores (score_posts.py): writes a new score file
whose prediction is the members' mean log10(likes + 1) plus the correction of the post's type columns, the columns
being computed for the draft exactly as serving computes them.

    python scripts/posttype/apply_correction.py --scores data/posttype/scores --members a,b,c --correction corr.json --name baseline_corr
"""
import argparse, json, numpy as np, polars as pl
from xviral.features import post_type as PTF
ap = argparse.ArgumentParser(); ap.add_argument("--scores", default="data/posttype/scores"); ap.add_argument("--members", required=True); ap.add_argument("--correction", required=True); ap.add_argument("--name", required=True)
args = ap.parse_args()
S = None
for i, n in enumerate(args.members.split(",")):
    x = pl.read_parquet(f"{args.scores}/{n}.*.parquet").select("post_id", (pl.col("pred_likes") + 1).log10().alias(f"m{i}"))
    S = x if S is None else S.join(x, on="post_id")
S = S.select("post_id", pl.mean_horizontal(pl.col(r"^m\d+$")).alias("lp"))
fpath = f"{args.scores}/post_type_columns.parquet"
try: F = pl.read_parquet(fpath)
except FileNotFoundError:
    P = pl.read_parquet(f"{args.scores}/posts.parquet")
    d = (pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id", "author_id", "created_at_ms", "text", "kind", "likes", "media", "source")
         .filter(pl.col("author_id").is_in(P["author_id"].unique().implode())).collect())
    # the scored post as a draft: its own likes are not read for its own row in any case (strictly earlier posts only)
    F = PTF.post_type_features(d, only=P["post_id"]); F.write_parquet(fpath)
w = json.load(open(args.correction))["weights_log10"]
J = S.join(F, on="post_id").with_columns(pl.sum_horizontal([pl.col(c).cast(pl.Float64) * v for c, v in w.items()]).alias("corr"))
J.select("post_id", (10 ** (pl.col("lp") + pl.col("corr")) - 1).alias("pred_likes"), pl.col("corr")).write_parquet(f"{args.scores}/{args.name}.0.parquet")
print(args.name, J.height, "posts; corrected:", int((J["corr"] != 0).sum()), "median correction among corrected x", round(float(10 ** J.filter(pl.col("corr") != 0)["corr"].median()), 3))
