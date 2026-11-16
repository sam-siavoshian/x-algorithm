#!/usr/bin/env python3
"""Choose the controlled pairs to train on from the scored candidates of mine_pairs.py.

    python scripts/evidence/select_pairs.py --dir <pairs dir> --posts <work>/filtered/posts.parquet --benchmark benchmark_pairs.parquet

Kept: pairs of one account at most 14 days apart whose normalised texts differ, both at least 6 words and neither
a reply-style post (opening with @), that either share a link or quoted post (tier "tac", the benchmark's own
definition, any cosine) or have bge-small cosine of at least --cos-min (tier "strict" at 0.90 and over, read as the
same message reworded; "loose" from --cos-min to 0.90, same topic or template). Posts that sit in more than
--max-degree mined pairs are templates (daily deals, scheduled series) and are dropped; an account contributes at most
--max-per-account pairs. Training pairs have both posts before the cutoff, contain no benchmark post and come from
no account with a benchmark calibration pair.
"""
import argparse
import json

import numpy as np
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--posts", required=True)
ap.add_argument("--benchmark", required=True)
ap.add_argument("--cutoff", default="2026-04-23")
ap.add_argument("--cos-min", type=float, default=0.85)
ap.add_argument("--max-degree", type=int, default=3)
ap.add_argument("--max-per-account", type=int, default=40)
args = ap.parse_args()
from xviral.util.core import date_to_ms

cut = date_to_ms(args.cutoff)
s = pl.read_parquet(f"{args.dir}/scored_joined.parquet").drop("deg", "deg_b")
ids = pl.concat([s["post_id"], s["post_id_b"]]).unique().to_list()
at = pl.scan_parquet(args.posts).filter(pl.col("post_id").is_in(ids)).select("post_id", pl.col("text").fill_null("").str.strip_chars().str.starts_with("@").alias("at")).collect()
s = s.join(at, on="post_id").join(at.rename({"post_id": "post_id_b", "at": "at_b"}), on="post_id_b")
period = pl.when(pl.col("created_at_ms_b") < cut).then(pl.lit("before")).when(pl.col("created_at_ms") >= cut).then(pl.lit("after")).otherwise(pl.lit("straddles"))
s = s.with_columns(period.alias("period"), ((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log()).alias("d"))
rep: dict = {}


def count(name: str, f: pl.DataFrame) -> None:
    c = {r["period"]: r["len"] for r in f.group_by("period").len().iter_rows(named=True)}
    g = f.filter(pl.col("d").abs() >= np.log(1.25))
    rep[name] = {"pairs": f.height, "before_cutoff": c.get("before", 0), "after_cutoff": c.get("after", 0), "straddling": c.get("straddles", 0), "accounts": f["author_id"].n_unique(),
                 "likes_differ_1.25x": g.height, "likes_differ_2x": f.filter(pl.col("d").abs() >= np.log(2)).height,
                 "before_cutoff_1.25x": g.filter(pl.col("period") == "before").height}
    print(name, rep[name])


count("0 same link or quoted post, 48 hours (benchmark definition)", s.filter(pl.col("tac").is_not_null() & ((pl.col("created_at_ms_b") - pl.col("created_at_ms")) <= 48 * 3_600_000)))
count("0b same link or quoted post, 14 days", s.filter(pl.col("tac").is_not_null()))
for lo in (0.80, 0.85, 0.90, 0.95):
    count(f"1 candidates with cosine >= {lo:.2f}, texts differ", s.filter(pl.col("cos") >= lo))
k = s.filter((pl.col("nw") >= 6) & (pl.col("nw_b") >= 6) & ~pl.col("at") & ~pl.col("at_b"))
k = k.with_columns(pl.when(pl.col("tac").is_not_null()).then(pl.lit("tac")).when(pl.col("cos") >= 0.90).then(pl.lit("strict")).when(pl.col("cos") >= args.cos_min).then(pl.lit("loose")).alias("tier"))
k = k.filter(pl.col("tier").is_not_null())
count("2 after length and reply filters, cosine >= %.2f or shared link/quote" % args.cos_min, k)
m = k.filter(pl.col("tier") != "tac")
deg = pl.concat([m["post_id"], m["post_id_b"]]).value_counts().rename({"count": "deg"})
k = k.join(deg, on="post_id", how="left").join(deg.rename({"post_id": "post_id_b", "deg": "deg_b"}), on="post_id_b", how="left")
k = k.filter((pl.col("tier") == "tac") | ((pl.col("deg") <= args.max_degree) & (pl.col("deg_b") <= args.max_degree)))
count("3 after dropping template posts (in more than %d mined pairs)" % args.max_degree, k)
k = k.sample(fraction=1.0, shuffle=True, seed=0).with_columns(pl.int_range(pl.len()).over("author_id", "period").alias("r")).filter(pl.col("r") < args.max_per_account).drop("r")
count("4 after the cap of %d pairs per account and period" % args.max_per_account, k)
for t in ("tac", "strict", "loose"):
    count(f"4 tier {t}", k.filter(pl.col("tier") == t))
b = pl.read_parquet(args.benchmark)
bposts = pl.concat([b["post_id"], b["post_id_b"]]).unique()
cal_auth = b.filter(pl.col("part") == "calibration")["author_id"].unique()
tr = k.filter((pl.col("period") == "before") & ~pl.col("post_id").is_in(bposts.implode()) & ~pl.col("post_id_b").is_in(bposts.implode()) & ~pl.col("author_id").is_in(cal_auth.implode()))
count("5 training pool: before the cutoff, no benchmark post, no benchmark calibration account", tr)
for t in ("tac", "strict", "loose"):
    count(f"5 tier {t}", tr.filter(pl.col("tier") == t))
keep = ["post_id", "post_id_b", "author_id", "cos", "lex", "tac", "tier", "likes", "likes_b", "created_at_ms", "created_at_ms_b", "m", "m_b", "kind", "kind_b", "period"]
tr.select(keep).write_parquet(f"{args.dir}/train_pool.parquet")
k.filter(pl.col("period") == "after").select(keep).write_parquet(f"{args.dir}/after_cutoff_pairs.parquet")
k.select(keep).write_parquet(f"{args.dir}/selected_all.parquet")
json.dump(rep, open(f"{args.dir}/pair_counts.json", "w"), indent=1)
