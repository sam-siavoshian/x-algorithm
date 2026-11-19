#!/usr/bin/env python3
"""Selective track: the serving path (DraftScorer with selective_inputs, as the advisor runs it) must give the rule the same
inputs as the offline feature tables. Scores a few benchmark pairs with the baseline three and compares feature by feature."""
import math, pickle, sys
import numpy as np, polars as pl
sys.path.insert(0, "scripts/selective")
from sel_score import COLS
from xviral import selective as S
from xviral.serve import DraftScorer
N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
U = pl.read_parquet("data/sel/units_bench.parquet").sort("unit_id").with_row_index("_r").filter(pl.col("_r") % 4 == 0).head(N)
F = pl.read_parquet("data/sel/feat_bench_base3.parquet")
d = pl.scan_parquet("data/work/filtered/posts.parquet").select(COLS).filter(pl.col("author_id").is_in(U["author_id"].unique().to_list())).collect()
sc = DraftScorer(["data/runs/ts46_s1/best.pt", "data/runs/ts46_s2/best.pt", "data/work/model/compare_v2/full/best.pt"], lambda t: (_ for _ in ()).throw(AssertionError("cache miss")),
                 sid_codebook="data/work/emb/sid_codebook.npz")
sc.feature_cache, sc.selective_inputs = pickle.load(open("data/bench3/cache_0.pkl", "rb")), True
worst, n = {}, 0
for r in U.iter_rows(named=True):
    g = d.filter(pl.col("author_id") == r["author_id"]).sort("created_at_ms")
    xs = [g.filter(pl.col("post_id") == p).row(0, named=True) for p in r["post_ids"]]
    hist = g.filter(pl.col("created_at_ms") < r["when_ms"]).tail(56)
    acct = {"author_id": r["author_id"], "handle": xs[0]["author_handle"], "display_name": xs[0]["author_display_name"], "followers": xs[0]["author_followers"], "following": xs[0]["author_following"]}
    ra, rb = sc.score(hist, [{"text": x["text"], "kind": x["kind"], "media": x["media"], "urls": x["urls"], "quoted_text": x["quoted_text"], "aux": {}} for x in xs], acct, r["when_ms"])
    f = S.pair_features(S.version_from_report(ra), S.version_from_report(rb))
    off = F.filter((pl.col("post_id") == r["post_ids"][0]) & (pl.col("post_id_b") == r["post_ids"][1]))
    if not off.height:
        continue
    n += 1
    for k, v in f.items():
        o = off["f_" + k][0]
        if k in ("ovl_hist", "ovl_mix", "entropy", "o_gap", "o_sd", "o_agree", "d_hashtags", "d_mentions"):      # histograms and the second ensemble are not passed by the product
            continue
        if o is None or (isinstance(o, float) and math.isnan(o)):
            assert math.isnan(v), (k, v, o)
            continue
        worst[k] = max(worst.get(k, 0.0), abs(v - o))
bad = {k: v for k, v in worst.items() if v > 2e-3}
print(n, "pairs compared,", len(worst), "features; largest difference", max(worst.values()), "| features off by more than 0.002:", bad)
