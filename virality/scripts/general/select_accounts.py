#!/usr/bin/env python3
"""The account sample of the general track. The test bed is drawn FIRST, from its own random stream, so it does not depend
on how much training data is added afterwards.

    python scripts/general/select_accounts.py [--train-tw 600] [--train-nitter 2500]

Eligible: at least 30 English originals or quotes in the source's window, not in work/dataset/rows.parquet (no baseline
model trained on it), not in the expand track's rows, not an account of a benchmark calibration pair or a development pair.
Roles: bed (test bed, never trained or selected on), val (model selection and fitting the level correction), train.
Brand accounts: every brand account outside the expand track's training split is bed (`seen_by_expand` false); 25 of the
expand track's training brands are bed too (held out of every model trained here, `seen_by_expand` true) and 10 are val.
"""
import argparse
import numpy as np, polars as pl
ap = argparse.ArgumentParser()
ap.add_argument("--train-tw", type=int, default=600, help="training accounts per follower tier from twitter100m")
ap.add_argument("--train-nitter", type=int, default=2500)
args = ap.parse_args()
st = pl.read_parquet("data/general/account_stats.parquet")
filt = pl.scan_parquet("data/work/filtered/posts.parquet").select("author_id", pl.col("author_handle").str.to_lowercase().alias("handle")).unique().collect()
bench = pl.read_parquet("data/tracks/pairs/benchmark_pairs.parquet").filter(pl.col("part") == "calibration")["author_id"].unique()
dev = pl.read_parquet("data/tracks/replicate/dev_pairs.parquet")["author_id"].unique()
block = filt.filter(pl.col("author_id").is_in(pl.concat([bench, dev]).unique().implode()))["handle"].unique()
fol = pl.coalesce("followers", "tw_followers")
tier = (pl.when(fol.is_null() | (fol <= 0)).then(pl.lit("unknown")).when(fol < 1000).then(pl.lit("under_1k")).when(fol < 5000).then(pl.lit("1k_5k"))
        .when(fol < 10000).then(pl.lit("5k_10k")).when(fol < 100000).then(pl.lit("10k_100k")).otherwise(pl.lit("100k_plus")))
st = st.with_columns(fol.alias("fol"), tier.alias("tier"), pl.col("handle").is_in(block.implode()).alias("blocked"))
elig = st.filter((pl.col("n_en") >= 30) & ~pl.col("in_rows") & ~pl.col("in_expand") & ~pl.col("blocked"))
print("eligible:", dict(elig.group_by("src").len().iter_rows()), "blocked handles", block.len())
TIERS = ["under_1k", "1k_5k", "5k_10k", "10k_100k", "100k_plus"]
picked: list[pl.DataFrame] = []
taken: set = set()

def draw(pool: pl.DataFrame, n: int, rng, role: str, **extra) -> None:
    pool = pool.filter(~pl.col("handle").is_in(list(taken))).sort("handle")
    idx = rng.permutation(pool.height)[:n]
    d = pool[np.sort(idx).tolist()].with_columns(pl.lit(role).alias("role"), *[pl.lit(v).alias(k) for k, v in extra.items()])
    taken.update(d["handle"].to_list())
    picked.append(d)

# --- test bed: its own generator ---
rb = np.random.default_rng(20261117)
tw, ni, zb = (elig.filter(pl.col("src") == s) for s in ("twitter100m", "nitter_2026_replies", "zenodo_brands"))
niv = ni.filter(pl.col("n_views") >= 30)
# 2026 view-corpus accounts with a known follower count under 10k (the count is twitter100m's 2023 reading): all of them, as the
# small-account slice with views. They are taken before the twitter100m draw, so each account is in the bed under one source.
draw(niv.filter(pl.col("tier").is_in(["under_1k", "1k_5k", "5k_10k"])), 10_000, rb, "bed")
for t in TIERS:
    draw(tw.filter(pl.col("tier") == t), 100, rb, "bed")
ni = niv.filter(~pl.col("handle").is_in(tw["handle"].implode()))      # the rest: an account is sampled from one source
draw(ni.filter(pl.col("n_en") >= 40), 420, rb, "bed")
zall = st.filter((pl.col("src") == "zenodo_brands") & (pl.col("n_en") >= 30) & ~pl.col("in_rows") & ~pl.col("blocked"))
draw(zall.filter(~pl.col("in_expand_train")), 1000, rb, "bed", seen_by_expand=False)
draw(zall.filter(pl.col("in_expand_train")), 25, rb, "bed", seen_by_expand=True)
# --- validation, then training: a second generator ---
rt = np.random.default_rng(7)
draw(zall.filter(pl.col("in_expand_train")), 10, rt, "val", seen_by_expand=True)
for t in TIERS:
    draw(tw.filter(pl.col("tier") == t), 60, rt, "val")
draw(ni.filter(pl.col("n_en") >= 40), 300, rt, "val")
for t in TIERS:
    draw(tw.filter(pl.col("tier") == t), args.train_tw, rt, "train")
draw(ni, args.train_nitter, rt, "train")
S = pl.concat(picked, how="diagonal_relaxed").with_columns(pl.col("seen_by_expand").fill_null(False))
assert S["handle"].n_unique() == S.height
S.write_parquet("data/general/accounts.parquet")
pl.Config.set_tbl_rows(60)
print(S.group_by("role", "src", "tier").agg(pl.len().alias("accounts"), pl.col("n_en").sum().alias("posts_available")).sort("role", "src", "tier"))
print(S.group_by("role", "src").agg(pl.len().alias("accounts"), pl.col("seen_by_expand").sum()).sort("role", "src"))
