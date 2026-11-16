#!/usr/bin/env python3
"""What launch posts do for accounts of each size: the product's "accounts of your size" line.

    python scripts/launch/by_size.py [launch_posts.parquet]     # the table of scripts/launch/build_table.py (dataset: tracks/launch/)

Same yardstick and same resampling as the type table of effects.py: likes of a launch post divided by the median
likes of the account's previous 30 posts, company and founder accounts, intervals from resampling accounts (2,000
draws). Adds the raw like counts (median and middle half) per follower tier. A tier with fewer than 50 posts is
written with `too_thin` and the product does not quote it. Writes xviral/data/launch_by_size.json.
"""
import json
import sys

import numpy as np
import polars as pl

T = pl.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "data/tracks/launch/launch_posts.parquet")
z = T.filter((pl.col("akind") != "other") & (pl.col("ptype") == "launch") & pl.col("lift_likes").is_not_null())
rng = np.random.default_rng(0)
TIERS = [("<1k", 0, 1_000), ("1k-10k", 1_000, 10_000), ("10k-100k", 10_000, 100_000), ("100k+", 100_000, None)]


def measure(q: pl.DataFrame) -> dict:
    per = q.group_by("author_id").agg(pl.col("lift_likes").mean().alias("m"), (pl.col("lift_likes") > 0).mean().alias("b"), pl.col("top10_likes").mean().alias("t"),
                                      pl.len().alias("n")).sort("author_id")
    m, b, t, n = (per[c].to_numpy().astype(float) for c in ("m", "b", "t", "n"))
    idx = rng.integers(0, len(m), (2000, len(m)))
    w = lambda v, i: np.sum(v[i] * n[i]) / np.sum(n[i])           # noqa: E731
    ci = lambda v: [float(x) for x in np.quantile([w(v, i) for i in idx], [0.025, 0.975])]   # noqa: E731
    all_ = np.arange(len(m))
    (l1, h1), (l2, h2), (l3, h3) = ci(m), ci(b), ci(t)
    likes = q["likes"].to_numpy().astype(float)
    return {"posts": q.height, "accounts": per.height, "likes_vs_usual": round(float(np.exp(w(m, all_))), 3), "lo": round(float(np.exp(l1)), 3), "hi": round(float(np.exp(h1)), 3),
            "beat_usual": round(float(w(b, all_)), 4), "beat_lo": round(l2, 4), "beat_hi": round(h2, 4), "own_top_10pct": round(float(w(t, all_)), 4), "top10_lo": round(l3, 4), "top10_hi": round(h3, 4),
            "likes_median": float(np.median(likes)), "likes_p25": float(np.quantile(likes, 0.25)), "likes_p75": float(np.quantile(likes, 0.75))}


out = {"_meta": {"built": "2026-11-16", "source": "tracks/launch/launch_posts.parquet (scripts/launch/build_table.py)", "posts": z.height, "accounts": z["author_id"].n_unique(),
                 "scope": "launch posts of company and founder accounts with a like baseline (previous 30 posts)", "yardstick": "likes divided by the median likes of the account's previous 30 posts",
                 "intervals": "95%, resampling accounts, 2,000 draws", "tier_by": "largest follower count seen for the account"}, "tiers": []}
out["all"] = measure(z)
for name, lo, hi in TIERS:
    q = z.filter(pl.col("tier") == name)
    e = {"tier": name, "followers_from": lo, "followers_below": hi, **(measure(q) if q.height >= 50 else {"posts": q.height, "accounts": q["author_id"].n_unique(), "too_thin": True})}
    out["tiers"].append(e)
    print(name, json.dumps(e))
unk = z.filter(pl.col("tier") == "unknown")
out["_meta"]["posts_follower_count_unknown"] = unk.height
print("all", json.dumps(out["all"]), "unknown tier posts", unk.height)
json.dump(out, open("xviral/data/launch_by_size.json", "w"), indent=1)
