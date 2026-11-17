#!/usr/bin/env python3
"""The level bias of the home-timeline scrape, measured from data (no model).

    python scripts/scrape/levels.py        # data/scrape_track/levels.json

1. Selection. For accounts that are also in the corpus's Community Archive rows during the scrape window, compare the
   posts the home timeline showed (in the scrape) with the account's other archive posts of the same weeks, on the
   archive's later, mature like count. Inside accounts; accounts resampled.
2. Immaturity. The same post read by the scrape and later by the archive: growth of likes by age at capture.
3. Net level against the corpus's view-labelled posts (a search sample), by follower tier and capture age.
4. Posts captured days after posting with almost no likes per view (paid distribution).
"""
import json
from pathlib import Path
import numpy as np, polars as pl
from xviral.util.core import date_to_ms

OUT = Path("data/scrape_track"); R = {}
rng = np.random.default_rng(0)
LOW = pl.col("author_handle").str.to_lowercase().alias("handle")
s = pl.read_parquet(OUT / "posts.parquet", columns=["post_id", "author_handle", "views", "likes", "snapshot_at_ms", "created_at_ms", "author_followers", "kind"]).with_columns(LOW, ((pl.col("snapshot_at_ms") - pl.col("created_at_ms")) / 3.6e6).alias("age_h")).filter(pl.col("kind") == "original")
w0, w1 = s["snapshot_at_ms"].min(), s["snapshot_at_ms"].max()
f = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id", LOW, "source", "likes", "views", "created_at_ms", "snapshot_at_ms", "author_followers", "kind").collect()
ca = f.filter((pl.col("source") == "community_archive") & (pl.col("created_at_ms") >= w0 - 86_400_000) & (pl.col("created_at_ms") <= w1 - 3 * 86_400_000) & ((pl.col("snapshot_at_ms") - pl.col("created_at_ms")) >= 72 * 3.6e6))
ca = ca.with_columns(pl.col("post_id").is_in(s["post_id"].implode()).alias("shown"))
R["archive_posts_in_window"] = {"posts": ca.height, "accounts": ca["handle"].n_unique(), "shown_in_scrape": int(ca["shown"].sum()),
                                "accounts_with_a_shown_post": ca.filter(pl.col("shown"))["handle"].n_unique()}
g = ca.group_by("handle").agg(pl.len().alias("n"), pl.col("shown").sum().alias("k"),
                              (pl.col("likes") + 1).log10().filter(pl.col("shown")).mean().alias("m_shown"), (pl.col("likes") + 1).log10().filter(~pl.col("shown")).mean().alias("m_not"),
                              (pl.col("likes") + 1).log10().filter(pl.col("shown")).median().alias("md_shown"), (pl.col("likes") + 1).log10().filter(~pl.col("shown")).median().alias("md_not"))
both = g.filter((pl.col("k") >= 2) & (pl.col("n") - pl.col("k") >= 5))
d, w = (both["m_shown"] - both["m_not"]).to_numpy(), np.minimum(both["k"].to_numpy(), (both["n"] - both["k"]).to_numpy()).astype(float)
bs = [10 ** (np.sum(d[i] * w[i]) / w[i].sum()) for i in (rng.integers(0, len(d), len(d)) for _ in range(2000))]
R["shown_vs_not_shown_same_account_mature_likes"] = {"accounts": both.height, "shown_posts": int(both["k"].sum()), "not_shown_posts": int((both["n"] - both["k"]).sum()),
    "ratio_weighted": float(10 ** (np.sum(d * w) / w.sum())), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
    "median_account_ratio": float(10 ** np.median(d)), "share_of_accounts_where_shown_is_higher": float((d > 0).mean()),
    "share_of_archive_posts_shown": float(both["k"].sum() / both["n"].sum())}
# The archive is itself a sample of posts that travelled; the part of an account it holds is already its better posts.

# 2. growth of the same post
j = s.join(f.filter(pl.col("source") == "community_archive").select("post_id", pl.col("likes").alias("likes_c"), pl.col("snapshot_at_ms").alias("snap_c")), on="post_id")
j = j.filter((pl.col("snap_c") - pl.col("created_at_ms") >= 72 * 3.6e6) & (pl.col("likes_c") >= 5)).with_columns(((pl.col("likes_c") + 1) / (pl.col("likes") + 1)).log10().alias("g"))
cur = []
for lo, hi in ((0, 1), (1, 3), (3, 6), (6, 12), (12, 24), (24, 72), (72, 1e9)):
    b = j.filter((pl.col("age_h") >= lo) & (pl.col("age_h") < hi))
    if b.height >= 20:
        cur.append({"age_h": f"{lo}-{hi}", "posts": b.height, "final_over_captured_likes_median": float(10 ** b["g"].median()), "p25": float(10 ** b["g"].quantile(0.25)), "p75": float(10 ** b["g"].quantile(0.75)),
                    "share_within_1.25x_of_final": float((b["g"].abs() < np.log10(1.25)).mean()), "sd_log10": float(b["g"].std())})
R["likes_growth_same_post"] = cur
growth = {c["age_h"]: c["final_over_captured_likes_median"] for c in cur}

# 3. level against the corpus's view-labelled posts
names = {1: "under 100", 2: "100-1k", 3: "1k-10k", 4: "10k-100k", 5: "100k-1M", 6: "1M+"}
tier = lambda c: pl.col(c).clip(lower_bound=1).log10().floor().clip(1, 6).cast(pl.Int32)
cv = f.filter(pl.col("views").is_not_null() & (pl.col("author_followers") > 0) & (pl.col("views") > 0)).with_columns(tier("author_followers").alias("tier"), pl.col("views").log10().alias("lv"), (pl.col("likes") + 1).log10().alias("ll"))
sv = s.filter((pl.col("author_followers") > 0) & (pl.col("views") > 0)).with_columns(tier("author_followers").alias("tier"), pl.col("views").log10().alias("lv"), (pl.col("likes") + 1).log10().alias("ll"),
                                                                              (pl.col("likes") / pl.col("views")).alias("lpv"))
organic = sv.filter(~((pl.col("age_h") >= 48) & (pl.col("lpv") < 0.003)))
tab = []
for t in range(2, 7):
    c = cv.filter(pl.col("tier") == t); a = organic.filter(pl.col("tier") == t)
    row = {"tier": names[t], "corpus_posts": c.height, "corpus_median_views": float(10 ** c["lv"].median()), "scrape_posts": a.height, "scrape_median_views_as_read": float(10 ** a["lv"].median()),
           "as_read_over_corpus": float(10 ** (a["lv"].median() - c["lv"].median()))}
    for lo, hi, nm in ((0, 1, "0-1h"), (1, 6, "1-6h"), (6, 12, "6-12h"), (12, 24, "12-24h"), (24, 72, "24-72h")):
        b = a.filter((pl.col("age_h") >= lo) & (pl.col("age_h") < hi))
        row[f"over_corpus_{nm}"] = float(10 ** (b["lv"].median() - c["lv"].median())) if b.height >= 30 else None
        row[f"posts_{nm}"] = b.height
    tab.append(row)
R["views_level_vs_corpus_view_posts_by_tier"] = tab
R["likes_level_vs_archive_by_tier"] = []
cl = f.filter((pl.col("source") == "community_archive") & (pl.col("author_followers") > 0)).with_columns(tier("author_followers").alias("tier"), (pl.col("likes") + 1).log10().alias("ll"))
for t in range(2, 7):
    c = cl.filter(pl.col("tier") == t); a = organic.filter(pl.col("tier") == t)
    row = {"tier": names[t], "archive_median_likes": float(10 ** c["ll"].median() - 1), "scrape_median_likes_as_read": float(10 ** a["ll"].median() - 1)}
    for lo, hi, nm in ((12, 24, "12-24h"), (24, 72, "24-72h")):
        b = a.filter((pl.col("age_h") >= lo) & (pl.col("age_h") < hi))
        row[f"scrape_{nm}_over_archive"] = float(10 ** (b["ll"].median() - c["ll"].median())) if b.height >= 30 else None
    R["likes_level_vs_archive_by_tier"].append(row)

# 4. paid distribution
old = sv.filter(pl.col("age_h") >= 48)
R["captured_48h_or_later"] = {"posts": old.height, "likes_per_view_under_0.003": int((old["lpv"] < 0.003).sum()), "median_views_of_those": float(old.filter(pl.col("lpv") < 0.003)["views"].median()),
                              "median_likes_per_view_all_scrape": float(sv["lpv"].median()), "accounts_of_those": old.filter(pl.col("lpv") < 0.003)["handle"].n_unique(),
                              "top_accounts": old.filter(pl.col("lpv") < 0.003).group_by("handle").len().sort("len", descending=True).head(15).to_dicts()}
(OUT / "levels.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1))
