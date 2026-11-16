#!/usr/bin/env python3
"""Measurements around the launch post that are not effects on the launch itself: how the follow-up and teaser posts do,
and how much of launch-day quoting the corpus can see.   Writes data/marketing/context_extra.json."""
import json, sys, numpy as np, polars as pl
sys.path.insert(0, "scripts/marketing")
from traits import TEASER
rng = np.random.default_rng(0)
T = pl.read_parquet("data/marketing/table.parquet").filter(pl.col("source") == "community_archive")
LAUNCH = ["launch", "open_source"]
L = T.filter(pl.col("ptype").is_in(LAUNCH)).select(pl.col("post_id").alias("lid"), pl.col("author_id"), pl.col("created_at_ms").alias("lms"), pl.col("lift_likes").alias("llift"), "akind")
def ci(x, g):
    u, inv = np.unique(g, return_inverse=True); s = np.zeros(len(u)); c = np.zeros(len(u)); np.add.at(s, inv, x); np.add.at(c, inv, 1)
    idx = rng.integers(0, len(u), (2000, len(u))); b = s[idx].sum(1) / c[idx].sum(1)
    return {"ratio": round(float(np.exp(x.mean())), 3), "lo": round(float(np.exp(np.quantile(b, .025))), 3), "hi": round(float(np.exp(np.quantile(b, .975))), 3), "posts": int(len(x)), "accounts": int(len(u)), "beat_own_median": round(float((x > 0).mean()), 4)}
out = {}
# 1. the account quote-posting its own launch (follow-up): how the follow-up post itself does
sq = T.filter(pl.col("quoted_post_id").is_not_null()).join(L.rename({"author_id": "la"}), left_on="quoted_post_id", right_on="lid").filter(pl.col("author_id") == pl.col("la")).with_columns(((pl.col("created_at_ms") - pl.col("lms")) / 3_600_000).alias("dh"))
sq = sq.filter(pl.col("lift_likes").is_not_null() & (pl.col("dh") > 0))
uq = T.filter((pl.col("kind") == "quote") & pl.col("lift_likes").is_not_null()).join(sq.select("post_id"), on="post_id", how="anti").group_by("author_id").agg(pl.col("lift_likes").mean().alias("uq"), pl.len().alias("nq"))
for nm, f in (("within_24h", pl.col("dh") <= 24), ("day_2_to_7", pl.col("dh").is_between(24, 168)), ("after_7_days", pl.col("dh") > 168)):
    z = sq.filter(f)
    if z.height >= 15:
        r = {"vs_own_median": ci(z["lift_likes"].to_numpy(), z["author_id"].to_numpy())}
        z2 = z.join(uq, on="author_id").filter(pl.col("nq") >= 5)
        if z2.height >= 15: r["vs_own_other_quote_posts"] = ci((z2["lift_likes"] - z2["uq"]).to_numpy(), z2["author_id"].to_numpy())
        z3 = z.filter(pl.col("llift").is_not_null())
        if z3.height >= 15: r["likes_of_followup_over_likes_of_launch"] = round(float(np.exp(np.median(np.log(z3["likes"].to_numpy() + 1) - (z3["llift"].to_numpy() + np.log(z3["med_likes"].to_numpy() + 1))))), 3)
        out["own_quote_of_own_launch_" + nm] = r
# 2. teaser posts themselves (the account's posts matching the teaser pattern 1 to 72 h before one of its launch posts)
lc = pl.col("text").fill_null("").str.replace_all(r"https?://\S+", " ").str.to_lowercase().str.replace_all("’", "'")
j = L.join(T.select("author_id", "post_id", "created_at_ms", "text", "lift_likes"), on="author_id").with_columns(((pl.col("created_at_ms") - pl.col("lms")) / 3_600_000).alias("dh")).filter(pl.col("dh").is_between(-72, -1) & lc.str.contains(TEASER) & pl.col("lift_likes").is_not_null()).unique("post_id")
out["teaser_posts_vs_own_median"] = ci(j["lift_likes"].to_numpy(), j["author_id"].to_numpy())
# 3. the posts after a launch: any own post 1 to 4 days later against the account's usual (does a launch lift what follows?)
k = L.filter(pl.col("akind") != "other").join(T.select("author_id", "post_id", "created_at_ms", "lift_likes", "ptype"), on="author_id").with_columns(((pl.col("created_at_ms") - pl.col("lms")) / 3_600_000).alias("dh")).filter(pl.col("lift_likes").is_not_null() & ~pl.col("ptype").is_in(LAUNCH))
for nm, a, b in (("1_to_24h_after", 1, 24), ("day_2_to_4_after", 24, 96), ("day_5_to_7_after", 96, 168), ("day_2_to_4_before", -96, -24)):
    z = k.filter(pl.col("dh").is_between(a, b)).unique("post_id")
    out["other_own_posts_" + nm] = ci(z["lift_likes"].to_numpy(), z["author_id"].to_numpy())
# 4. coverage of quoting
S = T.filter(pl.col("ptype").is_in(LAUNCH) & pl.col("lift_likes").is_not_null() & (pl.col("akind") != "other"))
out["coverage"] = {"launch_posts": S.height, "quoted_by_a_corpus_account_within_1h": int(S["quoted_1h"].sum()), "within_48h": int(S["quoted_48h"].sum()), "same_team_within_1h": int(S["team_quoted_1h"].sum()), "same_team_within_24h": int(S["team_quoted_24h"].sum()),
                   "note": "only quote posts by accounts that are themselves in the corpus are visible; a launch with no visible quote may still have been quoted"}
# 5. dose: more first-hour quoters, same account
d = S.with_columns(pl.col("quotes_1h").clip(0, 3).alias("q"))
per = d.group_by("author_id").agg(pl.len().alias("n"), pl.col("q").n_unique().alias("u")).filter(pl.col("u") > 1)
dd = d.join(per.select("author_id"), on="author_id", how="semi").with_columns((pl.col("lift_likes") - pl.col("lift_likes").mean().over("author_id")).alias("dl"))
out["first_hour_quoters_dose_within_account"] = {str(r["q"]) + ("+" if r["q"] == 3 else ""): {"posts": r["n"], "likes_vs_account_mean_launch": round(float(np.exp(r["m"])), 3)} for r in dd.group_by("q").agg(pl.len().alias("n"), pl.col("dl").mean().alias("m")).sort("q").iter_rows(named=True)}
json.dump(out, open("data/marketing/context_extra.json", "w"), indent=1)
print(json.dumps(out, indent=1))
