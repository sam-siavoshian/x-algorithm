#!/usr/bin/env python3
"""The same launch on the company account and on a team member's account.
A team member = a person account that quote-posts a company account's own announcements at least twice in the first
person (we / our). Pairs: the company's launch post and that person's quote of it within 7 days.
Writes data/launch/founder_vs_company.json."""
import json, numpy as np, polars as pl
rng = np.random.default_rng(0)
T = pl.read_parquet("data/launch/table.parquet")
A = pl.read_parquet("data/launch/accounts.parquet").select("author_id", "akind", "person", "handle", pl.col("f").alias("followers"))
L = T.filter(pl.col("ptype").is_in(["launch", "open_source", "funding", "feature"]) & (pl.col("akind") == "company"))
allp = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id", "author_id", "author_handle", "created_at_ms", "quoted_post_id", "text", "likes", "author_followers").filter(pl.col("quoted_post_id").is_not_null()).collect()
q = allp.join(L.select(pl.col("post_id").alias("quoted_post_id"), pl.col("author_id").alias("co"), pl.col("author_handle").alias("co_handle"), pl.col("created_at_ms").alias("co_ms"), pl.col("likes").alias("co_likes"), pl.col("lift_likes").alias("co_lift"),
                       pl.col("followers_max").alias("co_followers"), "ptype"), on="quoted_post_id").filter(pl.col("author_id") != pl.col("co"))
fp = pl.col("text").fill_null("").str.to_lowercase().str.replace_all("’", "'").str.contains(r"\b(we|we're|we've|our)\b")
q = q.with_columns(fp.alias("fp"), ((pl.col("created_at_ms") - pl.col("co_ms")) / 3_600_000).alias("lag_h")).join(A.select("author_id", "person", pl.col("akind").alias("q_akind")), on="author_id", how="left")
team = q.filter(pl.col("fp")).group_by("author_id", "co").len().filter(pl.col("len") >= 2).select("author_id", "co")
p = q.join(team, on=["author_id", "co"], how="semi").filter((pl.col("lag_h") >= 0) & (pl.col("lag_h") <= 168) & (pl.col("q_akind").fill_null("other") != "company"))
hist = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id", "author_id", "created_at_ms", "likes").collect().join(p.select("author_id").unique(), on="author_id", how="semi").sort("author_id", "created_at_ms")
hist = hist.with_columns(pl.col("likes").cast(pl.Float64).shift(1).rolling_median(window_size=30, min_samples=8).over("author_id").alias("med")).with_columns(((pl.col("likes") + 1) / (pl.col("med") + 1)).log().alias("lift"))
p = p.join(hist.select("post_id", "lift"), on="post_id", how="left")
fmax = allp.group_by("author_id").agg(pl.col("author_followers").max().alias("q_followers"))
p = p.join(fmax, on="author_id", how="left")
print("team-member quote of a company announcement:", p.height, "pairs;", p["author_id"].n_unique(), "team accounts;", p["co"].n_unique(), "company accounts")
def ci(x, g, f=np.mean):
    u, inv = np.unique(g, return_inverse=True); rows = [np.where(inv == k)[0] for k in range(len(u))]
    b = [f(x[np.concatenate([rows[k] for k in rng.integers(0, len(u), len(u))])]) for _ in range(1000)]
    return float(f(x)), float(np.quantile(b, .025)), float(np.quantile(b, .975))
out = {"pairs": p.height, "team_accounts": int(p["author_id"].n_unique()), "company_accounts": int(p["co"].n_unique()), "definition": __doc__.split("\n")[1:4]}
raw = (np.log(p["likes"].to_numpy() + 1) - np.log(p["co_likes"].to_numpy() + 1)); g = p["co"].to_numpy()
m, lo, hi = ci(raw, g, np.median); w, wl, wh = ci((raw > 0).astype(float), g)
out["raw_likes_team_quote_over_company_post"] = {"median_ratio": round(float(np.exp(m)), 3), "lo": round(float(np.exp(lo)), 3), "hi": round(float(np.exp(hi)), 3), "team_quote_gets_more_likes": round(w, 4), "share_lo": round(wl, 4), "share_hi": round(wh, 4)}
z = p.filter(pl.col("lift").is_not_null())
m, lo, hi = ci(z["lift"].to_numpy(), z["author_id"].to_numpy())
out["team_quote_vs_own_median"] = {"ratio": round(float(np.exp(m)), 3), "lo": round(float(np.exp(lo)), 3), "hi": round(float(np.exp(hi)), 3), "posts": z.height, "accounts": int(z["author_id"].n_unique()), "beat_own_median": round(float((z["lift"] > 0).mean()), 4)}
z2 = z.filter(pl.col("co_lift").is_not_null())
m, lo, hi = ci((z2["lift"] - z2["co_lift"]).to_numpy(), z2["co"].to_numpy())
out["lift_team_quote_over_lift_company_post"] = {"ratio": round(float(np.exp(m)), 3), "lo": round(float(np.exp(lo)), 3), "hi": round(float(np.exp(hi)), 3), "pairs": z2.height}
f = p.filter(pl.col("q_followers").is_not_null() & pl.col("co_followers").is_not_null())
if f.height >= 20:
    big = f.filter(pl.col("q_followers") > pl.col("co_followers")); r2 = np.log(big["likes"].to_numpy() + 1) - np.log(big["co_likes"].to_numpy() + 1)
    out["when_team_member_has_more_followers_than_company"] = {"pairs": big.height, "team_quote_gets_more_likes": None if big.height == 0 else round(float((r2 > 0).mean()), 4), "median_ratio": None if big.height == 0 else round(float(np.exp(np.median(r2))), 3)}
    out["pairs_with_both_follower_counts"] = f.height
print(json.dumps(out, indent=1))
json.dump(out, open("data/launch/founder_vs_company.json", "w"), indent=1)
for r in p.sample(min(10, p.height), seed=2).iter_rows(named=True):
    print(f"[{r['author_handle']} -> {r['co_handle']} | {r['likes']} vs {r['co_likes']}]", r["text"][:110].replace("\n", " / "))
