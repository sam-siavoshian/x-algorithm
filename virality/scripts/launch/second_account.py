#!/usr/bin/env python3
"""Launch posts quote-posted from a second account, and founder account against company account.
Writes data/launch/second_account.json."""
import json, numpy as np, polars as pl
rng = np.random.default_rng(0)
T = pl.read_parquet("data/launch/table.parquet")
L = T.filter(pl.col("ptype").is_in(["launch", "open_source", "funding"]))
# every corpus post that quotes one of these launch posts, from another account
allp = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id", "author_id", "author_handle", "created_at_ms", "quoted_post_id", "text", "likes", "views").filter(pl.col("quoted_post_id").is_not_null()).collect()
q = allp.join(L.select(pl.col("post_id").alias("quoted_post_id"), pl.col("author_id").alias("orig_author"), pl.col("author_handle").alias("orig_handle"), pl.col("created_at_ms").alias("orig_ms"), pl.col("likes").alias("orig_likes"),
                       pl.col("lift_likes").alias("orig_lift"), pl.col("akind").alias("orig_akind"), pl.col("person").alias("orig_person"), "ptype"), on="quoted_post_id")
q = q.filter(pl.col("author_id") != pl.col("orig_author")).with_columns(((pl.col("created_at_ms") - pl.col("orig_ms")) / 3_600_000).alias("lag_h"))
lc = pl.col("text").fill_null("").str.to_lowercase().str.replace_all("’", "'")
q = q.with_columns(lc.str.contains(r"\b(we|we're|we've|our|my team|i'm (so )?(excited|proud|thrilled)|i've been (working|building)|proud of (the|our) team|been working on this|what we('ve| have) been (building|working))\b").alias("first_person"))
print("quotes of launch posts by another corpus account:", q.height, "launches quoted:", q["quoted_post_id"].n_unique(), " within 48h:", q.filter(pl.col("lag_h") <= 48).height, " first-person (same team):", q.filter(pl.col("first_person") & (pl.col("lag_h") <= 168)).height)
# the quoter's own baseline: need trailing median for the quoting account (any account in the corpus)
aff = q.filter(pl.col("first_person") & (pl.col("lag_h") <= 168))
hist = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id", "author_id", "created_at_ms", "likes", "kind").collect().join(aff.select("author_id").unique(), on="author_id", how="semi").sort("author_id", "created_at_ms")
hist = hist.with_columns(pl.col("likes").cast(pl.Float64).shift(1).rolling_median(window_size=30, min_samples=8).over("author_id").alias("med"))
hist = hist.with_columns(((pl.col("likes") + 1) / (pl.col("med") + 1)).log().alias("lift"))
aff = aff.join(hist.select("post_id", "lift"), on="post_id", how="left")
qb = hist.filter((pl.col("kind") == "quote") & pl.col("lift").is_not_null()).join(aff.select("post_id"), on="post_id", how="anti").group_by("author_id").agg(pl.col("lift").mean().alias("usual_quote_lift"), pl.len().alias("nq"))
aff = aff.join(qb, on="author_id", how="left")
out = {"quotes_of_launches_by_another_account": q.height, "launch_posts_quoted": int(q["quoted_post_id"].n_unique()), "same_team_quotes": aff.height}
a = aff.filter(pl.col("lift").is_not_null())
def ci(x, g):
    u, inv = np.unique(g, return_inverse=True); s = np.zeros(len(u)); c = np.zeros(len(u)); np.add.at(s, inv, x); np.add.at(c, inv, 1)
    idx = rng.integers(0, len(u), (2000, len(u))); b = s[idx].sum(1) / c[idx].sum(1)
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)), int(len(x)), int(len(u))
if a.height >= 20:
    m, lo, hi, n, na = ci(a["lift"].to_numpy(), a["author_id"].to_numpy())
    out["same_team_quote_vs_quoter_own_median"] = {"ratio": round(float(np.exp(m)), 3), "lo": round(float(np.exp(lo)), 3), "hi": round(float(np.exp(hi)), 3), "posts": n, "accounts": na, "beat_own_median": round(float((a["lift"] > 0).mean()), 4)}
    print("same-team quote of a launch vs the quoting account's own median:", out["same_team_quote_vs_quoter_own_median"])
    b = a.filter(pl.col("usual_quote_lift").is_not_null() & (pl.col("nq") >= 5))
    if b.height >= 20:
        m, lo, hi, n, na = ci((b["lift"] - b["usual_quote_lift"]).to_numpy(), b["author_id"].to_numpy())
        out["same_team_quote_vs_quoter_usual_quote_post"] = {"ratio": round(float(np.exp(m)), 3), "lo": round(float(np.exp(lo)), 3), "hi": round(float(np.exp(hi)), 3), "posts": n, "accounts": na}
        print("... vs the same account's other quote posts:", out["same_team_quote_vs_quoter_usual_quote_post"])
    # quote (second account) against the original: raw likes and normalized
    c = a.filter(pl.col("orig_lift").is_not_null())
    if c.height >= 20:
        m, lo, hi, n, na = ci((c["lift"] - c["orig_lift"]).to_numpy(), c["orig_author"].to_numpy())
        r = (np.log(c["likes"].to_numpy() + 1) - np.log(c["orig_likes"].to_numpy() + 1))
        out["same_team_quote_vs_original"] = {"lift_ratio_quote_over_original": round(float(np.exp(m)), 3), "lo": round(float(np.exp(lo)), 3), "hi": round(float(np.exp(hi)), 3), "pairs": n, "original_accounts": na,
                                              "median_raw_likes_quote_over_original": round(float(np.exp(np.median(r))), 3), "quote_gets_more_raw_likes": round(float((r > 0).mean()), 4)}
        print("same-team quote against the original launch post:", out["same_team_quote_vs_original"])
# does a launch that a second account quoted do better than the same account's launches nobody in the corpus quoted? (reverse causation: good launches get quoted)
qq = q.filter(pl.col("lag_h") <= 48).select(pl.col("quoted_post_id").alias("post_id")).unique().with_columns(pl.lit(True).alias("quoted48"))
Lq = L.join(qq, on="post_id", how="left").with_columns(pl.col("quoted48").fill_null(False)).filter(pl.col("lift_likes").is_not_null())
per = Lq.group_by("author_id").agg(pl.col("lift_likes").filter(pl.col("quoted48")).mean().alias("m1"), pl.col("quoted48").sum().alias("n1"), pl.col("lift_likes").filter(~pl.col("quoted48")).mean().alias("m0"), (~pl.col("quoted48")).sum().alias("n0")).filter((pl.col("n1") > 0) & (pl.col("n0") > 0))
if per.height >= 10:
    dlt = (per["m1"] - per["m0"]).to_numpy(); w = (per["n1"] * per["n0"] / (per["n1"] + per["n0"])).to_numpy().astype(float)
    idx = rng.integers(0, len(w), (2000, len(w))); b = (w[idx] * dlt[idx]).sum(1) / w[idx].sum(1)
    out["launch_quoted_by_second_corpus_account_within_48h"] = {"ratio": round(float(np.exp((w * dlt).sum() / w.sum())), 3), "lo": round(float(np.exp(np.quantile(b, .025))), 3), "hi": round(float(np.exp(np.quantile(b, .975))), 3),
        "accounts": per.height, "posts_quoted": int(per["n1"].sum()), "posts_not_quoted": int(per["n0"].sum()), "note": "not a lever: a launch that is doing well attracts quotes"}
    print("launch quoted within 48h vs same account's unquoted launches:", out["launch_quoted_by_second_corpus_account_within_48h"])
# founder (person) account vs company account: lift of launch posts relative to own usual, by account kind
for nm, f in (("company", pl.col("akind") == "company"), ("founder", pl.col("akind") == "founder")):
    z = T.filter(f & pl.col("ptype").is_in(["launch", "open_source"]) & pl.col("lift_likes").is_not_null())
    m, lo, hi, n, na = ci(z["lift_likes"].to_numpy(), z["author_id"].to_numpy())
    out[f"launch_lift_{nm}_accounts"] = {"ratio": round(float(np.exp(m)), 3), "lo": round(float(np.exp(lo)), 3), "hi": round(float(np.exp(hi)), 3), "posts": n, "accounts": na, "median_raw_likes": float(z["likes"].median())}
    print(nm, out[f"launch_lift_{nm}_accounts"])
json.dump(out, open("data/launch/second_account.json", "w"), indent=1)
for r in aff.sort("likes", descending=True).head(12).iter_rows(named=True):
    print(f"[{r['author_handle']} -> {r['orig_handle']} | q {r['likes']} vs o {r['orig_likes']}]", r["text"][:120].replace("\n", " / "))
