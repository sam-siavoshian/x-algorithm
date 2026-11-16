#!/usr/bin/env python3
"""Same-account, same-topic pairs (same link or same quoted post, different wording, within 30 days, likes differ by
1.25x or more; scripts/evidence/tac_pairs.py) restricted to pairs in which at least one side is an own announcement.
Win rate of the version with each attribute, posting order balanced, intervals from resampling accounts.
Writes data/launch/same_topic.json."""
import json, sys, numpy as np, polars as pl
rng = np.random.default_rng(0)
T = pl.read_parquet("data/launch/table.parquet")
T = T.with_columns((pl.col("chars") >= 280).alias("long280"), (pl.col("lines") >= 3).alias("lines3"), (pl.col("media_kind") == "video").alias("video"), (pl.col("media_kind") == "image").alias("image"),
                   (pl.col("media_kind") != "none").alias("media"), (pl.col("ext_link") & ~pl.col("points_to_reply")).alias("link_in_post"))
d = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id", "author_id", "urls", "quoted_post_id", "text").collect().join(T.select("post_id"), on="post_id", how="semi")
u = d.select("post_id", pl.col("urls").list.eval(pl.element().struct.field("expanded")).alias("u")).explode("u").filter(pl.col("u").is_not_null())
u = u.with_columns(pl.col("u").str.to_lowercase().str.replace(r"^https?://(www\.)?", "").str.replace(r"[?#].*$", "").str.strip_chars_end("/").alias("key"))
u = u.filter(~pl.col("key").str.contains(r"^(x\.com|twitter\.com|t\.co)") & (pl.col("key").str.len_chars() > 12) & pl.col("key").str.contains("/") & ~pl.col("key").str.contains("youtube.com/watch$"))
u = u.unique(["post_id", "key"]).select("post_id", pl.concat_str([pl.lit("url:"), pl.col("key")]).alias("key"))
q = d.filter(pl.col("quoted_post_id").is_not_null()).select("post_id", pl.concat_str([pl.lit("quote:"), pl.col("quoted_post_id").cast(pl.String)]).alias("key"))
k = pl.concat([u, q]).join(T, on="post_id").with_columns(pl.col("text").fill_null("").str.to_lowercase().str.replace_all(r"https?://\S+", "").str.replace_all(r"\s+", " ").str.strip_chars().alias("nt"))
g = k.group_by("author_id", "key").agg(pl.len().alias("n"), pl.col("nt").n_unique().alias("w")).filter((pl.col("n") >= 2) & (pl.col("n") <= 4) & (pl.col("w") >= 2))
k = k.join(g.select("author_id", "key"), on=["author_id", "key"], how="semi")
p = k.join(k, on=["author_id", "key"], suffix="_b").filter((pl.col("post_id") < pl.col("post_id_b")) & (pl.col("nt") != pl.col("nt_b")) & ((pl.col("created_at_ms") - pl.col("created_at_ms_b")).abs() <= 30 * 86_400_000))
p = p.with_columns(((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log()).alias("d"), (pl.col("created_at_ms") < pl.col("created_at_ms_b")).alias("a_first")).filter(pl.col("d").abs() >= np.log(1.25))
ANN = ["launch", "open_source", "feature", "funding", "milestone", "hiring"]
SC = {"one_side_launch": pl.col("ptype").is_in(["launch", "open_source"]) | pl.col("ptype_b").is_in(["launch", "open_source"]),
      "one_side_announcement": pl.col("ptype").is_in(ANN) | pl.col("ptype_b").is_in(ANN),
      "one_side_announcement_company_founder": (pl.col("ptype").is_in(ANN) | pl.col("ptype_b").is_in(ANN)) & (pl.col("akind") != "other")}
E = ["video", "image", "media", "link_in_post", "thread_marker", "open_announce", "open_problem", "open_number", "names_product_first_line", "long280", "lines3", "tags", "hashtag", "is_quote"]
out = {}
for sk, sf in SC.items():
    a = p.filter(sf)
    out[sk] = {"pairs": a.height, "accounts": a["author_id"].n_unique(), "effects": {}}
    print(f"\n== {sk}: {a.height} pairs, {a['author_id'].n_unique()} accounts; quote-topic {a.filter(pl.col('key').str.starts_with('quote:')).height}, link-topic {a.filter(pl.col('key').str.starts_with('url:')).height}")
    later = a.with_columns(((pl.col("d") > 0) != pl.col("a_first")).alias("later_wins"))
    print("  later post wins:", round(float(later["later_wins"].mean()), 3))
    out[sk]["later_post_wins"] = round(float(later["later_wins"].mean()), 4)
    for e in E:
        z = a.filter(pl.col(e) != pl.col(e + "_b")).with_columns((pl.when(pl.col(e)).then(pl.col("d")).otherwise(-pl.col("d")) > 0).alias("win"), (pl.col(e) == pl.col("a_first")).alias("wf"))
        n1, n2 = z.filter(pl.col("wf")).height, z.filter(~pl.col("wf")).height
        if z.height < 30 or n1 < 10 or n2 < 10:
            out[sk]["effects"][e] = {"pairs": z.height, "too_thin": True}; print(f"  {e:<28} too thin ({z.height} pairs)"); continue
        per = z.group_by("author_id").agg((pl.col("win") & pl.col("wf")).sum().alias("w1"), pl.col("wf").sum().alias("n1"), (pl.col("win") & ~pl.col("wf")).sum().alias("w2"), (~pl.col("wf")).sum().alias("n2")).sort("author_id")
        w1, m1, w2, m2 = (per[c].to_numpy().astype(float) for c in ("w1", "n1", "w2", "n2"))
        bal = lambda i: 0.5 * (w1[i].sum() / max(m1[i].sum(), 1) + w2[i].sum() / max(m2[i].sum(), 1))
        idx = rng.integers(0, len(w1), (2000, len(w1))); b = np.array([bal(i) for i in idx]); r = bal(np.arange(len(w1)))
        lo, hi = float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
        out[sk]["effects"][e] = {"win_rate": round(float(r), 4), "lo": round(lo, 4), "hi": round(hi, 4), "pairs": z.height, "accounts": per.height}
        print(f"  {e:<28} {100*r:5.1f}%  {100*lo:5.1f} to {100*hi:5.1f}  pairs {z.height:>4}  accounts {per.height:>4} {'*' if lo > .5 or hi < .5 else ''}")
json.dump(out, open("data/launch/same_topic.json", "w"), indent=1)
