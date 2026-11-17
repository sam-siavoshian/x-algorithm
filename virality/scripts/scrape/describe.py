#!/usr/bin/env python3
"""Validate and describe the home-timeline scrape: ids, duplicates, overlap with the corpus, snapshot delay and what it
does to the counts, views against followers, topics, post types, language.

    python scripts/scrape/describe.py            # data/scrape_track/describe.json, lang.parquet, post_meta.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from xviral import filters
from xviral.config import Config
from xviral.sources import home_scrape
from xviral.text import launch_labels as ll
from xviral.text.tech import is_off_audience_post, is_tech_post
from xviral.util.core import date_to_ms, to_int

D, OUT = Path("data"), Path("data/scrape_track")
LOW = pl.col("author_handle").str.to_lowercase().alias("handle")
R: dict = {}


def q(x, ps=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return {f"p{int(p * 100)}": float(np.quantile(x, p)) for p in ps} | {"n": int(len(x))}


raw = pl.read_parquet(D / "scrape/converted/posts.parquet")
posts, rep = home_scrape.read(D / "scrape/converted/posts.parquet")
posts = posts.with_columns(LOW)
R["reader"] = rep

# 1. ids
ids = raw["id_str"].to_list()
R["ids"] = {"dtype": str(raw["id_str"].dtype), "all_digit_strings": bool(raw["id_str"].str.contains(r"^\d+$").all()),
            "to_int_exact": int(sum(str(to_int(s)) == s for s in ids)), "through_float_exact": int(sum(str(int(float(s))) == s for s in ids)),
            "rows": len(ids), "tweet_id_column_all_nan": bool(raw["tweet_id"].is_nan().all()),
            "handles_case_sensitive": raw["screen_name"].n_unique(), "handles_lower": raw["screen_name"].str.to_lowercase().n_unique()}
dup = raw.filter(pl.col("id_str").is_duplicated()).sort("id_str", "saved_at").select("id_str", "saved_at", "likes", "impressions")
R["duplicates"] = {"rows_with_a_repeated_id": dup.height, "detail": dup.to_dicts()}
txt = posts.with_columns(pl.col("text").str.replace_all(r"https?://\S+", "").str.strip_chars().str.to_lowercase().alias("_t")).filter(pl.col("_t").str.len_chars() >= 40)
g = txt.group_by("handle", "_t").len().filter(pl.col("len") > 1)
R["duplicates"]["same_account_same_text_groups"] = g.height
R["duplicates"]["same_account_same_text_posts"] = int(g["len"].sum())
g2 = txt.group_by("_t").agg(pl.len(), pl.col("handle").n_unique().alias("a")).filter(pl.col("a") > 1)
R["duplicates"]["same_text_across_accounts_groups"] = g2.height
R["duplicates"]["same_text_across_accounts_posts"] = int(g2["len"].sum())

# 2. overlap with the corpus
filt = pl.scan_parquet(D / "work/filtered/posts.parquet").select("post_id", "author_id", LOW, "source", "likes", "views", "snapshot_at_ms", "created_at_ms", "author_followers").collect()
rows = pl.read_parquet(D / "work/dataset/rows.parquet", columns=["post_id", "author_id", "author_handle", "split", "in_audience", "views"]).with_columns(LOW)
S = pl.read_parquet(D / "tracks/data/startup_accounts.parquet")
bench = pl.read_parquet(D / "tracks/pairs/benchmark_pairs.parquet")
n_acc = posts.group_by("handle").agg(pl.len().alias("n"))
fh, rh = filt["handle"].unique(), rows["handle"].unique()
bench_handles = filt.filter(pl.col("author_id").is_in(bench["author_id"].unique().implode()))["handle"].unique()
ov = {"posts_also_in_filtered_corpus": posts.join(filt.select("post_id"), on="post_id", how="semi").height,
      "posts_also_training_rows": posts.join(rows.select("post_id"), on="post_id", how="semi").height,
      "handles_in_filtered_corpus": n_acc.filter(pl.col("handle").is_in(fh.implode())).height,
      "handles_in_training_rows": n_acc.filter(pl.col("handle").is_in(rh.implode())).height,
      "posts_of_handles_in_filtered_corpus": int(n_acc.filter(pl.col("handle").is_in(fh.implode()))["n"].sum()),
      "posts_of_handles_in_training_rows": int(n_acc.filter(pl.col("handle").is_in(rh.implode()))["n"].sum()),
      "handles_with_20plus_in_training_rows": n_acc.filter((pl.col("n") >= 20) & pl.col("handle").is_in(rh.implode())).height,
      "handles_of_benchmark_accounts": n_acc.filter(pl.col("handle").is_in(bench_handles.implode())).height,
      "posts_of_benchmark_accounts": int(n_acc.filter(pl.col("handle").is_in(bench_handles.implode()))["n"].sum()),
      "handles_in_startup_accounts": n_acc.filter(pl.col("handle").is_in(S["handle"].implode())).height,
      "posts_of_startup_accounts": int(n_acc.filter(pl.col("handle").is_in(S["handle"].implode()))["n"].sum()),
      "startup_accounts_with_20plus": n_acc.filter((pl.col("n") >= 20) & pl.col("handle").is_in(S["handle"].implode())).height,
      "startup_accounts_with_5plus": n_acc.filter((pl.col("n") >= 5) & pl.col("handle").is_in(S["handle"].implode())).height}
sk = n_acc.join(S.select("handle", "kind"), on="handle").group_by("kind").agg(pl.len().alias("accounts"), pl.col("n").sum().alias("posts"))
ov["startup_by_kind"] = sk.to_dicts()
R["overlap"] = ov
R["posts_per_account"] = {"median": float(n_acc["n"].median()), "mean": float(n_acc["n"].mean()), "accounts": n_acc.height,
                          **{f"accounts_{k}plus": n_acc.filter(pl.col("n") >= k).height for k in (2, 5, 10, 20, 50, 100)},
                          **{f"posts_of_accounts_{k}plus": int(n_acc.filter(pl.col("n") >= k)["n"].sum()) for k in (2, 5, 10, 20, 50, 100)}}

# The same post read twice: here and, later, by another source. The ratio of the two like counts is how much the count grew.
same = posts.select("post_id", "likes", "views", "snapshot_at_ms", "created_at_ms").join(
    filt.select("post_id", pl.col("likes").alias("likes_c"), pl.col("views").alias("views_c"), pl.col("snapshot_at_ms").alias("snap_c"), pl.col("source").alias("source_c")), on="post_id")
same = same.with_columns(((pl.col("snapshot_at_ms") - pl.col("created_at_ms")) / 3.6e6).alias("age_h"))
R["same_post_two_readings"] = {"posts": same.height, "by_source": same.group_by("source_c").len().to_dicts()}
if same.height:
    s2 = same.filter((pl.col("likes") >= 5) & (pl.col("likes_c") >= 5) & (pl.col("snap_c") > pl.col("snapshot_at_ms")))
    s2 = s2.with_columns((pl.col("likes_c") / pl.col("likes")).log10().alias("lr"))
    by = []
    for lo, hi in ((0, 3), (3, 6), (6, 12), (12, 24), (24, 72), (72, 1e9)):
        b = s2.filter((pl.col("age_h") >= lo) & (pl.col("age_h") < hi))
        if b.height >= 5:
            by.append({"age_h": f"{lo}-{hi}", "posts": b.height, "median_later_over_scrape_likes": float(10 ** b["lr"].median()),
                       "p25": float(10 ** b["lr"].quantile(0.25)), "p75": float(10 ** b["lr"].quantile(0.75))})
    R["same_post_two_readings"]["likes_growth_by_age_at_scrape"] = by
    same.write_parquet(OUT / "same_post_two_readings.parquet")

# 3. snapshot delay and maturity
p = posts.with_columns(((pl.col("snapshot_at_ms") - pl.col("created_at_ms")) / 3.6e6).alias("age_h"))
age = p["age_h"].to_numpy()
R["delay_hours"] = q(age) | {f"share_under_{h}h": float((age < h).mean()) for h in (1, 3, 6, 12, 24, 48, 72, 168)} | {"share_over_30d": float((age > 720).mean())}
R["created"] = {"before_2025_10": int((p["created_at_ms"] < date_to_ms("2025-10-01")).sum()), "before_2025": int((p["created_at_ms"] < date_to_ms("2025-01-01")).sum()),
                "min": str(pl.from_epoch(p["created_at_ms"], "ms").min()), "max": str(pl.from_epoch(p["created_at_ms"], "ms").max()),
                "on_or_after_cutoff": int((p["created_at_ms"] >= date_to_ms("2026-04-23")).sum()),
                "by_month": p.with_columns(pl.from_epoch("created_at_ms", "ms").dt.strftime("%Y-%m").alias("m")).group_by("m").len().sort("m").tail(6).to_dicts()}
R["saved_by_day"] = q(p.with_columns(pl.from_epoch("snapshot_at_ms", "ms").dt.date().alias("d")).group_by("d").len()["len"].to_numpy())
p = p.with_columns((pl.col("views") / pl.col("author_followers").clip(lower_bound=1)).log10().alias("lvf"), (pl.col("likes") / pl.col("views").clip(lower_bound=1)).alias("lpv"),
                   pl.col("author_followers").clip(lower_bound=1).log10().floor().clip(1, 6).cast(pl.Int32).alias("tier"))
# Counts against age at capture, inside follower tiers: a rising curve is the counts still growing (and selection by age).
cur = []
for lo, hi in ((0, 1), (1, 3), (3, 6), (6, 12), (12, 24), (24, 48), (48, 72), (72, 168), (168, 720), (720, 1e9)):
    b = p.filter((pl.col("age_h") >= lo) & (pl.col("age_h") < hi))
    cur.append({"age_h": f"{lo}-{hi}", "posts": b.height, "median_views": float(b["views"].median()), "median_likes": float(b["likes"].median()),
                "median_views_per_follower": float(10 ** b["lvf"].median()), "median_likes_per_view": float(b["lpv"].median()),
                "median_views_per_follower_10k_100k": float(10 ** b.filter(pl.col("tier") == 4)["lvf"].median()) if b.filter(pl.col("tier") == 4).height else None})
R["counts_by_age_at_capture"] = cur
# Inside accounts: the same account's posts captured younger and older (accounts with 10+ posts), so the account is held fixed.
big = p.filter(pl.len().over("handle") >= 10).with_columns((pl.col("views").log10() - pl.col("views").log10().median().over("handle")).alias("rel"))
wa = []
for lo, hi in ((0, 1), (1, 3), (3, 6), (6, 12), (12, 24), (24, 48), (48, 72), (72, 168), (168, 1e9)):
    b = big.filter((pl.col("age_h") >= lo) & (pl.col("age_h") < hi) & pl.col("rel").is_finite())
    wa.append({"age_h": f"{lo}-{hi}", "posts": b.height, "accounts": b["handle"].n_unique(), "views_over_account_median": float(10 ** b["rel"].median())})
R["views_by_age_within_account"] = wa

# 4. views against followers
R["views"] = q(p["views"].to_numpy()) | {"zero": int((p["views"] == 0).sum())}
R["likes"] = q(p["likes"].to_numpy())
R["followers"] = q(p.unique("handle")["author_followers"].to_numpy())
R["likes_gt_views"] = int((p["likes"] > p["views"]).sum())
tiers = []
names = {1: "under 100", 2: "100-1k", 3: "1k-10k", 4: "10k-100k", 5: "100k-1M", 6: "1M+"}
for t in range(1, 7):
    b = p.filter(pl.col("tier") == t)
    tiers.append({"tier": names[t], "posts": b.height, "accounts": b["handle"].n_unique(), "median_views": float(b["views"].median()), "median_likes": float(b["likes"].median()),
                  "views_per_follower_p10": float(10 ** b["lvf"].quantile(0.1)), "views_per_follower_median": float(10 ** b["lvf"].median()),
                  "views_per_follower_p90": float(10 ** b["lvf"].quantile(0.9)), "likes_per_view_median": float(b["lpv"].median())})
R["by_follower_tier"] = tiers
ok = p.filter((pl.col("views") > 0) & (pl.col("author_followers") > 0))
x, y = np.log10(ok["author_followers"].to_numpy()), np.log10(ok["views"].to_numpy())
sl, ic = np.polyfit(x, y, 1)
R["log_views_on_log_followers"] = {"slope": float(sl), "intercept": float(ic), "r": float(np.corrcoef(x, y)[0, 1]), "residual_sd_log10": float(np.std(y - sl * x - ic))}
# The corpus's own view-labelled posts at the same follower tier (2026 view corpus: a search sample).
cf = filt.filter(pl.col("views").is_not_null() & (pl.col("author_followers") > 0)).with_columns(
    (pl.col("views") / pl.col("author_followers")).log10().alias("lvf"), pl.col("author_followers").log10().floor().clip(1, 6).cast(pl.Int32).alias("tier"), (pl.col("likes") / pl.col("views").clip(lower_bound=1)).alias("lpv"))
R["corpus_views_by_tier"] = [{"tier": names[t], "posts": cf.filter(pl.col("tier") == t).height,
                              "views_per_follower_median": float(10 ** cf.filter(pl.col("tier") == t)["lvf"].median()) if cf.filter(pl.col("tier") == t).height else None,
                              "likes_per_view_median": float(cf.filter(pl.col("tier") == t)["lpv"].median()) if cf.filter(pl.col("tier") == t).height else None} for t in range(1, 7)]
ca = filt.filter((pl.col("source") == "community_archive") & (pl.col("author_followers") > 0) & (pl.col("likes") >= 0)).with_columns(
    ((pl.col("likes") + 1) / pl.col("author_followers")).log10().alias("llf"), pl.col("author_followers").log10().floor().clip(1, 6).cast(pl.Int32).alias("tier"))
pp = p.with_columns(((pl.col("likes") + 1) / pl.col("author_followers").clip(lower_bound=1)).log10().alias("llf"))
R["likes_per_follower_by_tier"] = [{"tier": names[t], "scrape": float(10 ** pp.filter(pl.col("tier") == t)["llf"].median()),
                                    "community_archive": float(10 ** ca.filter(pl.col("tier") == t)["llf"].median()) if ca.filter(pl.col("tier") == t).height else None,
                                    "nitter_2026": float(10 ** cf.with_columns(((pl.col("likes") + 1) / pl.col("author_followers")).log10().alias("llf")).filter(pl.col("tier") == t)["llf"].median()) if cf.filter(pl.col("tier") == t).height else None}
                                   for t in range(1, 7)]

# 5. language, topics, post types
cfg = Config.load(D / "config.yaml")
lc = OUT / "lang.parquet"
lg = filters.resolve_language(posts.select("post_id", "lang_src", "text"), cfg.filters, pl.read_parquet(lc) if lc.exists() else None).select("post_id", "lang")
lg.write_parquet(lc)
posts = posts.join(lg, on="post_id", how="left")
vc = posts["lang"].value_counts(sort=True)
R["language"] = {"top": vc.head(12).to_dicts(), "english_share": float((posts["lang"] == "en").mean())}
posts = posts.with_columns(is_tech_post().alias("tech"), is_tech_post(strong=True).alias("tech_strong"), is_off_audience_post().alias("off"),
                           pl.col("text").fill_null("").str.contains(r"(?i)(\$[A-Z]{2,6}\b|\b(airdrop|memecoin|altcoins?|presale|tokenomics|staking|defi|nft|web3|solana|ethereum|bitcoin|btc|eth|hodl|crypto|onchain|on-chain)\b)").alias("crypto"),
                           pl.col("text").fill_null("").str.contains(r"(?i)\b(trump|biden|maga|democrats?|republicans?|congress|senate|election|immigration|gaza|israel|ukraine|putin|nato|woke|liberals?|conservatives?)\b").alias("politics"))
en = posts.filter(pl.col("lang") == "en")
R["topics"] = {"posts_tech_pattern": float(posts["tech"].mean()), "posts_tech_strong": float(posts["tech_strong"].mean()), "posts_off_audience": float(posts["off"].mean()),
               "posts_crypto": float(posts["crypto"].mean()), "posts_politics": float(posts["politics"].mean()),
               "english_posts_tech_strong": float(en["tech_strong"].mean()), "english_posts_off_audience": float(en["off"].mean())}
acc = posts.group_by("handle").agg(pl.len().alias("n"), pl.col("tech_strong").mean().alias("ts"), pl.col("off").mean().alias("os"), (pl.col("lang") == "en").mean().alias("en"),
                                   pl.col("author_followers").max().alias("followers"))
for k in (1, 5, 10, 20):
    a = acc.filter(pl.col("n") >= k)
    aud = a.filter((pl.col("ts") >= 0.15) & (pl.col("os") <= pl.col("ts")))
    R["topics"][f"accounts_{k}plus"] = {"accounts": a.height, "posts": int(a["n"].sum()), "audience_rule_accounts": aud.height, "audience_rule_posts": int(aud["n"].sum()),
                                         "english_majority_accounts": a.filter(pl.col("en") >= 0.5).height, "audience_and_english_accounts": aud.filter(pl.col("en") >= 0.5).height,
                                         "audience_and_english_posts": int(aud.filter(pl.col("en") >= 0.5)["n"].sum())}
# Post types (launch track rules). The account kind comes from the startup list where the handle is on it.
t = ll.add_types(posts.join(S.select("handle", pl.col("kind").alias("akind")), on="handle", how="left").with_columns(
    pl.col("media").list.eval(pl.element().struct.field("type") == "video").list.any().fill_null(False).alias("has_video")))
t = t.with_columns(ll.own_expr().alias("own"))
R["post_types"] = {"raw": t["ptype_raw"].value_counts(sort=True).to_dicts(), "own": t.filter(pl.col("own"))["ptype_raw"].value_counts(sort=True).to_dicts(),
                   "own_total": int(t["own"].sum()), "own_accounts": t.filter(pl.col("own"))["handle"].n_unique(),
                   "own_launch_by_startup_accounts": t.filter(pl.col("own") & (pl.col("ptype_raw") == "launch") & pl.col("akind").is_not_null()).height,
                   "own_launch_accounts_with_2plus": t.filter(pl.col("own") & (pl.col("ptype_raw") == "launch")).group_by("handle").len().filter(pl.col("len") >= 2).height,
                   "own_launch_under_10k": t.filter(pl.col("own") & (pl.col("ptype_raw") == "launch") & (pl.col("author_followers") < 10_000)).height,
                   "own_launch_under_1k": t.filter(pl.col("own") & (pl.col("ptype_raw") == "launch") & (pl.col("author_followers") < 1_000)).height}
R["media"] = {"with_media": int((posts["media"].list.len() > 0).sum()),
              "first_media_type": posts.select(pl.col("media").list.first().struct.field("type").alias("t"))["t"].value_counts().to_dicts(),
              "with_description": rep["with_image_description"],
              "description_by_first_media_type": posts.select(pl.col("media").list.first().struct.field("type").alias("t"), pl.col("media").list.first().struct.field("alt").is_not_null().alias("d")).group_by("t").agg(pl.len(), pl.col("d").sum()).to_dicts(),
              "text_ends_in_link_and_long": int(posts.filter((pl.col("text").str.len_chars() >= 270) & pl.col("text").str.contains(r"https://t\.co/\w+$")).height),
              "text_len": q(posts["text"].str.len_chars().to_numpy())}
R["reposts"] = {"n": rep["reposts"], "median_views": float(posts.filter(pl.col("kind") == "repost")["views"].median()), "zero_views": int(posts.filter((pl.col("kind") == "repost") & (pl.col("views") == 0)).height)}
R["verified_share"] = float(posts["author_verified"].mean())
t.select("post_id", "handle", "lang", "tech", "tech_strong", "off", "crypto", "politics", "ptype_raw", "own", "akind", "has_video").write_parquet(OUT / "post_meta.parquet")
acc.write_parquet(OUT / "accounts.parquet")
(OUT / "describe.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps(R, indent=1, default=str))
