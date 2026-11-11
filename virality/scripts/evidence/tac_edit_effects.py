#!/usr/bin/env python3
# Usage: python scripts/evidence/tac_edit_effects.py <dir>   where <dir> holds work/ (pipeline output); results go to <dir>/evidence/
"""On same-account, same-topic pairs: when exactly one version has an attribute, how often does it win?"""
import sys, json
import numpy as np, polars as pl
from xviral.features import text_structure as ts
S = sys.argv[1]
import os
os.makedirs(f"{S}/evidence", exist_ok=True)
p = pl.read_parquet(f"{S}/evidence/tac_pairs.parquet")
# Evergreen links (a storefront, a profile) are not one topic: keep link groups an account used at most 4 times; keep quote groups.
cnt = pl.concat([p.select("author_id", "key", "post_id"), p.select("author_id", "key", pl.col("post_id_b").alias("post_id"))]).unique().group_by("author_id", "key").len()
p = p.join(cnt.filter(pl.col("len") <= 4).select("author_id", "key"), on=["author_id", "key"], how="semi").filter(~pl.col("key").str.contains("youtube.com/watch$"))
posts = pl.concat([p.select("post_id", "author_id", "created_at_ms", "text", "has_media"),
                   p.select(pl.col("post_id_b").alias("post_id"), "author_id", pl.col("created_at_ms_b").alias("created_at_ms"), pl.col("text_b").alias("text"),
                            pl.col("has_media_b").alias("has_media"))]).unique("post_id")
f = ts.text_structure(posts).join(posts.select("post_id", "has_media", pl.col("text").fill_null("").str.replace_all(r"https?://\S+", "").str.len_chars().alias("chars")), on="post_id")
a = p.join(f, on="post_id").join(f.rename({c: c + "_b" for c in f.columns}), on="post_id_b")
print("pairs:", a.height, "accounts:", a["author_id"].n_unique(), " quote-topic:", a.filter(pl.col("key").str.starts_with("quote:")).height, " link-topic:", a.filter(pl.col("key").str.starts_with("url:")).height)
la, lb = (pl.col("likes") + 1).log(), (pl.col("likes_b") + 1).log()
a = a.with_columns((la - lb).alias("d")).filter(pl.col("d").abs() >= np.log(1.25))     # drop near-ties
E = {
    "has an image or video": (pl.col("has_media"), pl.col("has_media_b")),
    "text at least 1.5x longer": (pl.col("chars") >= 1.5 * pl.col("chars_b").clip(lower_bound=20), pl.col("chars_b") >= 1.5 * pl.col("chars").clip(lower_bound=20)),
    "three or more lines": (pl.col("ts_lines") >= 3, pl.col("ts_lines_b") >= 3),
    "opens with a question": (pl.col("ts_open_question") > 0, pl.col("ts_open_question_b") > 0),
    "opens in first person": (pl.col("ts_open_first_person") > 0, pl.col("ts_open_first_person_b") > 0),
    "has a hashtag": (pl.col("text").str.contains(r"#\w"), pl.col("text_b").str.contains(r"#\w")),
    "mentions another account": (pl.col("text").str.contains(r"@\w"), pl.col("text_b").str.contains(r"@\w")),
    "link at the very end (link topics)": (pl.col("ts_url_at_end") > 0, pl.col("ts_url_at_end_b") > 0),
    "link in the first line (link topics)": (pl.col("ts_url_first_line") > 0, pl.col("ts_url_first_line_b") > 0),
    "posted first of the two": (pl.col("created_at_ms") < pl.col("created_at_ms_b"), pl.col("created_at_ms_b") < pl.col("created_at_ms")),
}
rng = np.random.default_rng(0)
out = {}
print(f"{'version with ...':<40} {'wins':>6} {'95% interval':>14} {'pairs':>7} {'accounts':>9} {'median lift':>12}")
for name, (fa, fb) in E.items():
    q = a.with_columns(fa.fill_null(False).alias("fa"), fb.fill_null(False).alias("fb")).filter(pl.col("fa") != pl.col("fb"))
    if "link" in name:
        q = q.filter(pl.col("key").str.starts_with("url:"))
    if q.height < 100:
        continue
    q = q.with_columns(pl.when(pl.col("fa")).then(pl.col("d")).otherwise(-pl.col("d")).alias("gain"))
    # account-level bootstrap: an account with many pairs counts once per resample draw
    per = q.group_by("author_id").agg((pl.col("gain") > 0).sum().alias("w"), pl.len().alias("n"))
    w, n = per["w"].to_numpy(), per["n"].to_numpy()
    idx = rng.integers(0, len(w), (1000, len(w)))
    boots = w[idx].sum(1) / n[idx].sum(1)
    out[name] = {"win_rate": float(w.sum() / n.sum()), "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
                 "pairs": int(n.sum()), "accounts": int(len(w)), "median_lift": float(np.exp(q["gain"].median()))}
    o = out[name]
    print(f"{name:<40} {100*o['win_rate']:>5.1f}% {100*o['lo']:>5.1f} to {100*o['hi']:<5.1f} {o['pairs']:>7} {o['accounts']:>9} {o['median_lift']:>11.2f}x")
json.dump(out, open(f"{S}/evidence/tac_edits.json", "w"), indent=1)
