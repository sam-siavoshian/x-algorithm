#!/usr/bin/env python3
"""Measured effect of each kind of change, with posting order balanced: the table the advice layer reads.

    python scripts/evidence/build_edit_evidence.py <dir>      # needs <dir>/evidence/tac_pairs.parquet (tac_pairs.py)

For every attribute, among same-account, same-topic pairs where exactly one version has it, the win
rate of the version with it is computed separately for pairs where that version was posted first and
where it was posted second, and the two are averaged. Of two posts on one topic the later one wins
more often, so an unbalanced rate would credit an attribute with the effect of posting second.
Intervals come from resampling accounts. Writes xviral/data/edit_evidence.json.
"""
import json
import sys

import numpy as np
import polars as pl

from xviral.features import text_structure as ts

S = sys.argv[1]
p = pl.read_parquet(f"{S}/evidence/tac_pairs.parquet")
cnt = pl.concat([p.select("author_id", "key", "post_id"), p.select("author_id", "key", pl.col("post_id_b").alias("post_id"))]).unique().group_by("author_id", "key").len()
p = p.join(cnt.filter(pl.col("len") <= 4).select("author_id", "key"), on=["author_id", "key"], how="semi").filter(~pl.col("key").str.contains("youtube.com/watch$"))
posts = pl.concat([p.select("post_id", "author_id", "created_at_ms", "text", "has_media"),
                   p.select(pl.col("post_id_b").alias("post_id"), "author_id", pl.col("created_at_ms_b").alias("created_at_ms"), pl.col("text_b").alias("text"),
                            pl.col("has_media_b").alias("has_media"))]).unique("post_id")
f = ts.text_structure(posts).join(posts.select("post_id", pl.col("text").fill_null("").str.replace_all(r"https?://\S+", "").str.len_chars().alias("chars")), on="post_id")
a = p.join(f, on="post_id").join(f.rename({c: c + "_b" for c in f.columns}), on="post_id_b")
d = ((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log())
a = a.with_columns(d.alias("d"), (pl.col("created_at_ms") < pl.col("created_at_ms_b")).alias("a_first")).filter(pl.col("d").abs() >= np.log(1.25))
E = {
    "media": ("an image or video is attached", pl.col("has_media"), pl.col("has_media_b"), None),
    "lines": ("the text is laid out on three or more lines", pl.col("ts_lines") >= 3, pl.col("ts_lines_b") >= 3, None),
    "longer": ("the text is at least 1.5 times longer", pl.col("chars") >= 1.5 * pl.col("chars_b").clip(lower_bound=20), pl.col("chars_b") >= 1.5 * pl.col("chars").clip(lower_bound=20), None),
    "link_not_first": ("the link is not in the first line", pl.col("ts_url_first_line") == 0, pl.col("ts_url_first_line_b") == 0, "url:"),
    "link_at_end": ("the link is at the very end", pl.col("ts_url_at_end") > 0, pl.col("ts_url_at_end_b") > 0, "url:"),
    "question_open": ("the post opens with a question", pl.col("ts_open_question") > 0, pl.col("ts_open_question_b") > 0, None),
    "first_person_open": ("the post opens in the first person", pl.col("ts_open_first_person") > 0, pl.col("ts_open_first_person_b") > 0, None),
    "hashtag": ("the post has a hashtag", pl.col("text").str.contains(r"#\w"), pl.col("text_b").str.contains(r"#\w"), None),
    "no_hashtag": ("the post has no hashtag", ~pl.col("text").str.contains(r"#\w"), ~pl.col("text_b").str.contains(r"#\w"), None),
    "mention": ("the post mentions another account", pl.col("text").str.contains(r"@\w"), pl.col("text_b").str.contains(r"@\w"), None),
}
rng = np.random.default_rng(0)
out = {}
for key, (desc, fa, fb, only) in E.items():
    q = a.with_columns(fa.fill_null(False).alias("fa"), fb.fill_null(False).alias("fb")).filter(pl.col("fa") != pl.col("fb"))
    if only:
        q = q.filter(pl.col("key").str.starts_with(only))
    q = q.with_columns((pl.when(pl.col("fa")).then(pl.col("d")).otherwise(-pl.col("d")) > 0).alias("win"), (pl.col("fa") == pl.col("a_first")).alias("with_first"))
    if q.height < 100 or q.filter(pl.col("with_first")).height < 30 or q.filter(~pl.col("with_first")).height < 30:
        continue
    per = q.group_by("author_id").agg(
        (pl.col("win") & pl.col("with_first")).sum().alias("w1"), pl.col("with_first").sum().alias("n1"),
        (pl.col("win") & ~pl.col("with_first")).sum().alias("w2"), (~pl.col("with_first")).sum().alias("n2"))
    w1, n1, w2, n2 = (per[c].to_numpy().astype(float) for c in ("w1", "n1", "w2", "n2"))
    bal = lambda i: 0.5 * (w1[i].sum() / max(n1[i].sum(), 1) + w2[i].sum() / max(n2[i].sum(), 1))   # noqa: E731
    idx = rng.integers(0, len(w1), (2000, len(w1)))
    boots = np.array([bal(i) for i in idx])
    rate = bal(np.arange(len(w1)))
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    out[key] = {"description": desc, "win_rate": round(float(rate), 4), "lo": round(lo, 4), "hi": round(hi, 4), "pairs": int(q.height), "accounts": int(per.height),
                "supported": bool(lo > 0.5), "outcome": "likes", "design": "same account, same topic, posting order balanced"}
    print(f"{key:<18} {100*rate:5.1f}%  {100*lo:5.1f} to {100*hi:5.1f}  pairs {q.height:>5}  accounts {per.height:>4}  {'SUPPORTED' if lo > 0.5 else ''}")
out["_meta"] = {"built": "2026-11-11", "corpus": "4.48M filtered posts (Community Archive + 2026 view corpus)", "min_like_ratio": 1.25, "max_gap_days": 30}
json.dump(out, open("xviral/data/edit_evidence.json", "w"), indent=1)
