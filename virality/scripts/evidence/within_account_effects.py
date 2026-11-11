#!/usr/bin/env python3
# Usage: python scripts/evidence/within_account_effects.py <dir>   where <dir> holds work/ (pipeline output); results go to <dir>/evidence/
"""Within-account association of each editable attribute with outcomes (audience accounts)."""
import sys, json
import numpy as np, polars as pl
S = sys.argv[1]
import os
os.makedirs(f"{S}/evidence", exist_ok=True)
d = pl.read_parquet(f"{S}/work/dataset/rows.parquet").filter(pl.col("in_audience") & (pl.col("split") != "test"))
d = d.with_columns(
    (pl.col("views").cast(pl.Float64) + 1).log10().alias("lv"), (pl.col("likes").cast(pl.Float64) + 1).log10().alias("ll"),
    pl.from_epoch("created_at_ms", "ms").dt.weekday().is_in([6, 7]).alias("weekend"),
)
A = {
    "external link in the post": pl.col("has_url") | (pl.col("external_urls_text") > 0),
    "image attached": pl.col("has_photo"),
    "video attached": pl.col("has_video"),
    "quote post": pl.col("is_quote"),
    "opens with a question": pl.col("ts_open_question") > 0,
    "opens with announcement phrasing": pl.col("ts_open_announce") > 0,
    "opens with a number": pl.col("ts_open_number") > 0,
    "opens in first person": pl.col("ts_open_first_person") > 0,
    "opens with an @mention": pl.col("ts_open_mention") > 0,
    "has bullet lines": pl.col("ts_bullet_lines") > 0,
    "three or more lines": pl.col("ts_lines") >= 3,
    "long post (over 280)": pl.col("is_long_post"),
    "very short (under 60 chars)": pl.col("weighted_text_len") < 60,
    "has a hashtag": pl.col("hashtags_n") > 0,
    "mentions another account": pl.col("mentions_n") > 0,
    "posted within 2h of previous post": pl.col("hours_since_prev_original") < 2,
    "posted on a weekend": pl.col("weekend"),
}
SUB = {"link in first line (among link posts)": (pl.col("ts_url_first_line") > 0, pl.col("ts_chars_before_url_share").is_not_null()),
       "link at the very end (among link posts)": (pl.col("ts_url_at_end") > 0, pl.col("ts_chars_before_url_share").is_not_null())}
rng = np.random.default_rng(0)

def effect(frame, flag, y, min_each=5):
    f = frame.filter(pl.col(y).is_not_null() & flag.is_not_null()).with_columns(flag.alias("_f"))
    g = f.group_by("author_id").agg(pl.col(y).filter(pl.col("_f")).mean().alias("a"), pl.col(y).filter(~pl.col("_f")).mean().alias("b"),
                                    pl.col("_f").sum().alias("na"), (~pl.col("_f")).sum().alias("nb"))
    g = g.filter((pl.col("na") >= min_each) & (pl.col("nb") >= min_each))
    if g.height < 30:
        return None
    diff = (g["a"] - g["b"]).to_numpy()
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
    return {"accounts": g.height, "posts_with": int(g["na"].sum()), "ratio": float(10 ** diff.mean()),
            "lo": float(10 ** np.quantile(boots, 0.025)), "hi": float(10 ** np.quantile(boots, 0.975)), "share_accounts_positive": float((diff > 0).mean())}

out = {}
for name, flag in A.items():
    out[name] = {"views": effect(d, flag, "lv"), "likes": effect(d, flag, "ll")}
for name, (flag, cond) in SUB.items():
    out[name] = {"views": effect(d.filter(cond), flag, "lv"), "likes": effect(d.filter(cond), flag, "ll")}
json.dump(out, open(f"{S}/evidence/within.json", "w"), indent=1)
print(f"{'attribute':<42} {'views x':>8} {'95% interval':>15} {'accts':>6} {'%acct+':>7} | {'likes x':>8} {'95% interval':>15} {'accts':>6}")
for k, v in out.items():
    def fmt(e): return f"{e['ratio']:>8.2f} {e['lo']:>6.2f} to {e['hi']:<5.2f} {e['accounts']:>6}" if e else f"{'-':>8} {'':>15} {'':>6}"
    pv = f"{100*v['views']['share_accounts_positive']:>6.0f}%" if v["views"] else f"{'':>7}"
    print(f"{k:<42} {fmt(v['views'])} {pv} | {fmt(v['likes'])}")
