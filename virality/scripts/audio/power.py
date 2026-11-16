#!/usr/bin/env python3
"""What the corpus could show about sound if every video's audio were in hand: sample sizes and the smallest effect
that would be detectable, per set of posts. No audio is used: traits are dealt at random to the real video posts.

    python scripts/audio/power.py      # reads data/tracks/launch/launch_posts.parquet, data/media/videos/index.parquet; writes data/audio/power.json

For a trait carried by a share p of the videos in a subset that is a share s of all videos (e.g. s = 0.5 "videos with a
voiceover", p = 0.5 "speech in the first two seconds"), 40 random deals give the median number of same-account pairs and
accounts and the median half-width of the 95% interval of the win rate and of the likes ratio. An effect smaller than the
half-width cannot be told from none; one about 1.4 times the half-width would be found four times in five.
Also a check of the estimators: a planted 1.5x effect must be recovered.
"""
import json, sys, numpy as np, polars as pl
sys.path.insert(0, "scripts/audio")
import effects as E

r = np.random.default_rng(1)
T = pl.read_parquet("data/tracks/launch/launch_posts.parquet").filter(pl.col("has_video") & (pl.col("source") == "community_archive") & pl.col("lift_likes").is_not_null())
T = T.with_columns(pl.col("text").fill_null("").str.replace_all(r"https?://\S+", "").str.slice(0, 300).str.contains(E.STRICT).alias("strict"))
gif = set(pl.read_parquet("data/media/videos/index.parquet").filter(pl.col("mtype") == "gif")["post_id"].to_list())
T = T.with_columns(pl.col("post_id").is_in(list(gif)).alias("is_gif"))
out = {"note": __doc__.split("\n\n")[2].strip(), "scopes": {}}
CASES = [(1.0, 0.5), (1.0, 0.25), (0.5, 0.5), (0.5, 0.25), (0.25, 0.5)]
for sk, (desc, sf) in E.SCOPES.items():
    S0 = T.filter(sf); S = S0.filter(~pl.col("is_gif"))
    per = S.group_by("author_id").len()
    res = {"description": desc, "video_and_gif_posts": S0.height, "gif_posts": S0.height - S.height, "video_posts": S.height, "accounts": per.height, "accounts_with_2plus_videos": int((per["len"] >= 2).sum()),
           "video_posts_in_accounts_with_2plus": int(per.filter(pl.col("len") >= 2)["len"].sum()), "cases": []}
    for s, p in CASES:
        pr, ac, hw, hr = [], [], [], []
        for _ in range(40):
            q = S.filter(pl.Series(r.random(S.height) < s))
            q = q.with_columns(pl.Series("a", r.random(q.height) < p))
            a, b = E.pairs_est(q, B=300), E.ratio_est(q, B=300)
            if a and a["win_rate"] is not None and np.isfinite(a["lo"]): pr.append(a["pairs"]); ac.append(a["accounts"]); hw.append((a["hi"] - a["lo"]) / 2)
            if b: hr.append(np.sqrt(b["ratio_hi"] / b["ratio_lo"]))
        md = lambda v: None if len(v) < 20 else float(np.median(v))
        res["cases"].append({"subset_share": s, "trait_share": p, "median_pairs": md(pr), "median_accounts": md(ac), "win_rate_half_width": md(hw) and round(md(hw), 3), "likes_ratio_half_width_factor": md(hr) and round(md(hr), 3),
                             "estimable_deals_of_40": len(pr)})
    out["scopes"][sk] = res
    print(f"\n{sk}: {res['video_posts']} videos (+{res['gif_posts']} GIFs), {res['accounts']} accounts, {res['accounts_with_2plus_videos']} with 2+ ({res['video_posts_in_accounts_with_2plus']} posts)")
    for k in res["cases"]: print("   subset {subset_share:<5} trait {trait_share:<5} pairs {median_pairs}  accounts {median_accounts}  win rate +/- {win_rate_half_width}  ratio x/÷ {likes_ratio_half_width_factor}  ({estimable_deals_of_40}/40)".format(**k))
# estimator check: plant a 1.5x effect on a random half of launch-type videos of every account
S = T.filter(E.SCOPES["announce_all"][1] & ~pl.col("is_gif"))
a = r.random(S.height) < 0.5
q = S.with_columns(pl.Series("a", a), (pl.col("lift_likes") + pl.Series(np.where(a, np.log(1.5), 0.0))).alias("lift_likes"))
out["planted_1_5x_check"] = {"pairs": E.pairs_est(q), "within_account": E.ratio_est(q), "adjusted": E.adjusted_one(q, B=200)}
print("\nplanted 1.5x:", json.dumps(out["planted_1_5x_check"]))
json.dump(out, open("data/audio/power.json", "w"), indent=1)
