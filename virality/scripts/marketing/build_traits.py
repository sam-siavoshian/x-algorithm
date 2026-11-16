#!/usr/bin/env python3
"""Marketing-claim traits for every post of the launch table (scripts/launch/build_table.py must have run).

    python scripts/marketing/build_traits.py     # reads data/launch/table.parquet + data/work/filtered/posts.parquet
                                                 # writes data/marketing/table.parquet

Each trait is the measurable form of one outside claim (research/tracks/marketing/MARKETING_RESEARCH.md). Text traits are
regular expressions on the post text with links removed; context traits use the account's other posts and quote posts by
other corpus accounts. Nothing here looks at the outcome.
"""
import os, sys, polars as pl
sys.path.insert(0, "scripts/marketing")
from traits import TEXT_TRAITS, TEASER, add_text_traits

os.makedirs("data/marketing", exist_ok=True)
T = pl.read_parquet("data/launch/table.parquet")
P = pl.scan_parquet("data/work/filtered/posts.parquet")
LAUNCH = ["launch", "open_source"]

# ---- media detail (count of images, video duration) and repost / reply counts for secondary outcomes
m = P.select("post_id", "author_id", "media", "quotes", "bookmarks").collect().join(T.select("author_id").unique(), on="author_id", how="semi")
vid = pl.element().struct.field("type").is_in(["video", "gif", "animated_gif"])
m = m.select("post_id", "quotes", "bookmarks",
             pl.col("media").list.len().fill_null(0).alias("media_n"),
             pl.col("media").list.eval(pl.element().struct.field("duration_ms").filter(vid)).list.max().alias("video_ms"),
             pl.col("media").list.eval(pl.element().struct.field("type") == "photo").list.sum().fill_null(0).alias("photo_n"),
             pl.col("media").list.eval(pl.element().struct.field("alt").is_not_null() & (pl.element().struct.field("alt").str.len_chars() > 0)).list.any().fill_null(False).alias("has_alt"),
             pl.col("media").list.eval(pl.element().struct.field("alt")).list.join(" ").fill_null("").str.to_lowercase().alias("alt_text"))
T = T.join(m, on="post_id", how="left")

# ---- secondary outcomes: reposts and replies relative to the account's trailing median (same definition as lift_likes)
def trailing(df, col, out):
    x = df.filter(pl.col(col).is_not_null()).select("post_id", "author_id", "created_at_ms", pl.col(col).cast(pl.Float64).alias("v")).sort("author_id", "created_at_ms")
    x = x.with_columns(pl.col("v").shift(1).rolling_median(window_size=30, min_samples=8).over("author_id").alias("_m"))
    return df.join(x.select("post_id", ((pl.col("v") + 1) / (pl.col("_m") + 1)).log().alias(out)), on="post_id", how="left")
T = trailing(T, "reposts", "lift_reposts"); T = trailing(T, "replies", "lift_replies")

# ---- text traits
T = add_text_traits(T)

# ---- Pacific time (America/Los_Angeles, daylight saving handled)
pt = pl.from_epoch("created_at_ms", "ms").dt.replace_time_zone("UTC").dt.convert_time_zone("America/Los_Angeles")
T = T.with_columns(pt.dt.weekday().alias("dow_pt"), pt.dt.hour().alias("hour_pt"))
wd = pl.col("dow_pt") <= 5
T = T.with_columns((wd & pl.col("hour_pt").is_between(6, 11)).alias("pt_weekday_morning"), (wd & pl.col("hour_pt").is_between(8, 10)).alias("pt_weekday_8_10"),
                   (pl.col("dow_pt").is_in([2, 3, 4]) & pl.col("hour_pt").is_between(6, 11)).alias("pt_tue_thu_morning"), (pl.col("dow_pt") == 2).alias("pt_tuesday"),
                   (wd & pl.col("hour_pt").is_between(12, 17)).alias("pt_weekday_afternoon"), (pl.col("hour_pt").is_between(18, 23) | pl.col("hour_pt").is_between(0, 5)).alias("pt_evening_night"),
                   (~wd).alias("pt_weekend"))

# ---- context: the account's own posts before and after each launch-type post
T = T.sort("author_id", "created_at_ms")
own = T.select("author_id", "post_id", "created_at_ms", "text", "kind", "quoted_post_id", "ptype", "lift_likes")
L = T.filter(pl.col("ptype").is_in(LAUNCH)).select("author_id", pl.col("post_id").alias("lid"), pl.col("created_at_ms").alias("lms"))
j = L.join(own, on="author_id").with_columns(((pl.col("created_at_ms") - pl.col("lms")) / 3_600_000).alias("dh")).filter((pl.col("dh") != 0) & pl.col("dh").is_between(-168, 168))
lc = pl.col("text").fill_null("").str.replace_all(r"https?://\S+", " ").str.to_lowercase().str.replace_all("’", "'")
j = j.with_columns(lc.str.contains(TEASER).alias("teaser_txt"), (pl.col("quoted_post_id") == pl.col("lid")).fill_null(False).alias("self_quote"))
ctx = j.group_by("lid").agg(
    (pl.col("teaser_txt") & pl.col("dh").is_between(-72, -1)).any().alias("teaser_72h"),                      # a teaser 1 to 72 hours before
    (pl.col("teaser_txt") & pl.col("dh").is_between(-36, -1)).any().alias("teaser_36h"),
    pl.col("dh").is_between(-24, -1e-9).sum().alias("own_posts_prev24h"),
    pl.col("dh").is_between(-72, -1e-9).sum().alias("own_posts_prev72h"),
    (pl.col("self_quote") & pl.col("dh").is_between(1e-9, 168)).any().alias("self_quote_7d"),                 # the account quote-posts its own launch within 7 days
    (pl.col("self_quote") & pl.col("dh").is_between(24, 168)).any().alias("self_quote_day2_7"),               # ... from the second day on
    pl.col("dh").is_between(24, 96).sum().alias("own_posts_day2_4"),
    (pl.col("dh").is_between(24, 96) & pl.col("ptype").is_in(LAUNCH + ["feature", "milestone", "demo"])).any().alias("followup_announce_day2_4"),
    (pl.col("dh").is_between(-168, -24) & pl.col("ptype").is_in(LAUNCH)).any().alias("launch_in_prev_week"),
).rename({"lid": "post_id"})
T = T.join(ctx, on="post_id", how="left").with_columns([pl.col(c).fill_null(False) for c in ("teaser_72h", "teaser_36h", "self_quote_7d", "self_quote_day2_7", "followup_announce_day2_4", "launch_in_prev_week")] +
                                                         [pl.col(c).fill_null(0) for c in ("own_posts_prev24h", "own_posts_prev72h", "own_posts_day2_4")])
T = T.with_columns((pl.col("own_posts_prev24h") == 0).alias("quiet_prev24h"), (pl.col("own_posts_prev72h") >= 6).alias("busy_prev72h"), (pl.col("own_posts_day2_4") >= 3).alias("active_day2_4"))

# ---- context: quote posts of the launch by OTHER corpus accounts, by delay; first-person ("we", "our") = the same team
q = P.select("post_id", "author_id", "created_at_ms", "quoted_post_id", "text", "author_followers").filter(pl.col("quoted_post_id").is_not_null()).collect()
q = q.join(T.filter(pl.col("ptype").is_in(LAUNCH)).select(pl.col("post_id").alias("quoted_post_id"), pl.col("author_id").alias("oa"), pl.col("created_at_ms").alias("oms")), on="quoted_post_id")
q = q.filter(pl.col("author_id") != pl.col("oa")).with_columns(((pl.col("created_at_ms") - pl.col("oms")) / 3_600_000).alias("lag_h"),
        pl.col("text").fill_null("").str.to_lowercase().str.replace_all("’", "'").str.contains(r"\b(we|we're|we've|our|my team|i'm (so )?(excited|proud|thrilled)|i've been (working|building)|proud of (the|our) team|been working on this|what we('ve| have) been (building|working))\b").alias("fp"))
q = q.filter(pl.col("lag_h") >= 0)
q.write_parquet("data/marketing/quotes_of_launches.parquet")
qa = q.group_by("quoted_post_id").agg(
    (pl.col("lag_h") <= 1).sum().alias("quotes_1h"), (pl.col("lag_h") <= 6).sum().alias("quotes_6h"), (pl.col("lag_h") <= 48).sum().alias("quotes_48h"),
    ((pl.col("lag_h") <= 1) & pl.col("fp")).sum().alias("team_quotes_1h"), ((pl.col("lag_h") <= 24) & pl.col("fp")).sum().alias("team_quotes_24h"),
    ((pl.col("lag_h") <= 1) & (pl.col("author_followers") >= 50_000)).sum().alias("big_quotes_1h"),
    pl.col("lag_h").min().alias("first_quote_lag_h")).rename({"quoted_post_id": "post_id"})
T = T.join(qa, on="post_id", how="left").with_columns([pl.col(c).fill_null(0) for c in ("quotes_1h", "quotes_6h", "quotes_48h", "team_quotes_1h", "team_quotes_24h", "big_quotes_1h")])
T = T.with_columns((pl.col("quotes_1h") > 0).alias("quoted_1h"), (pl.col("team_quotes_1h") > 0).alias("team_quoted_1h"), (pl.col("team_quotes_24h") > 0).alias("team_quoted_24h"), (pl.col("quotes_48h") > 0).alias("quoted_48h"),
                   ((pl.col("quotes_48h") > 0) & (pl.col("quotes_1h") == 0)).alias("quoted_late_only"))
T.write_parquet("data/marketing/table.parquet")
S = T.filter(pl.col("ptype").is_in(LAUNCH) & pl.col("lift_likes").is_not_null() & (pl.col("source") == "community_archive"))
print("launch-type posts with a baseline (Community Archive):", S.height, "of company/founder accounts:", S.filter(pl.col("akind") != "other").height)
names = list(TEXT_TRAITS) + ["pt_weekday_morning", "pt_weekday_8_10", "pt_tue_thu_morning", "pt_tuesday", "pt_weekend", "teaser_72h", "teaser_36h", "quiet_prev24h", "busy_prev72h", "self_quote_7d", "self_quote_day2_7", "active_day2_4",
                             "followup_announce_day2_4", "launch_in_prev_week", "quoted_1h", "team_quoted_1h", "team_quoted_24h", "quoted_48h"]
for n in names:
    a, c = S.filter(pl.col(n)), S.filter(pl.col(n) & (pl.col("akind") != "other"))
    print(f"{n:<28} all {a.height:>5} posts {a['author_id'].n_unique():>4} acc | company+founder {c.height:>5} posts {c['author_id'].n_unique():>4} acc")
print("video launch posts with a duration:", S.filter((pl.col("media_kind") == "video") & pl.col("video_ms").is_not_null()).height, "of", S.filter(pl.col("media_kind") == "video").height)
