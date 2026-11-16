#!/usr/bin/env python3
"""One row per post of every account that has at least one own announcement, with the outcome relative to the account's
own trailing median and the attributes the launch analysis tests.

    python scripts/launch/build_table.py            # reads data/work/filtered/posts.parquet, writes data/launch/table.parquet

lift_likes = log((likes+1) / (median likes of the account's previous 30 posts + 1)), at least 8 previous posts;
lift_views likewise on posts with a view count. Posts captured less than 3 days after posting are dropped.
"""
import sys, numpy as np, polars as pl
sys.path.insert(0, "scripts/launch")
from labels import add_types, own_expr, ORDER, clean
K, MINH = 30, 8
cols = ["post_id", "author_id", "author_handle", "author_display_name", "author_followers", "created_at_ms", "snapshot_at_ms", "text", "kind", "quoted_post_id", "quoted_author_id",
        "media", "urls", "mentions_n", "hashtags_n", "views", "likes", "reposts", "replies", "source", "lang"]
d = pl.scan_parquet("data/work/filtered/posts.parquet").select(cols).collect()
acc = pl.read_parquet("data/launch/accounts.parquet").select("author_id", "akind", "person", pl.col("f").alias("followers_max"))
d = d.join(acc, on="author_id", how="left").with_columns(pl.col("akind").fill_null("other"))
mt = pl.col("media").list.eval(pl.element().struct.field("type"))
d = d.with_columns(mt.list.eval(pl.element().is_in(["video", "gif"])).list.any().fill_null(False).alias("has_video"), mt.list.eval(pl.element() == "photo").list.any().fill_null(False).alias("has_photo"),
                   (pl.col("media").list.len() > 0).alias("has_media"))
d = add_types(d).with_columns(own_expr().alias("own"))
d = d.with_columns(pl.when(pl.col("own")).then(pl.col("ptype_raw")).otherwise(pl.lit("other")).alias("ptype"))
keep = d.filter(pl.col("ptype") != "other").select("author_id").unique()
d = d.join(keep, on="author_id", how="semi")
mature = pl.col("snapshot_at_ms").is_null() | ((pl.col("snapshot_at_ms") - pl.col("created_at_ms")) >= 3 * 86_400_000)
d = d.filter(mature).sort("author_id", "created_at_ms")
print("posts of accounts with an own announcement:", d.height, "accounts:", d["author_id"].n_unique())

# trailing medians (previous K posts, current excluded), per account; views over posts that have views
def trailing(df, col, out):
    x = df.filter(pl.col(col).is_not_null()).select("post_id", "author_id", "created_at_ms", pl.col(col).cast(pl.Float64).alias("v")).sort("author_id", "created_at_ms")
    x = x.with_columns(pl.col("v").shift(1).rolling_median(window_size=K, min_samples=MINH).over("author_id").alias(out), pl.col("v").shift(1).rolling_quantile(0.9, window_size=K, min_samples=MINH).over("author_id").alias(out + "_p90"))
    return df.join(x.select("post_id", out, out + "_p90"), on="post_id", how="left")
d = trailing(d, "likes", "med_likes"); d = trailing(d, "views", "med_views")
d = d.with_columns(((pl.col("likes") + 1) / (pl.col("med_likes") + 1)).log().alias("lift_likes"), ((pl.col("views") + 1) / (pl.col("med_views") + 1)).log().alias("lift_views"),
                   (pl.col("likes") > pl.col("med_likes_p90")).alias("top10_likes"))
# spacing
d = d.with_columns(((pl.col("created_at_ms") - pl.col("created_at_ms").shift(1).over("author_id")) / 3_600_000).alias("gap_prev_h"),
                   ((pl.col("created_at_ms").shift(-1).over("author_id") - pl.col("created_at_ms")) / 3_600_000).alias("gap_next_h"))
# text attributes
raw = pl.col("text").fill_null("")
nolink = raw.str.replace_all(r"https?://\S+", "").str.strip_chars()
first = nolink.str.split("\n").list.first().fill_null("")
lc1 = first.str.to_lowercase().str.replace_all("’", "'")
ext = pl.col("urls").list.eval(~pl.element().struct.field("is_media").fill_null(False) & ~pl.element().struct.field("final_domain").fill_null("").str.contains(r"^(x\.com|twitter\.com|t\.co)$") & pl.element().struct.field("expanded").is_not_null()).list.any().fill_null(False)
dt = pl.from_epoch("created_at_ms", "ms")
d = d.with_columns(
    nolink.str.len_chars().alias("chars"),
    nolink.str.split("\n").list.eval(pl.element().str.strip_chars().str.len_chars() > 0).list.sum().alias("lines"),
    ext.alias("ext_link"),
    pl.col("_lc").str.contains(r"(link|details|more|demo|repo|try it|sign up|read more|docs|blog|paper|code)[^.\n]{0,25}\b(in|the) (the )?(first |next )?(reply|replies|comments?|thread)\b|\blink (below|in bio|👇)|\b(in|see) (the )?(reply|replies|comments) (below|👇)|\bbelow ?[👇⬇↓]").alias("points_to_reply"),
    (pl.col("_lc").str.contains(r"🧵|\bthread\b|\(?\b1/\s?(\d+|n|x)?\)?(\s|$)|\b1\)\s") | raw.str.strip_chars().str.contains(r"(👇|⬇️|↓|:)\s*(https?://\S+\s*)*$")).alias("thread_marker"),
    lc1.str.contains(r"\b(introducing|announcing|unveiling|launching|launched|launch|meet|say hello|releasing|released|open[- ]?sourcing|excited to|thrilled to|proud to|today,? (we|i))\b").alias("open_announce"),
    (lc1.str.contains(r"\?\s*$|^(tired of|sick of|stop|ever wonder|what if|why|how|most|every|everyone|nobody|no one)\b|\b(problem|broken|pain|painful|struggle|hard to|too slow|slow|expensive|waste|wasting|frustrat|bottleneck|sucks|can't|cannot|don't|doesn't|shouldn't|isn't|never)\b")).alias("open_problem"),
    lc1.str.contains(r"\b(sota|state[- ]of[- ]the[- ]art|\d+(\.\d+)?x (faster|cheaper|smaller|better|more)|faster|cheaper|beats|outperforms|#1|top[- ]ranked|record|best[- ]in[- ]class|\d+(\.\d+)?%)").alias("open_result"),
    lc1.str.contains(r"(^|[\s$€£#(])\d").alias("open_number"),
    lc1.str.contains(r"\?").alias("open_question"),
    first.str.contains(r"(?:[Ii]ntroducing|[Aa]nnouncing|[Mm]eet|[Ll]aunching|[Ll]aunch(?:ed)?|[Rr]eleas(?:e|ed|ing)|[Uu]nveiling|[Ss]ay hello to|[Oo]pen[- ]?sourc(?:ed|ing))\s+(?:the |our |my |a |an )?(?:new )?[@A-Z0-9]|\b[A-Z][a-z]+[A-Z]\w*|\b[A-Z][A-Za-z]*[- ]?\d+(?:\.\d+)*\b|@\w+").alias("names_product_first_line"),
    (pl.col("mentions_n") > 0).alias("tags"), (pl.col("mentions_n") >= 3).alias("tags3"), (pl.col("hashtags_n") > 0).alias("hashtag"),
    raw.str.contains(r"[\U0001F300-\U0001FAFF☀-➿]").alias("emoji"),
    lc1.str.contains(r"\b(we|we're|we've|our)\b").alias("open_we"),
    dt.dt.weekday().alias("dow"), dt.dt.hour().alias("hour_utc"),
    (pl.col("kind") == "quote").alias("is_quote"),
)
d = d.with_columns(pl.when(pl.col("has_video")).then(pl.lit("video")).when(pl.col("has_photo")).then(pl.lit("image")).otherwise(pl.lit("none")).alias("media_kind"),
                   pl.when(pl.col("followers_max").is_null()).then(pl.lit("unknown")).when(pl.col("followers_max") < 1000).then(pl.lit("<1k")).when(pl.col("followers_max") < 10000).then(pl.lit("1k-10k"))
                   .when(pl.col("followers_max") < 100000).then(pl.lit("10k-100k")).otherwise(pl.lit("100k+")).alias("tier"))
out = d.drop("media", "urls", "_lc", "_head")
out.write_parquet("data/launch/table.parquet")
t = out.filter(pl.col("ptype") != "other")
pl.Config.set_tbl_rows(60)
print(t.group_by("akind", "ptype").agg(pl.len().alias("posts"), pl.col("author_id").n_unique().alias("accounts"), pl.col("lift_likes").is_not_null().sum().alias("with_baseline"), pl.col("views").is_not_null().sum().alias("with_views"),
      pl.col("lift_views").is_not_null().sum().alias("views_baseline")).sort("akind", "posts"))
print(t.group_by("source", "media_kind").len().sort("source"))
