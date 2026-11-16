"""startup_accounts.parquet: company and founder accounts with the evidence for each.

One row per account (key: lower-case handle). Four independent kinds of evidence, each kept as columns:
  list     the account is linked from a public launch list: the YC directory (company or founder Twitter URL) or a Product Hunt product page;
           or it is one of the 182 official brand accounts of Zenodo 21783628.
  bio      a profile bio that states a founder role or speaks as an organisation about a product (xviral/text/bio.py), with the bio text,
           its source and date: twitter100m users (April 2023), deberain/ChatGPT-Tweets (Dec 2022 to Feb 2023), Zenodo 21783628 (Dec 2025).
  rules    the launch track's label from display name and posting pattern (tracks/launch/accounts_labeled.parquet).
  posts    own first-person launch-type posts (xviral/text/launch_labels.py: launch, open-source, funding, feature, milestone), counted per source.
`kind` takes the first available of list role, brand set, bio, launch-track label. `evidence_n` counts the kinds present (posts counts from 2 posts).
"""
import glob, os
import polars as pl
from xviral.text.bio import bio_flags, bio_kind
from xviral.text.launch_labels import add_types

T = "data/track_data"; S = T + "/sources"
ANN = ["launch", "open_source", "funding", "feature", "milestone"]
low = lambda c: pl.col(c).str.to_lowercase()

# ---- lists
lists = pl.read_parquet(T + "/lists/list_handles.parquet")
L = lists.sort("list").group_by(pl.col("handle").alias("h")).agg(
    pl.col("list").unique().sort().str.join("|").alias("list_names"), pl.col("role").first().alias("list_role"), pl.col("role").n_unique().alias("_nr"), pl.col("name").first().alias("list_name"),
    pl.col("company").first().alias("list_company"), pl.col("batch_or_launch").first().alias("list_batch_or_launch"), pl.col("one_liner").first().alias("list_one_liner"), pl.col("website").first().alias("list_website"))
# a handle listed both as a company and as a founder (a founder who uses the company handle): keep company
L = L.with_columns(pl.when(pl.col("_nr") > 1).then(pl.lit("company")).otherwise(pl.col("list_role")).alias("list_role")).drop("_nr")

# ---- bios
e = pl.read_parquet(S + "/twitter100m/accounts_all.parquet").select("h", pl.col("author_handle").alias("handle_cased"), "author_id", pl.col("author_bio").alias("bio"), pl.col("followers"), pl.col("verified"),
                                                                    pl.lit("twitter100m_users").alias("bio_source"), pl.lit("2023-04").alias("bio_date"), pl.col("n_orig").alias("n_posts_twitter100m"), "n_reply")
c = pl.read_parquet(T + "/lists/chatgpt_tweets_accounts.parquet").select("h", pl.col("handle").alias("handle_cased"), pl.lit(None, dtype=pl.Int64).alias("author_id"), "bio", "followers", pl.lit(None, dtype=pl.Boolean).alias("verified"),
                                                                         pl.lit("chatgpt_tweets").alias("bio_source"), pl.col("bio_date").str.slice(0, 7), pl.lit(None, dtype=pl.UInt32).alias("n_posts_twitter100m"), pl.lit(None, dtype=pl.UInt32).alias("n_reply"))
bp = pl.scan_parquet(S + "/zenodo_brands/posts/*.parquet")
b = (bp.group_by("author_id").agg(pl.len().alias("n_posts_brands"), pl.col("author_handle").first(), pl.col("author_bio").drop_nulls().first().alias("bio"), pl.col("author_followers").max().alias("followers"),
                                  pl.col("author_display_name").first().alias("display_name"), pl.col("author_verified").first().alias("verified"), pl.col("views").is_not_null().sum().alias("n_views_brands"))
     .filter(pl.col("n_posts_brands") >= 50).collect().with_columns(low("author_handle").alias("h")))
bios = pl.concat([b.select("h", pl.col("author_handle").alias("handle_cased"), "author_id", "bio", "followers", "verified", pl.lit("zenodo_brands").alias("bio_source"), pl.lit("2025-12").alias("bio_date"),
                           pl.lit(None, dtype=pl.UInt32).alias("n_posts_twitter100m"), pl.lit(None, dtype=pl.UInt32).alias("n_reply")), e.select(c.columns), c], how="vertical_relaxed")
bios = bios.filter(pl.col("bio").fill_null("").str.len_chars() > 0).unique("h", keep="first", maintain_order=True)       # newest source first
bios = bios.with_columns(pl.col("bio").alias("author_bio"), pl.lit(None, dtype=pl.String).alias("author_display_name")).with_columns(bio_flags()).with_columns(bio_kind().alias("bio_kind")).drop("author_bio", "author_display_name", "bio_founder", "bio_company", "bio_tech", "bio_excluded")
bios.select("h", "handle_cased", "author_id", "bio", "bio_source", "bio_date", "bio_kind", "followers").write_parquet(T + "/account_bios.parquet")

# ---- launch track labels and the current corpus
lt = pl.read_parquet("data/tracks/launch/accounts_labeled.parquet").filter(pl.col("handle").fill_null("") != "").select(low("handle").alias("h"), pl.col("akind").alias("launch_track_kind"), pl.col("name").alias("lt_name"), pl.col("f").alias("lt_followers"))
F = pl.scan_parquet("data/work/filtered/posts.parquet").group_by(low("author_handle").alias("h")).agg(
    pl.len().alias("n_posts_corpus"), pl.col("views").is_not_null().sum().alias("n_views_corpus"), pl.col("author_followers").max().alias("corpus_followers"),
    pl.col("author_display_name").drop_nulls().first().alias("corpus_name"), pl.col("author_id").filter(pl.col("author_id") > 0).first().alias("corpus_author_id")).collect()
R = pl.scan_parquet("data/work/dataset/rows.parquet").group_by(low("author_handle").alias("h")).agg(pl.len().alias("n_training_rows")).collect()
lp = pl.read_parquet("data/tracks/launch/launch_posts.parquet").filter(pl.col("own") & pl.col("ptype").is_in(ANN)).group_by(low("author_handle").alias("h")).agg(
    pl.len().alias("n_ann_corpus"), pl.col("views").is_not_null().sum().alias("n_ann_corpus_views"), pl.col("text").sort_by("likes", descending=True).first().str.slice(0, 280).alias("ex_corpus"))

# ---- own announcements in the new sources
ty = pl.scan_parquet(S + "/twitter100m/pass1/*.typed.parquet").filter(pl.col("fp_head") & pl.col("ptype_raw").is_in(ANN)).group_by("h").agg(
    pl.len().alias("n_ann_twitter100m"), pl.col("text").sort_by("likes", descending=True).first().str.slice(0, 280).alias("ex_t100m")).collect()
na = pl.scan_parquet(S + "/nitter_2026_replies/launch_any/*.parquet").filter(pl.col("ptype_raw").is_in(ANN)).group_by("h").agg(
    pl.len().alias("n_ann_nitter2026"), pl.col("text").sort_by("likes", descending=True).first().str.slice(0, 280).alias("ex_nitter"), pl.col("name").first().alias("nitter_name")).collect()
npst = pl.scan_parquet(S + "/nitter_2026_replies/posts/*.parquet").group_by(low("author_handle").alias("h")).agg(
    pl.len().alias("n_posts_nitter2026"), pl.col("is_reply").sum().alias("n_replies_nitter2026"), pl.col("author_display_name").first().alias("nitter_name2")).collect()
bx = bp.filter(pl.col("kind").is_in(["original", "quote"]) & (pl.col("lang_src") == "en")).select("author_id", "text", "likes", "views").collect()
bx = add_types(bx).filter(pl.col("ptype_raw").is_in(ANN) & pl.col("techy") & ~pl.col("reprint") & ~pl.col("at_start") & (pl.col("fp_head") | pl.col("opener")))
bx = bx.group_by("author_id").agg(pl.len().alias("n_ann_brands"), pl.col("views").is_not_null().sum().alias("n_ann_brands_views"), pl.col("text").sort_by("likes", descending=True).first().str.slice(0, 280).alias("ex_brands"))
B = b.join(bx, on="author_id", how="left").select("h", "n_posts_brands", "n_views_brands", "n_ann_brands", "n_ann_brands_views", "ex_brands", pl.col("display_name").alias("brand_name"), pl.lit(True).alias("in_brand_set"))

# ---- assemble: every handle with list, brand, bio or rules evidence
cand = pl.concat([L.select("h"), B.select("h"), bios.filter(pl.col("bio_kind") != "none").select("h"), lt.select("h")]).unique()
A = cand
for t in (L, bios, lt, F, R, lp, ty, na, npst, B):
    A = A.join(t, on="h", how="left")
z = lambda c: pl.col(c).fill_null(0)
A = A.with_columns(
    pl.col("in_brand_set").fill_null(False), pl.col("list_names").is_not_null().alias("ev_list"), (pl.col("bio_kind").fill_null("none") != "none").alias("ev_bio"), pl.col("launch_track_kind").is_not_null().alias("ev_rules"),
    (z("n_ann_corpus") + z("n_ann_twitter100m") + z("n_ann_nitter2026") + z("n_ann_brands")).alias("n_own_announcements"),
    (z("n_ann_corpus_views") + z("n_ann_nitter2026") + z("n_ann_brands_views")).alias("n_own_announcements_with_views"),
    pl.coalesce("ex_corpus", "ex_brands", "ex_nitter", "ex_t100m").alias("example_announcement"),
    pl.coalesce("corpus_name", "brand_name", "list_name", "lt_name", "nitter_name", "nitter_name2").alias("display_name"),
    pl.coalesce("corpus_author_id", "author_id").alias("author_id"),
    pl.coalesce("followers", "corpus_followers", "lt_followers").alias("followers"),
    pl.when(pl.col("followers").is_not_null()).then(pl.col("bio_source")).when(pl.col("corpus_followers").is_not_null()).then(pl.lit("corpus")).otherwise(None).alias("followers_source"),
).with_columns(
    (pl.col("ev_list") | pl.col("in_brand_set")).alias("ev_list"), (pl.col("n_own_announcements") >= 2).alias("ev_posts"),
    # a Product Hunt product page sometimes links a person's account: there the bio decides
    pl.when((pl.col("list_names") == "producthunt_top") & ((pl.col("bio_kind") == "founder") | pl.col("bio_person").fill_null(False) | pl.col("bio").fill_null("").str.contains(r"(?i)\b(ceo|cto|co-?founder|founder)\b"))).then(pl.lit("founder"))
      .when(pl.col("list_role").is_not_null()).then(pl.col("list_role")).when(pl.col("in_brand_set")).then(pl.lit("company")).when(pl.col("bio_kind").fill_null("none") != "none").then(pl.col("bio_kind")).otherwise(pl.col("launch_track_kind")).alias("kind"),
    pl.when(pl.col("list_role").is_not_null()).then(pl.lit("list")).when(pl.col("in_brand_set")).then(pl.lit("brand_set")).when(pl.col("bio_kind").fill_null("none") != "none").then(pl.lit("bio")).otherwise(pl.lit("rules")).alias("kind_from"),
    pl.when(pl.col("followers").is_null()).then(pl.lit("unknown")).when(pl.col("followers") < 1000).then(pl.lit("under_1k")).when(pl.col("followers") < 10000).then(pl.lit("1k_10k"))
      .when(pl.col("followers") < 100000).then(pl.lit("10k_100k")).otherwise(pl.lit("100k_plus")).alias("follower_tier"),
).with_columns((pl.col("ev_list").cast(pl.Int8) + pl.col("ev_bio").cast(pl.Int8) + pl.col("ev_rules").cast(pl.Int8) + pl.col("ev_posts").cast(pl.Int8)).alias("evidence_n"),
               (pl.col("bio_kind").is_not_null() & (pl.col("bio_kind") != "none") & (pl.col("bio_kind") != pl.col("kind"))).alias("bio_disagrees"),
               (pl.col("launch_track_kind").is_not_null() & (pl.col("launch_track_kind") != pl.col("kind"))).alias("rules_disagree"))
cols = ["h", "handle_cased", "author_id", "display_name", "kind", "kind_from", "evidence_n", "ev_list", "ev_bio", "ev_rules", "ev_posts", "bio_disagrees", "rules_disagree",
        "list_names", "list_role", "list_company", "list_batch_or_launch", "list_one_liner", "list_website", "in_brand_set", "bio", "bio_kind", "bio_source", "bio_date", "launch_track_kind",
        "n_own_announcements", "n_own_announcements_with_views", "n_ann_corpus", "n_ann_twitter100m", "n_ann_nitter2026", "n_ann_brands", "example_announcement",
        "followers", "followers_source", "follower_tier", "n_training_rows", "n_posts_corpus", "n_views_corpus", "n_posts_twitter100m", "n_posts_nitter2026", "n_replies_nitter2026", "n_posts_brands", "n_views_brands"]
A = A.select(cols).rename({"h": "handle"}).sort(["evidence_n", "n_own_announcements"], descending=True)
A.write_parquet(T + "/startup_accounts.parquet")
print(A.height, "accounts")
print(A.group_by("kind").len().sort("kind")); print(A.group_by("evidence_n").len().sort("evidence_n")); print(A.group_by("kind_from", "kind").len().sort("kind_from", "kind"))
print(A.select(pl.col("ev_list").sum(), pl.col("ev_bio").sum(), pl.col("ev_rules").sum(), pl.col("ev_posts").sum(), pl.col("bio").is_not_null().sum().alias("has_bio_text"), pl.col("n_training_rows").is_not_null().sum().alias("in_training"),
               pl.col("n_posts_corpus").is_not_null().sum().alias("in_corpus"), pl.col("n_posts_twitter100m").is_not_null().sum().alias("in_t100m"), pl.col("n_posts_nitter2026").is_not_null().sum().alias("in_nitter"),
               pl.col("bio_disagrees").sum(), pl.col("rules_disagree").sum()))
print(A.group_by("follower_tier").len())
