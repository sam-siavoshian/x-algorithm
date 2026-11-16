"""Counts for the converted twitter100m source: guards over all shards, accounts, bios, follower tiers, own launch-type posts."""
import glob, json
import polars as pl
from xviral.text.bio import bio_flags, bio_kind

O = "data/track_data/sources/twitter100m"
ck = [c for f in glob.glob(O + "/pass1/checks_*.json") for c in json.load(open(f))]
ck = list({c["shard"]: c for c in ck}.values())
P = pl.scan_parquet(O + "/posts/*.parquet")
u = pl.read_parquet(O + "/accounts_all.parquet").drop("bio_founder", "bio_company", "bio_person", "bio_tech", "bio_excluded", "bio_kind").with_columns(bio_flags()).with_columns(bio_kind().alias("bio_kind"))
tier = lambda c: (pl.when(pl.col(c) < 1000).then(pl.lit("under_1k")).when(pl.col(c) < 10000).then(pl.lit("1k_10k")).when(pl.col(c) < 100000).then(pl.lit("10k_100k")).otherwise(pl.lit("100k_plus")))
kept = P.group_by("author_id").agg(pl.len().alias("n"), (pl.col("kind") == "original").sum().alias("n_orig"), (pl.col("kind") == "reply").sum().alias("n_reply")).collect()
ku = u.join(kept, on="author_id", how="inner").with_columns(tier("followers").alias("tier"))
ty = pl.scan_parquet(O + "/pass1/*.typed.parquet").collect().join(u.select("h", "bio_kind", "followers", "in_corpus", "tech_bio"), on="h", how="left").with_columns(tier("followers").alias("tier"))
# the account's own announcement: first person in the opening, or a company bio and an announcement opener
own = ty.filter(pl.col("fp_head") | ((pl.col("bio_kind") == "company") & pl.col("opener")))
own.select("id", "h", "text", "ptype_raw", "likes", "retweets", "replies", "quotes", "created_at_ms", "bio_kind", "followers", "tier", "fp_head", "opener", "in_corpus").write_parquet(O + "/typed_own_posts.parquet")
corpus_ids = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id").collect()
L = own.filter(pl.col("ptype_raw").is_in(["launch", "open_source"]))
yr = pl.from_epoch("created_at_ms", "ms").dt.year().alias("year")
counts = {
    "source": "enryu43/twitter100m_tweets + twitter100m_users (Hugging Face; Zenodo 10.5281/zenodo.15086029)", "licence": "none stated",
    "shards": len(ck), "raw_rows": sum(c["rows"] for c in ck), "pre_snowflake_ids": sum(c["pre_snowflake"] for c in ck), "time_mismatch_rows_dropped": sum(c["time_mismatch"] for c in ck),
    "min_share_likes_ge_quotes": min(c["likes_ge_quotes"] / c["busy"] for c in ck), "min_share_likes_ge_reposts": min(c["likes_ge_reposts"] / c["busy"] for c in ck),
    "duplicate_ids_within_shards": sum(c["rows"] - c["unique_ids"] for c in ck), "last_post": str(pl.from_epoch(pl.Series([max(c["last_ms"] for c in ck)]), "ms")[0]),
    "rows_since_2020_not_repost": sum(c["rows_2020_nonrepost"] for c in ck),
    "users_table": u.height, "users_with_bio": u.filter(pl.col("author_bio").fill_null("").str.len_chars() > 0).height, "users_under_1k_followers": u.filter(pl.col("followers") < 1000).height,
    "users_bio_kind": u.group_by("bio_kind").len().to_dicts(), "users_in_current_corpus": int(u["in_corpus"].sum()),
    "converted_accounts": ku.height, "converted_posts": int(kept["n"].sum()), "converted_originals": int(kept["n_orig"].sum()), "converted_replies": int(kept["n_reply"].sum()),
    "converted_accounts_by_reason": {"founder_or_company_bio_current_rules": ku.filter(pl.col("bio_kind") != "none").height, "tech_bio": int(ku["tech_bio"].sum()), "in_current_corpus": int(ku["in_corpus"].sum()), "two_plus_first_person_typed_posts": ku.filter(pl.col("typed_fp") >= 2).height},
    "converted_accounts_by_follower_tier": ku.group_by("tier").len().to_dicts(),
    "converted_posts_by_year": P.select(yr, "kind").group_by("year").agg(pl.len().alias("posts"), (pl.col("kind") == "original").sum().alias("originals")).sort("year").collect().to_dicts(),
    "posts_also_in_current_corpus": P.select("post_id").collect().join(corpus_ids, on="post_id", how="semi").height,
    "typed_first_person_tech_posts_all_users_since_2020": ty.filter(pl.col("fp_head")).group_by("ptype_raw").agg(pl.len().alias("posts"), pl.col("h").n_unique().alias("accounts")).sort("posts", descending=True).to_dicts(),
    "own_launch_posts": {"all_accounts": L.height, "accounts": L["h"].n_unique(), "by_bio_kind": L.group_by("bio_kind").agg(pl.len().alias("posts"), pl.col("h").n_unique().alias("accounts")).to_dicts(),
                         "by_follower_tier": L.group_by("tier").agg(pl.len().alias("posts"), pl.col("h").n_unique().alias("accounts")).to_dicts(),
                         "founder_or_company_bio_by_tier": L.filter(pl.col("bio_kind") != "none").group_by("tier").agg(pl.len().alias("posts"), pl.col("h").n_unique().alias("accounts")).to_dicts(),
                         "by_year": L.with_columns(yr).group_by("year").len().sort("year").to_dicts(), "already_in_current_corpus": L.join(corpus_ids, left_on="id", right_on="post_id", how="semi").height,
                         "from_accounts_with_10plus_originals": L.join(u.filter(pl.col("n_orig") >= 10).select("h"), on="h", how="semi").height},
}
json.dump(counts, open(O + "/counts.json", "w"), indent=1, default=str)
print(json.dumps(counts, indent=0, default=str))
