"""Merge the per-account files of the brand set into one posts table and one thread table, and count what it adds."""
import json
import polars as pl
from xviral.text.launch_labels import add_types, own_expr

O = "data/track_data/sources/zenodo_brands"
ck = json.load(open(O + "/file_checks.json"))
d = pl.read_parquet(O + "/posts/*.parquet").unique("post_id", keep="first", maintain_order=True)
s = pl.read_parquet(O + "/threads/*.parquet").unique("post_id", keep="first", maintain_order=True)
acc = d.group_by("author_id").agg(pl.len().alias("n")).filter(pl.col("n") >= 50)["author_id"]
d = d.with_columns(pl.col("author_id").is_in(acc.implode()).alias("_brand"))
other = d.filter(~pl.col("_brand")).height
d = d.filter(pl.col("_brand")).drop("_brand"); s = s.join(d.select("post_id"), on="post_id", how="semi")
d.write_parquet(O + "/posts.parquet", compression="zstd"); s.write_parquet(O + "/threads.parquet", compression="zstd")
snap = d["snapshot_at_ms"][0]; age_d = (snap - pl.col("created_at_ms")) / 86400000
V = pl.col("views").is_not_null()
x = d.filter(pl.col("kind").is_in(["original", "quote"]) & (pl.col("lang_src") == "en"))
x = add_types(x).with_columns(pl.lit("company").alias("akind"), pl.col("media").list.eval(pl.element().struct.field("type").is_in(["video", "gif"])).list.any().alias("has_video")).with_columns(own_expr().alias("own"))
t = x.filter(pl.col("own")).with_columns(pl.when(pl.col("author_followers") < 1000).then(pl.lit("under_1k")).when(pl.col("author_followers") < 10000).then(pl.lit("1k_10k")).when(pl.col("author_followers") < 100000).then(pl.lit("10k_100k")).otherwise(pl.lit("100k_plus")).alias("tier"))
t.drop("_lc", "_head").select("post_id", "author_id", "author_handle", "created_at_ms", "text", "ptype_raw", "views", "likes", "reposts", "replies", "bookmarks", "author_followers", "tier", "has_video", "fp_head", "opener").write_parquet(O + "/typed_own_posts.parquet")
# threads: roots with a self-reply, and launch posts whose first self-reply carries a link
sr = s.filter(pl.col("is_self_reply")).join(d.select("post_id", "created_at_ms", "text", pl.col("urls").list.len().alias("n_urls"), "views"), on="post_id")
roots_with_thread = sr.join(d.filter(~pl.col("is_reply")).select(pl.col("post_id").alias("in_reply_to_post_id"), pl.col("created_at_ms").alias("root_ms"), pl.col("views").alias("root_views")), on="in_reply_to_post_id")
first = roots_with_thread.sort("created_at_ms").unique("in_reply_to_post_id", keep="first")
lk = first.filter((pl.col("n_urls") > 0) & ((pl.col("created_at_ms") - pl.col("root_ms")) <= 3600_000))
launch_ids = t.filter(pl.col("ptype_raw").is_in(["launch", "open_source"]))["post_id"]
counts = {
    "source": "Zenodo 21783628, twitterapi.io pages", "licence": "none stated on the record", "files": len(ck), "raw_rows": sum(c["raw"] for c in ck), "unique_after_dedup_in_file": sum(c["unique"] for c in ck),
    "dropped_time_mismatch_or_pre_2010_id": sum(c.get("time_mismatch", 0) for c in ck), "min_share_likes_le_views_per_file": min(c["likes_le_views"] for c in ck if c.get("likes_le_views") is not None and c["with_views"] >= 200),
    "rows_by_other_authors_dropped": other, "posts": d.height, "accounts": d["author_id"].n_unique(), "accounts_with_bio": d.filter(pl.col("author_bio").is_not_null())["author_id"].n_unique(),
    "date_min": str(pl.from_epoch(pl.Series([d["created_at_ms"].min()]), "ms")[0]), "date_max": str(pl.from_epoch(pl.Series([d["created_at_ms"].max()]), "ms")[0]), "snapshot": str(pl.from_epoch(pl.Series([snap]), "ms")[0]),
    "by_kind": d.group_by("kind").agg(pl.len().alias("posts"), V.sum().alias("with_views")).sort("posts", descending=True).to_dicts(),
    "with_views": d.filter(V).height, "with_views_older_than_3_days": d.filter(V & (age_d >= 3)).height, "posts_2023_on": d.filter(pl.col("created_at_ms") >= 1672531200000).height,
    "english": d.filter(pl.col("lang_src") == "en").height, "with_media": d.filter(pl.col("media").list.len() > 0).height,
    "with_video": d.filter(pl.col("media").list.eval(pl.element().struct.field("type") == "video").list.any()).height,
    "with_video_file_url": d.filter(pl.col("media").list.eval(pl.element().struct.field("video_url").is_not_null()).list.any()).height,
    "accounts_by_follower_tier": d.group_by("author_id").agg(pl.col("author_followers").max().alias("f")).select((pl.col("f") < 1000).sum().alias("under_1k"), ((pl.col("f") >= 1000) & (pl.col("f") < 10000)).sum().alias("1k_10k"),
                                                                                                        ((pl.col("f") >= 10000) & (pl.col("f") < 100000)).sum().alias("10k_100k"), (pl.col("f") >= 100000).sum().alias("100k_plus")).to_dicts()[0],
    "own_typed_posts": t.group_by("ptype_raw").agg(pl.len().alias("posts"), V.sum().alias("with_views"), pl.col("author_id").n_unique().alias("accounts")).sort("posts", descending=True).to_dicts(),
    "own_launch_posts_by_tier": t.filter(pl.col("ptype_raw").is_in(["launch", "open_source"])).group_by("tier").agg(pl.len().alias("posts"), V.sum().alias("with_views"), pl.col("author_id").n_unique().alias("accounts")).to_dicts(),
    "replies": s.filter(pl.col("in_reply_to_post_id").is_not_null()).height, "self_replies": s.filter(pl.col("is_self_reply")).height, "self_replies_with_parent_in_set": roots_with_thread.height + sr.join(d.filter(pl.col("is_reply")).select(pl.col("post_id").alias("in_reply_to_post_id")), on="in_reply_to_post_id", how="semi").height,
    "root_posts_with_a_self_reply": first.height, "root_posts_with_views_and_a_self_reply": first.filter(pl.col("root_views").is_not_null()).height,
    "root_posts_whose_first_self_reply_within_1h_has_a_link": lk.height, "of_those_with_views": lk.filter(pl.col("root_views").is_not_null()).height,
    "launch_posts_with_a_self_reply": first.filter(pl.col("in_reply_to_post_id").is_in(launch_ids.implode())).height, "launch_posts_with_link_in_first_self_reply": lk.filter(pl.col("in_reply_to_post_id").is_in(launch_ids.implode())).height,
    "accounts_with_5plus_link_in_reply_roots_and_views": lk.filter(pl.col("root_views").is_not_null()).join(d.select(pl.col("post_id").alias("in_reply_to_post_id"), "author_id"), on="in_reply_to_post_id").group_by("author_id").len().filter(pl.col("len") >= 5).height,
}
json.dump(counts, open(O + "/counts.json", "w"), indent=1, default=str)
print(json.dumps(counts, indent=1, default=str))
