"""Counts for the replies pass over the 2026 view corpus; recovers the parent of each self-reply by time."""
import glob, json
import polars as pl
from xviral.sources import nitter_sample as N
from xviral.text.launch_labels import add_types

O = "data/track_data/sources/nitter_2026_replies"
st = [json.load(open(f)) for f in sorted(glob.glob(O + "/stats/*.json"))]
COLS = ["post_id", "author_handle", "created_at_ms", "created_at_src_ms", "is_reply", "views", "likes"]
P = pl.scan_parquet(O + "/posts/*.parquet").select(COLS).collect().unique("post_id", keep="first")
S = pl.read_parquet(O + "/threads/*.parquet").unique("post_id", keep="first")
# the first run wrote has_link from a null test on `external-link`, which is an empty string when absent: recompute it from the posts
hl = pl.scan_parquet(O + "/posts/*.parquet").filter(pl.col("is_reply")).select("post_id", ((pl.col("urls").list.len() > 0) | pl.col("text").fill_null("").str.contains(N.LINK_IN_TEXT)).alias("has_link")).collect().unique("post_id")
S = S.drop("has_link").join(hl, on="post_id", how="left").with_columns(pl.col("has_link").fill_null(False))
S.write_parquet(O + "/threads.parquet", compression="zstd")
h = pl.col("author_handle").str.to_lowercase().alias("h")
P = P.with_columns(h)
cutoff = 1777593600000  # 2026-05-01
mature = pl.col("created_at_ms") < cutoff
link = N.link_self_replies(P, S, 30)
link.write_parquet(O + "/self_reply_links.parquet")
tr = pl.read_parquet("data/track_data/corpus_handles_train.parquet").select("h"); fl = pl.read_parquet("data/track_data/corpus_handles_filtered.parquet").select("h")
corpus_ids = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id").collect()
def grp(d):
    return {"posts": d.height, "accounts": d["h"].n_unique(), "mature_before_2026_05": d.filter(mature).height}
R = P.filter(pl.col("is_reply")); roots = P.filter(~pl.col("is_reply"))
Sj = S.join(P.select("post_id", "h", "created_at_ms", "views"), on="post_id")
selfr = Sj.filter(pl.col("is_self_reply"))
lk = link.join(S.select("post_id", "has_link"), on="post_id").join(P.select(pl.col("post_id").alias("parent_post_id"), "h", pl.col("created_at_ms").alias("parent_ms"), pl.col("views").alias("parent_views")), on="parent_post_id")
first = lk.filter(pl.col("parent_is_root")).sort("gap_ms").unique("parent_post_id", keep="first")
typed = []
for f in sorted(glob.glob(O + "/posts/*.parquet")):       # one shard at a time: the text of every root does not fit in memory at once
    r = pl.scan_parquet(f).filter(~pl.col("is_reply") & pl.col("text").fill_null("").str.slice(0, 400).str.contains(r"(?i)(introduc|announc|unveil|launch|releas|live|here|hello|meet |open[- ]?sourc|available|beta|waitlist|out now)")).select("post_id", pl.col("author_handle").str.to_lowercase().alias("h"), "text", "created_at_ms", "views", "likes").collect()
    typed.append(add_types(r).filter(pl.col("ptype_raw").is_in(["launch", "open_source"]) & pl.col("techy") & ~pl.col("reprint") & ~pl.col("at_start") & pl.col("fp_head")).drop("_lc", "_head"))
typed = pl.concat(typed).unique("post_id")
typed.write_parquet(O + "/launch_typed_roots_kept_accounts.parquet")
la = pl.read_parquet(O + "/launch_any/*.parquet").with_columns(pl.col("id").cast(pl.Int64).alias("post_id")).unique("post_id")
la.write_parquet(O + "/launch_any.parquet")
lam = la.filter(pl.col("date") < "2026-05-01")
agree = P.select(((pl.col("created_at_ms") - pl.col("created_at_src_ms")).abs() <= 60_000).mean()).item()
counts = {
    "source": "SinclairSchneider/tweets_sample_2026 (Hugging Face), all 73 shards; replies and roots of a list of accounts", "licence": "none stated",
    "shards": len(st), "rows_scanned": sum(s["rows"] for s in st), "replies_in_source": sum(s["replies"] for s in st), "reposts_in_source": sum(s["reposts"] for s in st),
    "keep_list_handles": pl.read_parquet("data/track_data/nitter_keep_handles.parquet").height,
    "kept_posts": P.height, "kept_accounts": P["h"].n_unique(), "share_likes_le_views": float((P["likes"].fill_null(0) <= P["views"].fill_null(0)).mean()), "share_time_within_one_minute_of_id": agree,
    "kept_roots": grp(roots), "kept_roots_already_in_corpus": roots.join(corpus_ids, on="post_id", how="semi").height, "kept_replies": grp(R),
    "replies_by_training_accounts": grp(R.join(tr, on="h", how="semi")), "replies_by_corpus_accounts": grp(R.join(fl, on="h", how="semi")),
    "self_replies": {"all": selfr.height, "accounts": selfr["h"].n_unique(), "by_training_accounts": selfr.join(tr, on="h", how="semi").height, "by_corpus_accounts": selfr.join(fl, on="h", how="semi").height,
                     "mature_before_2026_05": selfr.filter(mature).height, "with_a_link": int(selfr["has_link"].sum())},
    "self_replies_with_parent_recovered_within_30_min": link.height, "of_those_parent_is_a_root_post": int(link["parent_is_root"].sum()),
    "root_posts_with_a_recovered_self_reply": first.height, "of_those_first_self_reply_has_a_link": int(first["has_link"].sum()),
    "root_posts_with_link_in_first_self_reply_mature": first.filter(pl.col("has_link") & (pl.col("parent_ms") < cutoff)).height,
    "accounts_with_3plus_such_roots": first.filter(pl.col("has_link")).group_by("h").len().filter(pl.col("len") >= 3).height,
    "launch_typed_roots_of_kept_accounts": {"posts": typed.height, "mature": typed.filter(mature).height, "with_a_recovered_self_reply": typed.join(first.select(pl.col("parent_post_id").alias("post_id")), on="post_id", how="semi").height,
                                            "with_link_in_first_self_reply": typed.join(first.filter(pl.col("has_link")).select(pl.col("parent_post_id").alias("post_id")), on="post_id", how="semi").height},
    "launch_any_first_person_strong_tech_roots_all_accounts": {"posts": la.height, "accounts": la["h"].n_unique(), "mature_before_2026_05": lam.height, "mature_accounts": lam["h"].n_unique(),
                                                               "by_type_mature": lam.group_by("ptype_raw").len().sort("len", descending=True).to_dicts(), "mature_already_in_corpus": lam.join(corpus_ids, on="post_id", how="semi").height,
                                                               "mature_median_views": lam["views"].median(), "mature_views_under_1000": lam.filter(pl.col("views") < 1000).height},
}
json.dump(counts, open(O + "/counts.json", "w"), indent=1, default=str)
print(json.dumps(counts, default=str))
