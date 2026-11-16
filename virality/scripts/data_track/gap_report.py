"""What the additions do to each gap of the startup corpus, in numbers. Writes data/track_data/gaps.json."""
import json
import polars as pl
from xviral.text.tech import TECH_STRONG

T = "data/track_data"; S = T + "/sources"
A = pl.read_parquet(T + "/startup_accounts.parquet")
acc = A.select(pl.col("handle").alias("h"), pl.col("kind").alias("s_kind"), "kind_from", "evidence_n", pl.col("followers").alias("s_followers"))
strong_acc = acc.filter((pl.col("kind_from") != "bio") | (pl.col("evidence_n") >= 2))          # list, brand set, launch-track rules, or a bio backed by a second kind of evidence
tr = pl.read_parquet(T + "/corpus_handles_train.parquet").select("h"); fl = pl.read_parquet(T + "/corpus_handles_filtered.parquet").select("h")
bios = pl.read_parquet(T + "/account_bios.parquet")
V = pl.col("views").is_not_null()
g = {}

# ---- baseline, from the launch track's tables
lp = pl.read_parquet("data/tracks/launch/launch_posts.parquet").filter(pl.col("own")).with_columns(pl.col("author_handle").str.to_lowercase().alias("h"))
L0 = lp.filter(pl.col("ptype") == "launch"); cf0 = L0.filter(pl.col("akind").is_in(["company", "founder"]))
g["baseline"] = {"launch_posts": L0.height, "company_or_founder_launch_posts": cf0.height, "company_or_founder_launch_posts_with_views": cf0.filter(V).height,
                 "company_or_founder_launch_posts_under_1k_followers": cf0.filter(pl.col("tier") == "under_1k").height if "under_1k" in cf0["tier"].unique().to_list() else cf0.filter(pl.col("followers_max") < 1000).height,
                 "tiers_seen": cf0.group_by("tier").len().to_dicts(), "accounts_with_bio": 0, "replies": 0}

# ---- gap 1 and 2: launch posts, and company / founder launch posts with views
# (a) the current corpus re-read with the larger account list: launch posts by accounts the launch track called "other"
re = L0.filter(pl.col("akind") == "other").join(strong_acc, on="h", how="inner")
g["relabel_current_corpus"] = {"launch_posts_by_newly_identified_company_or_founder_accounts": re.height, "of_those_with_views": re.filter(V).height, "accounts": re["h"].n_unique(), "by_evidence": re.group_by("kind_from").len().to_dicts()}
# (b) twitter100m
t = pl.read_parquet(S + "/twitter100m/typed_own_posts.parquet").filter(pl.col("ptype_raw").is_in(["launch", "open_source"])).join(pl.read_parquet(S + "/twitter100m/accounts_all.parquet").select("h", "tech_bio", "n_orig"), on="h", how="left")
t = t.join(acc, on="h", how="left").with_columns(pl.col("text").str.contains(TECH_STRONG).alias("strong"))
conv = pl.scan_parquet(S + "/twitter100m/posts/*.parquet").select(pl.col("author_handle").str.to_lowercase().alias("h")).unique().collect()
t = t.with_columns(pl.col("h").is_in(conv["h"].implode()).alias("timeline_converted"), (pl.col("s_kind").is_not_null() | pl.col("strong") | pl.col("tech_bio")).alias("strict"))
def tt(d): return {"posts": d.height, "accounts": d["h"].n_unique(), "under_1k_followers": d.filter(pl.col("followers") < 1000).height, "under_1k_accounts": d.filter(pl.col("followers") < 1000)["h"].n_unique(),
                   "1k_to_10k": d.filter((pl.col("followers") >= 1000) & (pl.col("followers") < 10000)).height, "timeline_converted": d.filter(pl.col("timeline_converted")).height, "from_2022_on": d.filter(pl.col("created_at_ms") >= 1640995200000).height}
g["twitter100m_launch_posts"] = {"all_first_person_tech": tt(t), "strict_startup_account_or_strong_tech_text_or_tech_bio": tt(t.filter(pl.col("strict"))), "company_or_founder_account_any_evidence": tt(t.filter(pl.col("s_kind").is_not_null())),
                                 "company_or_founder_account_strong_evidence": tt(t.join(strong_acc.select("h"), on="h", how="semi")), "with_views": 0}
# (c) brand set
b = pl.read_parquet(S + "/zenodo_brands/typed_own_posts.parquet").filter(pl.col("ptype_raw").is_in(["launch", "open_source"]))
g["brand_set_launch_posts"] = {"posts": b.height, "with_views": b.filter(V).height, "accounts": b["author_id"].n_unique(), "under_1k_followers": b.filter(pl.col("author_followers") < 1000).height,
                               "under_1k_with_views": b.filter(V & (pl.col("author_followers") < 1000)).height, "1k_to_10k": b.filter((pl.col("author_followers") >= 1000) & (pl.col("author_followers") < 10000)).height,
                               "1k_to_10k_with_views": b.filter(V & (pl.col("author_followers") >= 1000) & (pl.col("author_followers") < 10000)).height, "with_video": int(b["has_video"].sum())}
# (d) the 2026 view corpus: first-person strong-tech launch posts of any account, mature only, not already in the corpus
corpus_ids = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id").collect()
la = pl.read_parquet(S + "/nitter_2026_replies/launch_any.parquet").filter((pl.col("date") < "2026-05-01") & pl.col("ptype_raw").is_in(["launch", "open_source"])).join(corpus_ids, on="post_id", how="anti").join(acc, on="h", how="left")
kt = pl.read_parquet(S + "/nitter_2026_replies/launch_typed_roots_kept_accounts.parquet").filter(pl.col("created_at_ms") < 1777593600000).join(corpus_ids, on="post_id", how="anti").join(acc, on="h", how="inner")
both = pl.concat([la.select("post_id", "h", "s_kind", "kind_from", "evidence_n", "s_followers"), kt.select("post_id", "h", "s_kind", "kind_from", "evidence_n", "s_followers")]).unique("post_id")
cfv = both.filter(pl.col("s_kind").is_not_null())
g["view_corpus_2026_launch_posts_not_in_corpus"] = {"mature_posts_any_account": la.height, "accounts": la["h"].n_unique(), "by_company_or_founder_accounts": cfv.height, "company_or_founder_accounts": cfv["h"].n_unique(),
                                                    "by_strong_evidence_accounts": cfv.join(strong_acc.select("h"), on="h", how="semi").height, "follower_count_known": cfv.filter(pl.col("s_followers").is_not_null()).height,
                                                    "under_1k_followers_known": cfv.filter(pl.col("s_followers") < 1000).height}
g["company_or_founder_launch_posts_with_views"] = {"baseline": g["baseline"]["company_or_founder_launch_posts_with_views"], "plus_relabelled_in_corpus": g["relabel_current_corpus"]["of_those_with_views"],
                                                   "plus_brand_set": g["brand_set_launch_posts"]["with_views"], "plus_view_corpus_2026": cfv.height}
g["company_or_founder_launch_posts_with_views"]["total"] = sum(v for k, v in g["company_or_founder_launch_posts_with_views"].items())

# ---- gap 4: bios
def cov(h): 
    j = h.join(bios.select("h", "bio_kind", "bio_source"), on="h", how="inner")
    return {"accounts": h.height, "with_bio": j.height, "share": round(j.height / h.height, 4), "by_source": j.group_by("bio_source").len().to_dicts(), "bio_reads_founder_or_company": j.filter(pl.col("bio_kind") != "none").height}
rows = pl.scan_parquet("data/work/dataset/rows.parquet").select(pl.col("author_handle").str.to_lowercase().alias("h"), "in_audience").collect()
lt = pl.read_parquet("data/tracks/launch/accounts_labeled.parquet").filter(pl.col("handle").fill_null("") != "").select(pl.col("handle").str.to_lowercase().alias("h"))
g["bios"] = {"bios_available_total": bios.height, "training_accounts": cov(tr), "training_rows_with_author_bio": rows.join(bios.select("h"), on="h", how="semi").height, "training_rows": rows.height,
             "audience_accounts": cov(rows.filter(pl.col("in_audience")).select("h").unique()), "corpus_accounts_4_48M": cov(fl), "launch_track_company_founder_accounts": cov(lt),
             "startup_accounts_with_bio_text": A.filter(pl.col("bio").is_not_null()).height, "startup_accounts": A.height}

# ---- gap 5: replies and threads
n = json.load(open(S + "/nitter_2026_replies/counts.json")); z = json.load(open(S + "/zenodo_brands/counts.json")); e = json.load(open(S + "/twitter100m/counts.json"))
g["replies"] = {"view_corpus_2026": {k: n[k] for k in ["kept_replies", "replies_by_training_accounts", "replies_by_corpus_accounts", "self_replies", "self_replies_with_parent_recovered_within_30_min", "root_posts_with_a_recovered_self_reply",
                                                         "of_those_first_self_reply_has_a_link", "root_posts_with_link_in_first_self_reply_mature", "accounts_with_3plus_such_roots", "launch_typed_roots_of_kept_accounts"]},
                "brand_set": {k: z[k] for k in ["replies", "self_replies", "self_replies_with_parent_in_set", "root_posts_with_a_self_reply", "root_posts_with_views_and_a_self_reply", "root_posts_whose_first_self_reply_within_1h_has_a_link",
                                                 "of_those_with_views", "launch_posts_with_a_self_reply", "launch_posts_with_link_in_first_self_reply", "accounts_with_5plus_link_in_reply_roots_and_views"]},
                "twitter100m": {"replies_to_other_accounts": e["converted_replies"], "self_replies_identifiable": 0}}
# ---- small accounts overall
g["small_accounts"] = {"twitter100m_converted_accounts_under_1k": next(x["len"] for x in e["converted_accounts_by_follower_tier"] if x["tier"] == "under_1k"), "brand_set_accounts_under_1k": z["accounts_by_follower_tier"]["under_1k"],
                       "startup_accounts_under_1k": A.filter(pl.col("follower_tier") == "under_1k").height, "startup_accounts_under_1k_with_posts_somewhere": A.filter((pl.col("follower_tier") == "under_1k") & (pl.col("n_posts_twitter100m").is_not_null() | pl.col("n_posts_corpus").is_not_null() | pl.col("n_posts_nitter2026").is_not_null() | pl.col("n_posts_brands").is_not_null())).height}
json.dump(g, open(T + "/gaps.json", "w"), indent=1, default=str)
print(json.dumps(g, default=str))
