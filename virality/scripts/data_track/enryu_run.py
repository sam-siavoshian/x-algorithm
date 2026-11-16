"""enryu43/twitter100m: guards on every shard, per-account statistics, launch-type candidates, then conversion.

  pass 1  per shard: check_shard; per-account counts since 2020 (originals, replies, strong-tech posts, first-person shares);
          every original since 2020 that the launch rules type (xviral/text/launch_labels.py), with the flags the own-announcement test needs.
  select  accounts to convert: founder or company bio (xviral/text/bio.py), tech bio, an account of the current corpus, or 2+ typed first-person tech posts.
  pass 2  all originals and replies of those accounts in the post schema -> data/track_data/sources/twitter100m/posts/*.parquet
"""
import glob, json, os, sys, time
import polars as pl
from xviral.sources import twitter100m as T
from xviral.text.bio import bio_flags, bio_kind
from xviral.text.launch_labels import add_types
from xviral.text.tech import TECH_STRONG, is_tech_bio

RAW = "data/raw/enryu"; OUT = "data/track_data/sources/twitter100m"; os.makedirs(OUT + "/posts", exist_ok=True); os.makedirs(OUT + "/pass1", exist_ok=True)
users = T.all_users(glob.glob(RAW + "/users/data/*.parquet")[0])
shards = sorted(glob.glob(RAW + "/tweets/data/*.parquet"))
checks = []
for f in shards:
    name = os.path.basename(f).split(".")[0]
    if os.path.exists(f"{OUT}/pass1/{name}.stats.parquet"): continue
    t = time.time(); c = T.check_shard(f); c["shard"] = name
    b = T._base(T.Path(f), "2020-01-01").filter(pl.col("kind") != "repost").select("h", "id", "tweet", "kind", "likes", "retweets", "replies", "quotes", "created_at_ms").collect()
    txt = pl.col("tweet").fill_null("")
    st = b.group_by("h").agg(pl.len().alias("n"), (pl.col("kind") == "original").sum().alias("n_orig"), (pl.col("kind") == "reply").sum().alias("n_reply"),
                             ((pl.col("kind") == "original") & txt.str.contains(TECH_STRONG)).sum().alias("n_tech"),
                             ((pl.col("kind") == "original") & txt.str.contains(r"(?i)\b(i|i'm|i've|my|me)\b")).sum().alias("n_i"),
                             ((pl.col("kind") == "original") & txt.str.contains(r"(?i)\b(we|we're|we've|our|us)\b")).sum().alias("n_we"),
                             pl.col("likes").filter(pl.col("kind") == "original").median().alias("med_likes"), pl.col("created_at_ms").min().alias("first_ms"), pl.col("created_at_ms").max().alias("last_ms"))
    st.write_parquet(f"{OUT}/pass1/{name}.stats.parquet")
    cand = b.filter((pl.col("kind") == "original") & txt.str.slice(0, 400).str.contains(T.LAUNCH_HINT)).rename({"tweet": "text"})
    cand = add_types(cand).filter((pl.col("ptype_raw") != "other") & pl.col("techy") & ~pl.col("reprint") & ~pl.col("at_start")).drop("_lc", "_head")
    cand.write_parquet(f"{OUT}/pass1/{name}.typed.parquet")
    c["rows_2020_nonrepost"] = b.height; c["typed"] = cand.height; checks.append(c)
    print(name, c, round(time.time() - t, 1), flush=True)
    json.dump(checks, open(f"{OUT}/pass1/checks_{os.getpid()}.json", "w"))
if "--pass1" in sys.argv: sys.exit()

st = pl.scan_parquet(f"{OUT}/pass1/*.stats.parquet").group_by("h").agg(pl.col("n", "n_orig", "n_reply", "n_tech", "n_i", "n_we").sum(), pl.col("first_ms").min(), pl.col("last_ms").max()).collect()
ty = pl.scan_parquet(f"{OUT}/pass1/*.typed.parquet").filter(pl.col("fp_head")).group_by("h").agg(pl.len().alias("typed_fp")).collect()
corpus = pl.read_parquet("data/track_data/corpus_handles_filtered.parquet")["h"]
u = users.with_columns(pl.lit(None, dtype=pl.String).alias("author_display_name")).with_columns(bio_flags()).with_columns(bio_kind().alias("bio_kind"), is_tech_bio().alias("tech_bio"))
u = u.join(st, on="h", how="left").join(ty, on="h", how="left").with_columns(pl.col("typed_fp").fill_null(0), pl.col("h").is_in(corpus.implode()).alias("in_corpus"))
keep = u.filter((pl.col("bio_kind") != "none") | pl.col("tech_bio") | pl.col("in_corpus") | (pl.col("typed_fp") >= 2)).filter(pl.col("n_orig").fill_null(0) > 0)
u.write_parquet(f"{OUT}/accounts_all.parquet")
print("accounts kept", keep.height, keep.group_by("bio_kind").len().to_dicts(), "tech_bio", int(keep["tech_bio"].sum()), "in_corpus", int(keep["in_corpus"].sum()), flush=True)
n = 0
for f in shards:
    df = T.read_shard_accounts(f, users, keep["h"])
    df.write_parquet(f"{OUT}/posts/{os.path.basename(f).split('.')[0]}.parquet", compression="zstd"); n += df.height
print("posts written", n)
