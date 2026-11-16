"""One streaming pass over SinclairSchneider/tweets_sample_2026 (73 shards, 25.7 GB): download, extract, delete.

Keeps (a) every root post and reply of the accounts in the keep list (current corpus accounts, list handles from YC and
Product Hunt, founder / company / tech-bio accounts of twitter100m), with the reply side table; (b) from every account,
root posts the launch rules type as a first-person tech announcement and that carry strong tech vocabulary.
"""
import os, sys, json, time, glob
import polars as pl
from huggingface_hub import hf_hub_download, list_repo_files
from xviral.sources import nitter_sample as N
from xviral.text.launch_labels import add_types
from xviral.text.tech import TECH_STRONG

REPO = "SinclairSchneider/tweets_sample_2026"; OUT = "data/track_data/sources/nitter_2026_replies"; TMP = "data/raw/nitter_stream"
for d in ("posts", "threads", "launch_any", "stats"): os.makedirs(f"{OUT}/{d}", exist_ok=True)
keep = pl.read_parquet("data/track_data/nitter_keep_handles.parquet")["h"]
files = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
for i, f in enumerate(files):
    name = os.path.basename(f).split(".")[0]
    if os.path.exists(f"{OUT}/stats/{name}.json"): continue
    t = time.time()
    p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=TMP)
    posts, side = N.read_shard_accounts(p, keep)
    posts.write_parquet(f"{OUT}/posts/{name}.parquet", compression="zstd"); side.write_parquet(f"{OUT}/threads/{name}.parquet", compression="zstd")
    lf = N._roots(pl.scan_parquet(p)).filter(pl.col("text").fill_null("").str.slice(0, 400).str.contains(r"(?i)(introduc|announc|unveil|launch|releas|live|here|hello|meet |shipped|open[- ]?sourc|raised|hiring|available|beta|waitlist|out now)"))
    c = lf.select("id", "text", "name", N._HANDLE.alias("h"), "date", pl.col("comments").alias("replies"), pl.col("retweets").alias("reposts"), pl.col("quotes").alias("likes"), pl.col("likes").alias("views"),
                  (pl.col("pictures").list.len().fill_null(0) > 0).alias("has_photo"), "external-link").collect()
    c = add_types(c).filter(pl.col("ptype_raw").is_in(["launch", "open_source", "funding", "feature", "milestone", "hiring"]) & pl.col("techy") & ~pl.col("reprint") & ~pl.col("at_start") & pl.col("fp_head")
                            & pl.col("text").str.contains(TECH_STRONG)).drop("_lc", "_head")
    c.write_parquet(f"{OUT}/launch_any/{name}.parquet")
    tot = pl.scan_parquet(p).select(pl.len().alias("rows"), pl.col("is-retweet").fill_null(False).sum().alias("reposts"), (pl.col("replying-to").list.len().fill_null(0) > 0).sum().alias("replies")).collect().row(0, named=True)
    tot.update(shard=name, kept=posts.height, kept_replies=side.height, kept_self_replies=int(side["is_self_reply"].sum()), launch_any=c.height, secs=round(time.time() - t, 1))
    json.dump(tot, open(f"{OUT}/stats/{name}.json", "w")); os.remove(p)
    print(i, tot, flush=True)
print("DONE")
