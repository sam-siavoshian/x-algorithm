#!/usr/bin/env python3
"""Score test-period posts through the draft path (the path the product uses), one model at a time, sharing one
model-independent feature cache: every own announcement the launch track labelled from 2026-04-23 on, plus a control
sample of ordinary posts by the same accounts. Each post is scored at its own time against the account's previous 56 posts.

    python scripts/posttype/score_posts.py --checkpoint name=path/best.pt [--checkpoint ...] --out data/posttype/scores
Writes <out>/<name>.<shard>.parquet (post_id, pred_likes, lift_likes) and <out>/posts.parquet (the posts, actual lift, types).
"""
import argparse, datetime as dt, json, logging, os, pickle
import numpy as np, polars as pl
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", action="append", required=True); ap.add_argument("--out", default="data/posttype/scores"); ap.add_argument("--history", type=int, default=56)
ap.add_argument("--sid-codebook", default="data/work/emb/sid_codebook.npz"); ap.add_argument("--control", type=int, default=600); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--correction", default=None, help="a type-correction JSON applied inside the scorer (every checkpoint is then scored as one ensemble under the first name)")
ap.add_argument("--shard", default="0/1", help="i/n: accounts with author_id % n == i, so several single-threaded processes can share the work")
args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s"); log = logging.getLogger("score")
from xviral.embed.text_small import SentenceEmbedder
from xviral.serve import DraftScorer
os.makedirs(args.out, exist_ok=True)
CUT = int(dt.datetime(2026, 4, 23, tzinfo=dt.timezone.utc).timestamp() * 1000)
K, MINH = 30, 8
LP = pl.read_parquet("data/tracks/launch/launch_posts.parquet").filter(pl.col("created_at_ms") >= CUT).select("post_id", "author_id", "ptype", "akind")
COLS = ["post_id", "author_id", "author_handle", "author_display_name", "created_at_ms", "snapshot_at_ms", "text", "kind", "media", "urls",
        "quoted_post_id", "quoted_text", "likes", "replies", "reposts", "quotes", "views", "author_followers", "author_following", "source"]
d = pl.scan_parquet("data/work/filtered/posts.parquet").select(COLS).filter(pl.col("author_id").is_in(LP["author_id"].unique().implode())).collect()
# actual lift, as scripts/launch/build_table.py: against the median likes of the previous 30 mature posts (at least 8)
mature = pl.col("snapshot_at_ms").is_null() | ((pl.col("snapshot_at_ms") - pl.col("created_at_ms")) >= 3 * 86_400_000)
m = d.filter(mature & pl.col("likes").is_not_null()).sort("author_id", "created_at_ms").with_columns(
    pl.col("likes").cast(pl.Float64).shift(1).rolling_median(window_size=K, min_samples=MINH).over("author_id").alias("med_likes"))
m = m.filter(pl.col("med_likes").is_not_null() & (pl.col("created_at_ms") >= CUT)).select("post_id", "author_id", "created_at_ms", "likes", "med_likes", "source")
typed = m.join(LP.select("post_id", "ptype", "akind"), on="post_id")
other = m.join(LP.select("post_id"), on="post_id", how="anti").sort("post_id").sample(args.control, seed=0).with_columns(pl.lit("other").alias("ptype"), pl.lit(None, dtype=pl.String).alias("akind"))
# the launch track's 582 scored posts are all kept, also the few without a trailing median (they only enter the pairs)
ref = pl.read_parquet("data/tracks/launch/launch_scores.parquet").select("post_id").join(typed.select("post_id"), on="post_id", how="anti")
extra = d.join(ref, on="post_id").select("post_id", "author_id", "created_at_ms", "likes", pl.lit(None, dtype=pl.Float64).alias("med_likes"), "source").join(LP.select("post_id", "ptype", "akind"), on="post_id")
P = pl.concat([typed, other.select(typed.columns), extra.select(typed.columns)]).sort("author_id", "created_at_ms")
if args.limit: P = P.sample(args.limit, seed=0)
P.write_parquet(f"{args.out}/posts.parquet")
SI, SN = (int(v) for v in args.shard.split("/")); P = P.filter(pl.col("author_id").rank("dense") % SN == SI)
log.info("posts to score: %s", dict(P.group_by("ptype").len().iter_rows()))
by_author = {a: g.sort("created_at_ms") for (a,), g in d.group_by("author_id")}
enc = SentenceEmbedder("BAAI/bge-small-en-v1.5", device="cpu")
cache_path = f"{args.out}/feature_cache_{SI}.pkl"
cache = pickle.load(open(cache_path, "rb")) if os.path.exists(cache_path) else {}
n0 = len(cache)
scorers = {}
if args.correction:
    name = args.checkpoint[0].split("=", 1)[0]
    scorers[name] = DraftScorer([c.split("=", 1)[1] for c in args.checkpoint], lambda t: enc.encode(t), device="cpu", sid_codebook=args.sid_codebook, type_correction=args.correction); scorers[name].feature_cache = cache
for spec in ([] if args.correction else args.checkpoint):
    name, path = spec.split("=", 1)
    scorers[name] = DraftScorer(path, lambda t: enc.encode(t), device="cpu", sid_codebook=args.sid_codebook); scorers[name].feature_cache = cache
first = next(iter(scorers.values()))
for s in scorers.values(): s._emb_cache = first._emb_cache
log.info("stored embeddings reused for %d history posts", first.prefill_embeddings("data/work/model/table", d.filter(pl.col("author_id").is_in(P["author_id"].unique().implode()))["post_id"].to_numpy()))
recs = {n: [] for n in scorers}; ptypes = []
for i, r in enumerate(P.iter_rows(named=True)):
    g = by_author[r["author_id"]]; x = g.filter(pl.col("post_id") == r["post_id"]).row(0, named=True)
    hist = g.filter(pl.col("created_at_ms") < x["created_at_ms"]).tail(args.history)
    account = {"author_id": r["author_id"], "handle": x["author_handle"], "display_name": x["author_display_name"], "followers": x["author_followers"], "following": x["author_following"]}
    for n, sc in scorers.items():
        try:
            rep = sc.score(hist, [{"text": x["text"], "kind": x["kind"], "media": x["media"], "urls": x["urls"], "quoted_text": x["quoted_text"]}], account, x["created_at_ms"])[0]
        except Exception as e:  # noqa: BLE001
            log.warning("post %d skipped for %s: %s", i, n, e); continue
        recs[n].append({"post_id": r["post_id"], "pred_likes": float(rep["likes"]["q50"]), "lift_likes": rep["lift_vs_own_median"]["likes"], "own_median": rep["own_median"]["likes"], "history": hist.height})
    if (i + 1) % 100 == 0:
        log.info("scored %d/%d", i + 1, P.height)
    if (i + 1) % 500 == 0 and len(cache) > n0:
        pickle.dump(cache, open(cache_path + ".tmp", "wb")); os.replace(cache_path + ".tmp", cache_path); n0 = len(cache)
if len(cache) > n0:
    pickle.dump(cache, open(cache_path + ".tmp", "wb")); os.replace(cache_path + ".tmp", cache_path)
for n, rr in recs.items():
    pl.DataFrame(rr, schema={"post_id": pl.Int64, "pred_likes": pl.Float64, "lift_likes": pl.Float64, "own_median": pl.Float64, "history": pl.Int64}).write_parquet(f"{args.out}/{n}.{SI}.parquet")
    print("wrote", n, len(rr))
