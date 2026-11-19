#!/usr/bin/env python3
"""The launch track's 582 test-period launch posts scored through the draft path with a rewritten feature cache
(bench_api.py), i.e. with the checkpoint's own embedding. Same procedure as the replicate track's rep_score_launch.py.

    python research/tracks/embed_api/score_launch_api.py --cache data/api/bench/<v>/launch_cache.pkl --sid-codebook data/api/table_<v>/sid_codebook.npz \
        --out data/api/launch name=ckpt.pt ...      -> <out>/<name>.parquet (post_id, pred_likes)
"""
import argparse, logging, os, pickle
import polars as pl
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--history", type=int, default=56); ap.add_argument("--cache", required=True)
ap.add_argument("--sid-codebook", required=True); ap.add_argument("ckpt", nargs="+"); args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s"); log = logging.getLogger("launch")
from xviral.serve import DraftScorer
os.makedirs(args.out, exist_ok=True)
T = pl.read_parquet("data/tracks/launch/launch_scores.parquet").select("post_id", "author_id")
COLS = ["post_id", "author_id", "author_handle", "author_display_name", "created_at_ms", "snapshot_at_ms", "text", "kind", "media", "urls",
        "quoted_post_id", "quoted_text", "likes", "replies", "reposts", "quotes", "views", "author_followers", "author_following", "source"]
d = pl.scan_parquet("data/work/filtered/posts.parquet").select(COLS).filter(pl.col("author_id").is_in(T["author_id"].unique().implode())).collect()
by_author = {a: g.sort("created_at_ms") for (a,), g in d.group_by("author_id")}
fcache = pickle.load(open(args.cache, "rb"))


def no_encoder(texts):
    raise RuntimeError("a post was not in the feature cache: the cache does not match these posts")


for spec in args.ckpt:
    name, path = spec.split("=", 1)
    if os.path.exists(f"{args.out}/{name}.parquet"): continue
    sc = DraftScorer(path, no_encoder, device="cpu", sid_codebook=args.sid_codebook)
    sc.feature_cache = fcache
    n0, recs = len(fcache), []
    for r in T.iter_rows(named=True):
        g = by_author[r["author_id"]]; x = g.filter(pl.col("post_id") == r["post_id"]).row(0, named=True)
        hist = g.filter(pl.col("created_at_ms") < x["created_at_ms"]).tail(args.history)
        account = {"author_id": r["author_id"], "handle": x["author_handle"], "display_name": x["author_display_name"], "followers": x["author_followers"], "following": x["author_following"]}
        rep = sc.score(hist, [{"text": x["text"], "kind": x["kind"], "media": x["media"], "urls": x["urls"], "quoted_text": x["quoted_text"]}], account, x["created_at_ms"])[0]
        recs.append({"post_id": r["post_id"], "pred_likes": float(rep["likes"]["q50"])})
    assert len(fcache) == n0, "cache grew: some posts were not in it"
    pl.DataFrame(recs).write_parquet(f"{args.out}/{name}.parquet"); log.info("scored %s", name)
