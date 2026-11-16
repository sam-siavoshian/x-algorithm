#!/usr/bin/env python3
"""scripts/launch/score_launch.py with a feature cache: the same posts (every launch-type post from 2026-04-23 on of an
account with at least two), the same draft path and 56 posts of history; the model-independent features and embeddings
are kept in a pickle, so each further checkpoint scores in under a minute.

    python scripts/expand/score_launch.py --checkpoint a.pt [--checkpoint b.pt] --out data/expand/launch/<name>.parquet --cache data/expand/launch/cache.pkl
"""
import argparse, datetime as dt, json, logging, os, pickle
import numpy as np, polars as pl, torch
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", action="append", required=True); ap.add_argument("--out", required=True); ap.add_argument("--history", type=int, default=56)
ap.add_argument("--sid-codebook", default="data/work/emb/sid_codebook.npz"); ap.add_argument("--types", default="launch,open_source"); ap.add_argument("--cache", required=True)
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(args.threads)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s"); log = logging.getLogger("launch")
from xviral.embed.text_small import SentenceEmbedder
from xviral.serve import DraftScorer
CUT = int(dt.datetime(2026, 4, 23, tzinfo=dt.timezone.utc).timestamp() * 1000)
T = pl.read_parquet("data/launch/table.parquet").filter(pl.col("ptype").is_in(args.types.split(",")) & (pl.col("created_at_ms") >= CUT))
T = T.filter(pl.len().over("author_id") >= 2).select("post_id", "author_id", "ptype", "akind")
COLS = ["post_id", "author_id", "author_handle", "author_display_name", "created_at_ms", "snapshot_at_ms", "text", "kind", "media", "urls",
        "quoted_post_id", "quoted_text", "likes", "replies", "reposts", "quotes", "views", "author_followers", "author_following", "source"]
d = pl.scan_parquet("data/work/filtered/posts.parquet").select(COLS).filter(pl.col("author_id").is_in(T["author_id"].unique().implode())).collect()
by_author = {a: g.sort("created_at_ms") for (a,), g in d.group_by("author_id")}
log.info("%d posts to score, %d accounts", T.height, T["author_id"].n_unique())
have = os.path.exists(args.cache)
enc = SentenceEmbedder("BAAI/bge-small-en-v1.5", device="cpu")
scorer = DraftScorer(args.checkpoint if len(args.checkpoint) > 1 else args.checkpoint[0], lambda t: enc.encode(t), device="cpu", sid_codebook=args.sid_codebook)
scorer.emb_cache_limit = 10 ** 9
scorer.feature_cache = pickle.load(open(args.cache, "rb")) if have else {}
if not have:
    need = []
    for (a,), g in T.group_by("author_id"):
        h = by_author[a]; ts_ = h["created_at_ms"].to_numpy(); mark = h["post_id"].is_in(g["post_id"].implode()).to_numpy()
        for i in np.flatnonzero(mark.copy()):
            mark[max(0, i - args.history):i] = True
        need.append(h.filter(pl.Series(mark)))
    log.info("encoded ahead: %d texts", scorer.preencode(pl.concat(need), log=log))
recs = []
for i, r in enumerate(T.iter_rows(named=True)):
    g = by_author[r["author_id"]]; x = g.filter(pl.col("post_id") == r["post_id"]).row(0, named=True)
    hist = g.filter(pl.col("created_at_ms") < x["created_at_ms"]).tail(args.history)
    account = {"author_id": r["author_id"], "handle": x["author_handle"], "display_name": x["author_display_name"], "followers": x["author_followers"], "following": x["author_following"]}
    try:
        rep = scorer.score(hist, [{"text": x["text"], "kind": x["kind"], "media": x["media"], "urls": x["urls"], "quoted_text": x["quoted_text"]}], account, x["created_at_ms"])[0]
    except Exception as e:  # noqa: BLE001
        log.warning("post %d skipped: %s", i, e); continue
    recs.append({"post_id": r["post_id"], "author_id": r["author_id"], "history": hist.height, "pred_likes": float(rep["likes"]["q50"]), "pred_views": float(rep["views_24h"]["q50"]),
                 "pred_likes_q10": float(rep["likes"]["q10"]), "pred_likes_q90": float(rep["likes"]["q90"]),
                 "members_likes": [float(m["likes_q50"]) for m in rep.get("members", [])] or [float(rep["likes"]["q50"])]})
    if (i + 1) % 100 == 0: log.info("scored %d/%d", i + 1, T.height)
if not have:
    pickle.dump(scorer.feature_cache, open(args.cache, "wb"))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
pl.DataFrame(recs, infer_schema_length=None).write_parquet(args.out); print("wrote", args.out, len(recs))
