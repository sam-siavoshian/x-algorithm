#!/usr/bin/env python3
"""Score every test-period launch-type post that has a same-account launch partner, through the draft path (the same
path the product uses): each post is scored at its own time against the account's previous 56 posts.

    python scripts/launch/score_launch.py --checkpoint a.pt --checkpoint b.pt ... --out data/launch/launch_scores.parquet
"""
import argparse, datetime as dt, json, logging, os, sys
import numpy as np, polars as pl
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", action="append", required=True); ap.add_argument("--out", required=True); ap.add_argument("--history", type=int, default=56)
ap.add_argument("--sid-codebook", default="data/work/emb/sid_codebook.npz"); ap.add_argument("--types", default="launch,open_source"); ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s"); log = logging.getLogger("launch")
from xviral.embed.text_small import SentenceEmbedder
from xviral.serve import DraftScorer
CUT = int(dt.datetime(2026, 4, 23, tzinfo=dt.timezone.utc).timestamp() * 1000)
T = pl.read_parquet("data/launch/table.parquet").filter(pl.col("ptype").is_in(args.types.split(",")) & (pl.col("created_at_ms") >= CUT))
T = T.filter(pl.len().over("author_id") >= 2).select("post_id", "author_id", "ptype", "akind")
if args.limit: T = T.head(args.limit)
COLS = ["post_id", "author_id", "author_handle", "author_display_name", "created_at_ms", "snapshot_at_ms", "text", "kind", "media", "urls",
        "quoted_post_id", "quoted_text", "likes", "replies", "reposts", "quotes", "views", "author_followers", "author_following", "source"]
d = pl.scan_parquet("data/work/filtered/posts.parquet").select(COLS).filter(pl.col("author_id").is_in(T["author_id"].unique().implode())).collect()
by_author = {a: g.sort("created_at_ms") for (a,), g in d.group_by("author_id")}
log.info("%d posts to score, %d accounts", T.height, T["author_id"].n_unique())
enc = SentenceEmbedder("BAAI/bge-small-en-v1.5", device="cpu")
scorer = DraftScorer(args.checkpoint if len(args.checkpoint) > 1 else args.checkpoint[0], lambda t: enc.encode(t), device="cpu", sid_codebook=args.sid_codebook)
lines = args.out + ".jsonl"
done = {json.loads(x)["post_id"] for x in open(lines)} if os.path.exists(lines) else set()
sink = open(lines, "a")
for i, r in enumerate(T.iter_rows(named=True)):
    if r["post_id"] in done: continue
    g = by_author[r["author_id"]]; x = g.filter(pl.col("post_id") == r["post_id"]).row(0, named=True)
    hist = g.filter(pl.col("created_at_ms") < x["created_at_ms"]).tail(args.history)
    account = {"author_id": r["author_id"], "handle": x["author_handle"], "display_name": x["author_display_name"], "followers": x["author_followers"], "following": x["author_following"]}
    try:
        rep = scorer.score(hist, [{"text": x["text"], "kind": x["kind"], "media": x["media"], "urls": x["urls"], "quoted_text": x["quoted_text"]}], account, x["created_at_ms"])[0]
    except Exception as e:  # noqa: BLE001
        log.warning("post %d skipped: %s", i, e); continue
    rec = {"post_id": r["post_id"], "author_id": r["author_id"], "history": hist.height, "pred_likes": float(rep["likes"]["q50"]), "pred_views": float(rep["views_24h"]["q50"]),
           "lift_vs_own_median": rep.get("lift_vs_own_median"), "members_likes": [m["likes_q50"] for m in rep.get("members", [])]}
    sink.write(json.dumps(rec, default=float) + "\n"); sink.flush()
    if (i + 1) % 50 == 0: log.info("scored %d/%d", i + 1, T.height)
sink.close()
R = pl.DataFrame([json.loads(x) for x in open(lines)], infer_schema_length=None)
R.write_parquet(args.out); print("wrote", args.out, R.height)
