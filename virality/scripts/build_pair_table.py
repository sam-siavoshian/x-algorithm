#!/usr/bin/env python3
"""Feature table of controlled pairs, built through the serving path.

    python scripts/build_pair_table.py --pairs pairs.parquet --posts <work>/filtered/posts.parquet --checkpoint <run>/best.pt \
        --table <work>/model/table --out <dir> [--shard i/n]

For each pair (post_id, post_id_b of one account) the two posts are handed to DraftScorer.featurize as alternative
drafts at the time of the earlier one, with the account's earlier posts as history: exactly what the pair benchmark
(scripts/evidence/tac_serve_eval.py) and the product do. The per-pair mini tables are concatenated into one table
whose rows are [history of pair 0, draft A, draft B, history of pair 1, ...]; every pair is its own "author" block,
so episodes.build(table, anchor=row of draft A, n_candidates=2) gives the serving episode. Labels (the real like
counts of both sides) are stored in pairs.parquet next to the arrays; the drafts' own counts stay hidden.

The checkpoint only supplies the stored transforms (all models trained on one feature table share them; narrowing
to a feature subset is done at training time with FeatureTable.without_num). Embeddings of posts that are rows of
the training table are taken from it by rendered text, the rest are computed with the encoder.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

log = logging.getLogger("pairtable")
COLS = ["post_id", "author_id", "author_handle", "author_display_name", "created_at_ms", "snapshot_at_ms", "text", "kind", "media", "urls",
        "quoted_post_id", "quoted_text", "likes", "replies", "reposts", "quotes", "views", "author_followers", "author_following", "source"]
KEEP = ("post_id", "created_s", "split", "w_row", "audience", "num", "cat", "fmt", "counts", "followers", "y_like", "y_reach", "y_likes", "y_views",
        "rates", "emb", "has_emb", "sid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--posts", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--table", default=None, help="training table directory: its embeddings are reused by rendered text")
    ap.add_argument("--sid-codebook", default=None)
    ap.add_argument("--history", type=int, default=48)
    ap.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    from xviral.embed.text_small import SentenceEmbedder
    from xviral.serve import DraftScorer

    todo = pl.read_parquet(args.pairs).sort("author_id", "post_id", "post_id_b")
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        todo = todo.filter(pl.col("author_id").hash(seed=1) % n == i)
    log.info("pairs: %d", todo.height)
    d = pl.scan_parquet(args.posts).select(COLS).filter(pl.col("author_id").is_in(todo["author_id"].unique().to_list())).collect()
    by_author = {a: g.sort("created_at_ms") for (a,), g in d.group_by("author_id")}
    by_id = {r["post_id"]: r for r in d.filter(pl.col("post_id").is_in(pl.concat([todo["post_id"], todo["post_id_b"]]).unique().implode())).iter_rows(named=True)}
    del d

    known: dict[str, np.ndarray] = {}
    if args.table:
        ids = np.load(Path(args.table) / "post_id.npy")
        emb = np.load(Path(args.table) / "emb.npy", mmap_mode="r")
        has = np.load(Path(args.table) / "has_emb.npy")
        want = set()
        for g in by_author.values():
            want.update(g["post_id"].to_list())
        tx = pl.read_parquet(Path(args.table) / "texts.parquet")
        assert (tx["post_id"].to_numpy() == ids).all()
        sel = np.flatnonzero(np.isin(ids, np.fromiter(want, np.int64, len(want))) & has)
        texts = tx["text"].gather(sel).to_list()
        for tt, r in zip(texts, sel):
            known[tt] = np.asarray(emb[r], np.float32)
        del tx
        log.info("embeddings reused from the training table: %d", len(known))
    enc = SentenceEmbedder(args.encoder, device="cpu")
    scorer = DraftScorer(args.checkpoint, lambda t: enc.encode(t, batch_size=64), device="cpu", sid_codebook=args.sid_codebook)
    scorer._emb_cache, scorer.emb_cache_limit = known, 10 ** 9
    need = []
    for a_, g in by_author.items():
        ts_ = g["created_at_ms"].to_numpy()
        mark = np.zeros(g.height, bool)
        for r in todo.filter(pl.col("author_id") == a_).iter_rows(named=True):
            t_a = min(by_id[r["post_id"]]["created_at_ms"], by_id[r["post_id_b"]]["created_at_ms"])
            hi = int(np.searchsorted(ts_, t_a, side="left"))
            mark[max(0, hi - args.history):hi] = True
            mark[np.isin(g["post_id"].to_numpy(), [r["post_id"], r["post_id_b"]])] = True
        if mark.any():
            need.append(g.filter(pl.Series(mark)))
    log.info("encoded ahead: %d texts", scorer.preencode(pl.concat(need), log=log))
    del need
    parts: dict[str, list] = {k: [] for k in KEEP}
    a_start, a_end, recs = [], [], []
    off, t0 = 0, time.time()
    for i, r in enumerate(todo.iter_rows(named=True)):
        a, b = by_id[r["post_id"]], by_id[r["post_id_b"]]
        if a["created_at_ms"] > b["created_at_ms"]:
            a, b = b, a
        hist = by_author[r["author_id"]].filter(pl.col("created_at_ms") < a["created_at_ms"]).tail(args.history)
        account = {"author_id": r["author_id"], "handle": a["author_handle"], "display_name": a["author_display_name"],
                   "followers": a["author_followers"], "following": a["author_following"]}
        drafts = [{"text": x["text"], "kind": x["kind"], "media": x["media"], "urls": x["urls"], "quoted_text": x["quoted_text"]} for x in (a, b)]
        try:
            table, rows = scorer.featurize(hist, drafts, account, a["created_at_ms"])
        except Exception as e:  # noqa: BLE001
            log.warning("pair %d skipped: %s", i, e)
            continue
        n = table.n
        # Drafts are the last two rows, ordered by their (hashed) ids; record which is which.
        da = int(scorer._draft_row(drafts[0], 0, account, a["created_at_ms"])["post_id"][0])
        tail_ids = table.t["post_id"][n - 2:].tolist()
        if da not in tail_ids or rows.tail(2)["source"].to_list() != ["draft", "draft"]:
            log.warning("pair %d skipped: drafts not last", i)
            continue
        for k in KEEP:
            if k in table.t:
                parts[k].append(table.t[k].numpy())
        a_start += [off] * n
        a_end += [off + n] * n
        recs.append({"post_id": a["post_id"], "post_id_b": b["post_id"], "author_id": r["author_id"], "likes": a["likes"], "likes_b": b["likes"],
                     "row_a": off + n - 2 + tail_ids.index(da), "row_b": off + n - 2 + (1 - tail_ids.index(da)), "anchor": off + n - 2, "history": hist.height})
        off += n
        if (i + 1) % 500 == 0:
            log.info("%d/%d pairs, %.2fs per pair", i + 1, todo.height, (time.time() - t0) / (i + 1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in parts.items():
        if v:
            np.save(out / f"{k}.npy", np.concatenate(v))
    np.save(out / "a_start.npy", np.asarray(a_start, np.int64))
    np.save(out / "a_end.npy", np.asarray(a_end, np.int64))
    np.save(out / "author_idx.npy", np.repeat(np.arange(len(recs), dtype=np.int32), [x["anchor"] + 2 - (recs[j - 1]["anchor"] + 2 if j else 0) for j, x in enumerate(recs)]))
    pl.DataFrame(recs).write_parquet(out / "pairs.parquet")
    (out / "meta.json").write_text(json.dumps({**scorer.meta, "n": off, "pairs": len(recs)}))
    log.info("pair table: %d pairs, %d rows", len(recs), off)


if __name__ == "__main__":
    main()
