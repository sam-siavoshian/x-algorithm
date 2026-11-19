#!/usr/bin/env python3
"""Selective track: embeddings for every post a unit list needs (history and versions), reusing stored ones by post id
(feature tables) and encoding the rest in one pass sorted by length.   python scripts/selective/sel_seed.py <set> [table_dir ...]
Writes data/sel/emb_seed_<set>.pkl: {rendered text: vector}."""
import pickle, sys, time
import numpy as np, polars as pl
from xviral.model import features as F
from xviral.schema import conform
nm, tables = sys.argv[1], sys.argv[2:]
COLS = ["post_id", "author_id", "author_handle", "author_display_name", "created_at_ms", "snapshot_at_ms", "text", "kind", "media", "urls",
        "quoted_post_id", "quoted_text", "likes", "replies", "reposts", "quotes", "views", "author_followers", "author_following", "source"]
U = pl.read_parquet(f"data/sel/units_{nm}.parquet")
d = pl.scan_parquet("data/work/filtered/posts.parquet").select(COLS).filter(pl.col("author_id").is_in(U["author_id"].unique().to_list())).collect()
need = []
for (a_,), g in U.group_by("author_id"):
    h = d.filter(pl.col("author_id") == a_).sort("created_at_ms"); ts_ = h["created_at_ms"].to_numpy(); mark = np.zeros(h.height, bool)
    for t in g["when_ms"].to_list():
        hi = int(np.searchsorted(ts_, t, side="left")); mark[max(0, hi - 56):hi] = True
    mark |= h["post_id"].is_in(g["post_ids"].explode().implode()).to_numpy()
    need.append(h.filter(pl.Series(mark)))
need = pl.concat(need).unique("post_id")
texts = F.render_texts(conform(need)); ids = need["post_id"].to_numpy()
print("posts needed", len(ids), "distinct texts", len(set(texts)), flush=True)
out = {}
for t in tables:
    tid = np.load(f"{t}/post_id.npy"); he = np.load(f"{t}/has_emb.npy"); E = np.load(f"{t}/emb.npy", mmap_mode="r")
    pos = {int(p): i for i, p in enumerate(tid) if he[i] and p > 0}
    n0 = len(out)
    for p, tx in zip(ids.tolist(), texts):
        if tx not in out and p in pos: out[tx] = np.asarray(E[pos[p]], np.float32)
    print(t, "gave", len(out) - n0, flush=True)
rest = sorted({t for t in texts if t not in out}, key=len)
print("to encode", len(rest), flush=True)
from xviral.embed.text_small import SentenceEmbedder
enc = SentenceEmbedder("BAAI/bge-small-en-v1.5", device="cpu"); t0 = time.time()
if out:   # check that stored vectors match this encoder on a few texts
    ks = list(out)[:16]; v = np.asarray(enc.encode(ks)); print("stored vs encoded cosine (min):", float(min((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)) for a, b in zip(v, [out[k] for k in ks]))), flush=True)
for k in range(0, len(rest), 1024):
    part = rest[k:k + 1024]
    for t, v in zip(part, np.asarray(enc.encode(part))): out[t] = np.asarray(v, np.float32)
    if (k // 1024) % 10 == 0: print(k + len(part), len(rest), round(time.time() - t0), flush=True)
pickle.dump(out, open(f"data/sel/emb_seed_{nm}.pkl", "wb")); print("wrote", len(out))
