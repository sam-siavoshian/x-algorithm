#!/usr/bin/env python3
"""Embed the rows of the broad table (tracks/general/table) that the OpenAI store does not hold yet, and write the
table's post embedding (text-embedding-3-large, first 1,024 dimensions, renormalised) aligned with its rows.

    python scripts/combine/embed_broad.py --estimate            # 2,000-text sample of the missing texts, projected cost, stop
    python scripts/combine/embed_broad.py --max-dollars 14      # embed what is missing, write data/combine/te3l1024_broad.npy

The store (research/tracks/embed_api/openai_embed.py) is keyed by a hash of the text, so a text is paid for once; new
shards are appended to the same directory and uploaded with it.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, polars as pl
sys.path.insert(0, "research/tracks/embed_api")
from openai_embed import MAX_CHARS, PRICE_PER_MTOK, ApiEmbedder, key_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--texts", default="data/tracks/general/table/texts.parquet")
ap.add_argument("--store", default="data/tracks/embed_api/store/te3large_3072")
ap.add_argument("--out", default="data/combine/te3l1024_broad.npy")
ap.add_argument("--dim", type=int, default=1024)
ap.add_argument("--estimate", action="store_true")
ap.add_argument("--max-dollars", type=float, default=14.0)
a = ap.parse_args()
T = pl.read_parquet(a.texts)["text"].to_list()
enc = ApiEmbedder("text-embedding-3-large", a.store, workers=8)
start = dict(enc.ledger)
norm = [(t or " ")[:MAX_CHARS] for t in T]
keys = np.fromiter((key_of(t) for t in norm), dtype=np.uint64, count=len(norm))
missing = {}
for k, t in zip(keys.tolist(), norm):
    if k not in enc._pos and k not in missing:
        missing[k] = t
print(json.dumps({"rows": len(T), "distinct": int(len(np.unique(keys))), "in_store": len(T) - sum(k not in enc._pos for k in keys.tolist()),
                  "distinct_missing": len(missing), "chars_missing": sum(len(t) for t in missing.values())}), flush=True)
todo = list(missing.values())
if a.estimate:
    rng = np.random.default_rng(0)
    sample = [todo[i] for i in rng.choice(len(todo), min(2000, len(todo)), replace=False)]
    enc.encode(sample)
    n, tok = enc.ledger["texts"] - start["texts"], enc.ledger["tokens"] - start["tokens"]
    chars = sum(len(t) for t in sample)
    proj = tok / chars * sum(len(t) for t in todo)       # by characters: the sample's token rate applied to every missing character
    print(json.dumps({"sample_texts": n, "sample_tokens": tok, "tokens_per_text": tok / n, "tokens_per_char": tok / chars,
                      "projected_tokens": proj, "projected_dollars": proj * PRICE_PER_MTOK["text-embedding-3-large"] / 1e6}))
    sys.exit(0)
enc.max_dollars = start["dollars"] + a.max_dollars
t0 = time.time()
for i in range(0, len(todo), 40_000):                    # in parts: encode() returns a native-size array, 0.5 GB per part
    enc.encode(todo[i:i + 40_000])
print("embedded", len(todo), "texts in", round(time.time() - t0), "s; spent here $%.3f" % (enc.ledger["dollars"] - start["dollars"]), json.dumps(enc.ledger), flush=True)
out = np.lib.format.open_memmap(a.out, mode="w+", dtype=np.float16, shape=(len(T), a.dim))
for i in range(0, len(T), 20_000):
    blk = np.stack([np.asarray(enc._vecs[s][r][:a.dim], dtype=np.float32) for s, r in (enc._pos[k] for k in keys[i:i + 20_000].tolist())])
    out[i:i + len(blk)] = (blk / np.maximum(np.linalg.norm(blk, axis=1, keepdims=True), 1e-9)).astype(np.float16)
out.flush()
print("written", a.out, out.shape)
