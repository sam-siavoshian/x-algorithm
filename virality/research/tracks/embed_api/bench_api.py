#!/usr/bin/env python3
"""Serving-side counterpart of api_table.py: rewrite a benchmark feature cache so that every post's embedding (the
account's history and the drafts) follows a table's recipe, with semantic IDs from that table's codebook.

    python research/tracks/embed_api/bench_api.py --table data/api/table_te3l1024 --store te3l=data/api/store/te3large_3072 \
        --cache data/tracks/pairs/bench/cache_0.pkl --out data/api/bench/te3l1024/cache_0.pkl [--count-only]

A cache maps a key to (posts frame, bge-small embeddings, semantic IDs), as DraftScorer.feature_cache keeps it
(tac_serve_eval.py, rep_score_launch.py). The bge-small part of a recipe is taken from the cache; OpenAI parts are
fetched through the on-disk store (openai_embed.py), so a text already paid for (the training table, an earlier
cache) costs nothing. --count-only prints how many texts would be sent and stops.
Scoring a checkpoint trained on the table against the rewritten cache is the benchmark with the same embedding at
training and serving time, on exactly the same pairs and features as every other track.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import recipe as R  # noqa: E402
from api_table import pca_apply  # noqa: E402
from openai_embed import MAX_CHARS, NATIVE_DIM, ApiEmbedder, key_of  # noqa: E402

MODEL = {"te3l": "text-embedding-3-large", "te3s": "text-embedding-3-small"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--store", action="append", default=[], help="src=cache directory of openai_embed.ApiEmbedder (native size)")
    ap.add_argument("--cache", action="append", required=True)
    ap.add_argument("--out", action="append", default=[])
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--max-dollars", type=float, default=5.0)
    args = ap.parse_args()
    from xviral.embed import sid as sidmod
    from xviral.model import features
    rec = R.load(os.path.join(args.table, "recipe.json"))
    stores = dict(s.split("=", 1) for s in args.store)
    encs = {s: ApiEmbedder(MODEL[s], stores[s], max_dollars=args.max_dollars) for s in R.sources(rec) if s != "bge"}
    pca = np.load(os.path.join(args.table, "pca.npz")) if os.path.exists(os.path.join(args.table, "pca.npz")) else None
    cb = os.path.join(args.table, "sid_codebook.npz")
    for ci, cpath in enumerate(args.cache):
        cache = pickle.load(open(cpath, "rb"))
        rendered = {k: features.render_texts(v[0]) for k, v in cache.items()}
        texts = sorted({t for r in rendered.values() for t in r})
        if args.count_only:
            for s, e in encs.items():
                new = [t for t in texts if key_of((t or " ")[:MAX_CHARS]) not in e._pos]
                print(json.dumps({"cache": cpath, "src": s, "distinct_texts": len(texts), "not_in_store": len(new), "chars_to_send": sum(len(t) for t in new)}))
            continue
        raw_by_text = {s: dict(zip(texts, e.encode(texts))) for s, e in encs.items()}
        out = {}
        for k, (posts, old, _) in cache.items():
            raw = {"bge": np.asarray(old, dtype=np.float32)} | {s: np.stack([raw_by_text[s][t] for t in rendered[k]]) for s in encs}
            emb = R.apply(rec, {s: raw[s] for s in R.sources(rec)})
            if pca is not None:
                emb = pca_apply(emb, pca["mu"], pca["comp"])
            emb = emb.astype(np.float16).astype(np.float32)                   # as the table stores it
            out[k] = (posts, emb, sidmod.phoenix_codes(sidmod.assign(emb, cb)))
        os.makedirs(os.path.dirname(args.out[ci]), exist_ok=True)
        pickle.dump(out, open(args.out[ci], "wb"))
        print("written", args.out[ci], len(out), {s: e.ledger for s, e in encs.items()}, flush=True)


if __name__ == "__main__":
    main()
