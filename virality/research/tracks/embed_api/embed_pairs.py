#!/usr/bin/env python3
"""(Copied from the encoder track, unchanged apart from paths.) Step 1a of the encoder track: embed every post that appears in a same-topic pair with one candidate encoder.

    python research/tracks/embed_api/embed_pairs.py --posts data/work/filtered/posts.parquet --after 2026-04-23 \
        --encoder BAAI/bge-base-en-v1.5 --out data/probe

Writes <out>/pairs.parquet once (all pre-cutoff pairs and all test-period pairs, as pair_model.py finds them),
<out>/texts.parquet (rendered text as the model's encoder sees it, and the raw text), and
<out>/emb_<name>_<rendered|raw>.npy aligned with texts.parquet.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "evidence"))
import tac_serve_eval as T  # noqa: E402

PREFIX = {"intfloat/e5-base-v2": "query: ", "intfloat/e5-small-v2": "query: ", "intfloat/e5-large-v2": "query: "}
TWITTER = ("cardiffnlp/", "vinai/bertweet")


def tw_norm(t: str) -> str:
    return re.sub(r"https?://\S+", "http", re.sub(r"@\w+", "@user", t))


def encode(model_id: str, texts: list[str], batch: int = 64, max_tokens: int = 512) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_id, device="cpu").float()      # some checkpoints ship fp16 weights
    lim = getattr(m[0].auto_model.config, "max_position_embeddings", 512)
    m.max_seq_length = min(max_tokens, lim - 2 if lim < 1000 else max_tokens)
    if model_id.startswith(TWITTER):
        texts = [tw_norm(t) for t in texts]
    texts = [PREFIX.get(model_id, "") + t for t in texts]
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):   # AMX bf16: 2.3x faster, cosine to fp32 above 0.9999
        return m.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--encoder", action="append", default=[])
    ap.add_argument("--views", default="rendered,raw")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from xviral.model import features
    from xviral.util.core import date_to_ms
    os.makedirs(args.out, exist_ok=True)
    pp, tp = os.path.join(args.out, "pairs.parquet"), os.path.join(args.out, "texts.parquet")
    if not os.path.exists(tp):
        d = pl.scan_parquet(args.posts).select(T.COLS).collect()
        pairs = T.find_pairs(d, 48.0).sort("post_id", "post_id_b")
        cut = date_to_ms(args.after)
        pairs = pairs.filter((pl.col("created_at_ms") >= cut) | (pl.col("created_at_ms_b") < cut))
        ids = pl.concat([pairs["post_id"], pairs["post_id_b"]]).unique()
        P = d.filter(pl.col("post_id").is_in(ids.implode())).sort("post_id")
        P.write_parquet(os.path.join(args.out, "posts.parquet"))
        pairs.write_parquet(pp)
        pl.DataFrame({"post_id": P["post_id"], "rendered": features.render_texts(P), "raw": P["text"].fill_null("")}).write_parquet(tp)
        print("pairs", pairs.height, "posts", P.height, flush=True)
    X = pl.read_parquet(tp)
    for enc in args.encoder:
        for view in args.views.split(","):
            f = os.path.join(args.out, f"emb_{enc.replace('/', '__')}_{view}.npy")
            if os.path.exists(f):
                continue
            t0 = time.time()
            E = encode(enc, X[view].to_list())
            np.save(f, E)
            print(enc, view, E.shape, f"{time.time() - t0:.0f}s", f"{len(E) / (time.time() - t0):.0f} posts/s", flush=True)


if __name__ == "__main__":
    main()
