#!/usr/bin/env python3
"""Mine controlled pairs: same account, near-duplicate topic, within --max-gap-days.

    python scripts/evidence/mine_pairs.py --posts <work>/filtered/posts.parquet --out <dir> [--stage candidates|embed]

Stage 1 (candidates): a word-overlap prefilter. Posts are turned into TF-IDF vectors of word 1- and 2-grams
(links removed); within each account, every pair of posts at most --max-gap-days apart whose TF-IDF cosine is
at least --lex-min is kept. This makes the 4.48M-post corpus affordable on a CPU: only posts in a candidate
pair are embedded.
Stage 2 (embed): candidate posts are embedded with the bge-small encoder (plain text, links removed, no author
line, so the similarity is about the message and not the account) and each pair gets its embedding cosine.
The same-link and same-quoted-post pairs of tac_serve_eval.find_pairs are added with their cosine.
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
import polars as pl

log = logging.getLogger("mine")
NORM = (pl.col("text").fill_null("").str.to_lowercase().str.replace_all(r"https?://\S+", "").str.replace_all(r"\s+", " ").str.strip_chars())


def candidates(d: pl.DataFrame, max_gap_days: float, lex_min: float, min_words: int, block: int = 4000) -> pl.DataFrame:
    from scipy import sparse
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

    d = d.with_columns(NORM.alias("nt")).with_columns(pl.col("nt").str.count_matches(r"\S+").alias("nw"))
    d = d.filter(pl.col("nw") >= min_words).sort("author_id", "created_at_ms", "post_id")
    log.info("posts with at least %d words: %d", min_words, d.height)
    hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm=None, token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
    X = TfidfTransformer(sublinear_tf=True).fit_transform(hv.transform(d["nt"].to_list())).tocsr().astype(np.float32)
    log.info("tf-idf %s nnz %d", X.shape, X.nnz)
    author, t = d["author_id"].to_numpy(), d["created_at_ms"].to_numpy()
    n = len(t)
    gap = int(max_gap_days * 86_400_000)
    starts = np.flatnonzero(np.r_[True, author[1:] != author[:-1]])
    ends = np.r_[starts[1:], n]
    out_i, out_j, out_s = [], [], []
    t0, done = time.time(), 0
    for a0, a1 in zip(starts, ends):
        if a1 - a0 < 2:
            continue
        for s in range(a0, a1, block):
            e = min(s + block, a1)
            hi = min(int(np.searchsorted(t[a0:a1], t[e - 1] + gap, side="right")) + a0, a1)
            S = (X[s:e] @ X[s:hi].T).tocoo()
            m = (S.data >= lex_min) & (S.col > S.row)
            i, j = S.row[m] + s, S.col[m] + s
            ok = (t[j] - t[i]) <= gap
            out_i.append(i[ok]); out_j.append(j[ok]); out_s.append(S.data[m][ok])
        done += 1
        if done % 5000 == 0:
            log.info("accounts %d, candidate pairs %d, %.0fs", done, sum(map(len, out_i)), time.time() - t0)
    i, j, s = np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_s)
    ids = d["post_id"].to_numpy()
    return pl.DataFrame({"post_id": ids[i], "post_id_b": ids[j], "author_id": author[i], "lex": s})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="candidates", choices=["candidates", "embed"])
    ap.add_argument("--max-gap-days", type=float, default=14.0)
    ap.add_argument("--lex-min", type=float, default=0.25)
    ap.add_argument("--min-words", type=int, default=6)
    ap.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "candidates":
        d = pl.scan_parquet(args.posts).select("post_id", "author_id", "created_at_ms", "text").collect()
        c = candidates(d, args.max_gap_days, args.lex_min, args.min_words)
        log.info("candidate pairs: %d", c.height)
        c.write_parquet(out / "candidates.parquet")
        return
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from tac_serve_eval import find_pairs
    import torch
    from sentence_transformers import SentenceTransformer

    c = pl.read_parquet(out / "candidates.parquet")
    d = pl.scan_parquet(args.posts).select("post_id", "author_id", "created_at_ms", "text", "urls", "quoted_post_id", "likes", "views").collect()
    tac = find_pairs(d, args.max_gap_days * 24).select("post_id", "post_id_b", "author_id", pl.col("key").str.split(":").list.first().alias("tac"))
    tac = tac.unique(["post_id", "post_id_b"])
    c = c.join(tac, on=["post_id", "post_id_b", "author_id"], how="full", coalesce=True)
    log.info("pairs to embed: %d (%d by link or quote)", c.height, c["tac"].is_not_null().sum())
    need = pl.concat([c["post_id"], c["post_id_b"]]).unique()
    p = d.filter(pl.col("post_id").is_in(need.implode())).select("post_id", NORM.alias("nt")).sort(pl.col("nt").str.len_chars())
    log.info("posts to embed: %d", p.height)
    torch.set_num_threads(4)
    model = SentenceTransformer(args.encoder, device="cpu")
    model.max_seq_length = args.max_tokens
    E = model.encode(p["nt"].to_list(), batch_size=256, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float16)
    np.save(out / "cand_emb.npy", E)
    p.select("post_id").write_parquet(out / "cand_emb_ids.parquet")
    pos = dict(zip(p["post_id"].to_list(), range(p.height)))
    ia = np.fromiter((pos[x] for x in c["post_id"].to_list()), np.int64, c.height)
    ib = np.fromiter((pos[x] for x in c["post_id_b"].to_list()), np.int64, c.height)
    cos = np.concatenate([(E[ia[k:k + 200_000]].astype(np.float32) * E[ib[k:k + 200_000]].astype(np.float32)).sum(1) for k in range(0, c.height, 200_000)])
    c = c.with_columns(pl.Series("cos", cos))
    c.write_parquet(out / "scored.parquet")
    log.info("done: %d pairs", c.height)


if __name__ == "__main__":
    main()
