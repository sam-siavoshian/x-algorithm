"""Near-duplicate tagging (DATA_PIPELINE.md section 3, steps 3-5).

Duplicates are tagged, not dropped: they are real posts with real outcomes,
and late copies of a viral post behave differently. X's DPP reranker gives a
near-duplicate of a higher-scoring post a score of zero in the same feed
(vm-ranker/scoring/dpp.rs), so cluster rank is a useful feature.

Text: MinHash over word 5-gram shingles, LSH banding, Jaccard >= threshold,
within a time window. Implemented in NumPy so it scales to millions of rows
without a per-pair Python loop.
"""

from __future__ import annotations

import logging
import re
import zlib

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_URL_RE = re.compile(r"https?://\S+")
_MERSENNE = (1 << 61) - 1


def shingles(text: str | None, k: int = 5) -> np.ndarray:
    toks = _TOKEN_RE.findall(_URL_RE.sub(" ", (text or "").lower()))
    if not toks:
        return np.zeros(0, dtype=np.uint64)
    if len(toks) < k:
        grams = [" ".join(toks)]
    else:
        grams = [" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)]
    return np.fromiter((zlib.crc32(g.encode()) for g in set(grams)), dtype=np.uint64)


def minhash_signatures(texts: list[str | None], num_perm: int = 64, seed: int = 1) -> np.ndarray:
    rng = np.random.RandomState(seed)
    a = rng.randint(1, _MERSENNE, size=num_perm, dtype=np.uint64)
    b = rng.randint(0, _MERSENNE, size=num_perm, dtype=np.uint64)
    sig = np.full((len(texts), num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, t in enumerate(texts):
        sh = shingles(t)
        if sh.size == 0:
            continue
        # (a*x + b) mod p, computed in uint64 with wraparound; adequate for LSH bucketing.
        h = (np.outer(sh, a) + b) % np.uint64(_MERSENNE)
        sig[i] = h.min(axis=0)
    return sig


def near_duplicate_clusters(df: pl.DataFrame, threshold: float = 0.9, window_days: float = 7.0,
                            num_perm: int = 64, bands: int = 16, min_chars: int = 40) -> pl.DataFrame:
    """Return post_id, near_dup_cluster_id, near_dup_rank, near_dup_cluster_size.

    Posts shorter than `min_chars` are skipped: short posts collide on phrasing, not copying.
    Candidate pairs come from LSH buckets, are verified by signature agreement (an unbiased
    Jaccard estimate), restricted to `window_days`, and merged with union-find.
    """
    d = df.select("post_id", "text", "created_at_ms").filter(pl.col("text").fill_null("").str.len_chars() >= min_chars)
    n = d.height
    if n == 0:
        return pl.DataFrame(schema={"post_id": pl.Int64, "near_dup_cluster_id": pl.Int64, "near_dup_rank": pl.Int32, "near_dup_cluster_size": pl.Int32})
    sig = minhash_signatures(d["text"].to_list(), num_perm=num_perm)
    rows = num_perm // bands
    ids = d["post_id"].to_numpy()
    ts = d["created_at_ms"].to_numpy()
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    window_ms = window_days * 86_400_000
    checked: set[tuple[int, int]] = set()
    for bnd in range(bands):
        keys = sig[:, bnd * rows:(bnd + 1) * rows]
        bucket = np.frombuffer(np.ascontiguousarray(keys).tobytes(), dtype=f"V{8 * rows}")
        order = np.argsort(bucket, kind="stable")
        sb = bucket[order]
        starts = np.flatnonzero(np.r_[True, sb[1:] != sb[:-1]])
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends):
            if e - s < 2 or e - s > 500:  # huge buckets are empty-text or boilerplate collisions
                continue
            members = order[s:e]
            for ii in range(len(members)):
                for jj in range(ii + 1, len(members)):
                    i, j = int(members[ii]), int(members[jj])
                    key = (i, j) if i < j else (j, i)
                    if key in checked:
                        continue
                    checked.add(key)
                    if abs(ts[i] - ts[j]) > window_ms:
                        continue
                    if (sig[i] == sig[j]).mean() >= threshold:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    out = pl.DataFrame({"post_id": ids, "_root": roots, "created_at_ms": ts})
    out = out.with_columns(
        pl.col("_root").count().over("_root").cast(pl.Int32).alias("near_dup_cluster_size"),
        pl.col("created_at_ms").rank("ordinal").over("_root").cast(pl.Int32).alias("near_dup_rank"),
        pl.col("post_id").min().over("_root").alias("near_dup_cluster_id"),
    )
    log.info("near-dup: %d posts, %d in clusters of size > 1", n, out.filter(pl.col("near_dup_cluster_size") > 1).height)
    return out.select("post_id", "near_dup_cluster_id", "near_dup_rank", "near_dup_cluster_size")


def author_copypasta(df: pl.DataFrame, window_days: float = 30.0) -> pl.DataFrame:
    """Per post: is it an exact repeat of the same author's earlier text within the window.

    Feeds the copypasta suppression-risk feature (botmaker-rules BBQDuplicateTextProd.bot clusters
    duplicate text) and the hygiene filter.
    """
    h = (
        pl.col("text").fill_null("").str.to_lowercase().str.replace_all(r"https?://\S+", "")
        .str.replace_all(r"\s+", " ").str.strip_chars()
    )
    d = df.select("post_id", "author_id", "created_at_ms", h.alias("_t")).filter(pl.col("_t").str.len_chars() >= 20)
    d = d.with_columns(pl.col("_t").hash(seed=3).alias("_h")).sort("author_id", "_h", "created_at_ms")
    d = d.with_columns(
        (pl.col("created_at_ms") - pl.col("created_at_ms").shift(1).over(["author_id", "_h"])).alias("_gap")
    ).with_columns(
        (pl.col("_gap").is_not_null() & (pl.col("_gap") <= int(window_days * 86_400_000))).alias("is_self_repeat")
    )
    return d.select("post_id", "author_id", "is_self_repeat")


def image_phash(images: list[bytes | None]) -> list[str | None]:
    import io

    import imagehash
    from PIL import Image

    out = []
    for b in images:
        if not b:
            out.append(None)
            continue
        try:
            out.append(str(imagehash.phash(Image.open(io.BytesIO(b)).convert("RGB"))))
        except Exception:  # noqa: BLE001 - corrupt image bytes
            out.append(None)
    return out
