#!/usr/bin/env python3
"""OpenAI text embeddings with an on-disk cache, batching, retries and a running token and cost ledger.

    from openai_embed import ApiEmbedder
    enc = ApiEmbedder("text-embedding-3-large", "data/api/store/te3l_1024", dimensions=1024)
    E = enc.encode(texts)                      # float32 [n, dim], unit length; each distinct text is paid for once

The cache is a directory of shards: shard-00000.keys.npy (uint64 hash of the text) and shard-00000.npy (float16 [n, dim]),
plus ledger.json (requests, tokens, dollars). It is safe to stop and restart; upload the directory and nobody pays twice.
The key for the proxy is injected by the network layer (Authorization: Bearer proxy-injected).

`dimensions` below the model's native size is the API's own shortening (Matryoshka): the leading coordinates,
renormalised. `shorten()` does the same to stored full-size vectors, so one paid call serves every size.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

PRICE_PER_MTOK = {"text-embedding-3-large": 0.13, "text-embedding-3-small": 0.02}
NATIVE_DIM = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536}
URL = "https://api.openai.com/v1/embeddings"
MAX_CHARS = 8_000        # the model takes 8,191 tokens; rendered posts are far shorter, this guards the few long-form ones


def key_of(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).digest(), "little")


def shorten(E: np.ndarray, dim: int) -> np.ndarray:
    E = np.asarray(E[:, :dim], dtype=np.float32)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)


class ApiEmbedder:
    def __init__(self, model: str, cache_dir: str | Path, dimensions: int | None = None, batch: int = 256, workers: int = 6,
                 max_dollars: float | None = None):
        self.model, self.dim = model, dimensions or NATIVE_DIM[model]
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.batch, self.workers, self.max_dollars = batch, workers, max_dollars
        self.lock = threading.Lock()
        lp = self.dir / "ledger.json"
        self.ledger = json.loads(lp.read_text()) if lp.exists() else {"model": model, "dimensions": self.dim, "requests": 0, "texts": 0, "tokens": 0, "dollars": 0.0}
        assert self.ledger["model"] == model and self.ledger["dimensions"] == self.dim, "cache directory belongs to another model or size"
        self._keys: np.ndarray | None = None
        self._vecs: list[np.ndarray] = []
        self._pos: dict[int, tuple[int, int]] = {}
        for s, p in enumerate(sorted(self.dir.glob("shard-*.keys.npy"))):
            k = np.load(p)
            self._vecs.append(np.load(str(p).replace(".keys.npy", ".npy"), mmap_mode="r"))
            for r, x in enumerate(k.tolist()):
                self._pos[x] = (s, r)

    # -- API -----------------------------------------------------------------------------------------------------
    def _call(self, texts: list[str]) -> tuple[np.ndarray, int]:
        body = {"model": self.model, "input": texts, "encoding_format": "base64"}
        if self.dim != NATIVE_DIM[self.model]:
            body["dimensions"] = self.dim
        for attempt in range(8):
            try:
                r = requests.post(URL, json=body, headers={"Authorization": "Bearer proxy-injected"}, timeout=120)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                j = r.json()
                import base64
                E = np.stack([np.frombuffer(base64.b64decode(d["embedding"]), dtype="<f4") for d in sorted(j["data"], key=lambda d: d["index"])])
                return E, int(j["usage"]["total_tokens"])
            if r.status_code == 400 and len(texts) == 1 and len(texts[0]) > 200:      # over the model's 8,192 tokens: shorten and retry
                body["input"] = texts = [texts[0][:len(texts[0]) // 2]]
                continue
            if r.status_code == 400 and len(texts) > 1:        # one over-long text in the batch: do them singly
                parts = [self._call([t]) for t in texts]
                return np.concatenate([p[0] for p in parts]), sum(p[1] for p in parts)
            if r.status_code in (403, 407):
                raise RuntimeError(f"refused by the proxy: {r.status_code} {r.text[:200]}")
            time.sleep(min(2 ** attempt, 60))                  # 429 and 5xx
        raise RuntimeError(f"embedding call failed after retries: {r.status_code} {r.text[:200]}")

    def _flush(self, keys: list[int], vecs: list[np.ndarray]) -> None:
        if not keys:
            return
        s = len(self._vecs)
        V = np.concatenate(vecs).astype(np.float16)
        np.save(self.dir / f"shard-{s:05d}.npy", V)
        np.save(self.dir / f"shard-{s:05d}.keys.npy", np.array(keys, dtype=np.uint64))
        self._vecs.append(V)
        for r, x in enumerate(keys):
            self._pos[x] = (s, r)
        (self.dir / "ledger.json").write_text(json.dumps(self.ledger, indent=1))

    def encode(self, texts: list[str], shard_texts: int = 40_000, log=print) -> np.ndarray:
        texts = [(t or " ")[:MAX_CHARS] for t in texts]
        keys = [key_of(t) for t in texts]
        todo: dict[int, str] = {}
        for k, t in zip(keys, texts):
            if k not in self._pos and k not in todo:
                todo[k] = t
        items = sorted(todo.items(), key=lambda kv: len(kv[1]))
        t0, done = time.time(), 0
        for i in range(0, len(items), shard_texts):
            part = items[i:i + shard_texts]
            batches = [part[j:j + self.batch] for j in range(0, len(part), self.batch)]

            def one(b):
                E, tok = self._call([t for _, t in b])
                with self.lock:
                    self.ledger["requests"] += 1
                    self.ledger["texts"] += len(b)
                    self.ledger["tokens"] += tok
                    self.ledger["dollars"] = self.ledger["tokens"] * PRICE_PER_MTOK[self.model] / 1e6
                return E

            if self.max_dollars is not None and self.ledger["dollars"] >= self.max_dollars:
                raise RuntimeError(f"spending cap reached: {self.ledger}")
            with ThreadPoolExecutor(self.workers) as ex:
                out = list(ex.map(one, batches))
            self._flush([k for b in batches for k, _ in b], out)
            done += len(part)
            log(f"{self.model}/{self.dim}: {done}/{len(items)} new texts, {done / (time.time() - t0):.0f}/s, "
                f"{self.ledger['tokens']:,} tokens, ${self.ledger['dollars']:.3f} so far", flush=True)
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, k in enumerate(keys):
            s, r = self._pos[k]
            out[i] = self._vecs[s][r]
        return out


if __name__ == "__main__":
    import argparse

    import polars as pl
    ap = argparse.ArgumentParser(description="Embed a text column of a parquet file; write <out>.npy aligned with it.")
    ap.add_argument("--texts", required=True)
    ap.add_argument("--column", default="rendered")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dimensions", type=int, default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-dollars", type=float, default=None)
    ap.add_argument("--estimate", action="store_true", help="embed a 2,000-text sample, print tokens per text and the projected cost of the rest, stop")
    a = ap.parse_args()
    T = pl.read_parquet(a.texts)[a.column].to_list()
    enc = ApiEmbedder(a.model, a.cache, a.dimensions, max_dollars=a.max_dollars)
    if a.estimate:
        rng = np.random.default_rng(0)
        before = dict(enc.ledger)
        sample = [T[i] for i in rng.choice(len(T), min(2000, len(T)), replace=False)]
        enc.encode(sample)
        n, tok = enc.ledger["texts"] - before["texts"], enc.ledger["tokens"] - before["tokens"]
        if n:
            print(json.dumps({"sample_texts": n, "tokens_per_text": tok / n, "texts": len(T), "projected_tokens": tok / n * len(T),
                              "projected_dollars": tok / n * len(T) * PRICE_PER_MTOK[a.model] / 1e6}))
    else:
        E = enc.encode(T)
        if a.out:
            np.save(a.out, E.astype(np.float16) if len(E) > 100_000 else E)
        print(json.dumps(enc.ledger))
