"""The text encoder a set of checkpoints was trained with, read from the checkpoints themselves, and the OpenAI
embedding client the product uses when that encoder is an API model (research/tracks/combine/REPORT.md).

    spec = encoder_spec(scorer.meta)                 # {"name": "openai:text-embedding-3-large@1024", "provider": "openai", ...}
    enc = make_encoder(spec)                         # OpenAIEmbedder, or the local sentence-transformers encoder
    vectors = enc.encode(texts)                      # float32 [n, dim], unit length
    enc.take_usage()                                 # {"calls", "texts", "tokens", "usd", "seconds", "error"} since the last take

What leaves the machine with an API encoder: the rendered text of every post the model reads, which is the author
line (display name and handle) followed by the post text with link wrappers removed, for each version being scored and
for the account's most recent posts (at most the model's history length). No counts, no follower numbers, no media.
Each distinct text is sent once: the scorer caches vectors per account.

A model must be served with the encoder it was trained with. The name is stored in the checkpoint
(`table_meta["encoder"]`); a checkpoint without it is a bge-small model, which is what every checkpoint before the
embedding track was trained with.
"""

from __future__ import annotations

import base64
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BGE_SMALL = "BAAI/bge-small-en-v1.5"
# price_per_mtok: US dollars per million input tokens, OpenAI's price list on 2026-11-19.
ENCODERS = {
    BGE_SMALL: {"name": BGE_SMALL, "provider": "local", "model": BGE_SMALL, "dimensions": 384, "price_per_mtok": 0.0, "parts": [{"src": "bge"}]},
    "openai:text-embedding-3-large@1024": {"name": "openai:text-embedding-3-large@1024", "provider": "openai", "model": "text-embedding-3-large", "dimensions": 1024,
                                           "max_chars": 8000, "price_per_mtok": 0.13, "parts": [{"src": "te3l", "dim": 1024}]},
    "openai:text-embedding-3-small@1024": {"name": "openai:text-embedding-3-small@1024", "provider": "openai", "model": "text-embedding-3-small", "dimensions": 1024,
                                           "max_chars": 8000, "price_per_mtok": 0.02, "parts": [{"src": "te3s", "dim": 1024}]},
}
_BY_SRC = {"bge": BGE_SMALL, "te3l": "openai:text-embedding-3-large@{dim}", "te3s": "openai:text-embedding-3-small@{dim}"}
SENT = ("the rendered text (author name and handle line, then the post text) of each version and of the account's most recent posts the model reads; "
        "no counts, no follower numbers, no media files")


class EmbeddingError(RuntimeError):
    """The embedding provider could not be reached or refused the request. Nothing was scored with partial vectors."""


def encoder_spec(meta: dict | None) -> dict:
    """The encoder of a checkpoint, from its `table_meta`. Accepts the three forms that exist: no entry (bge-small), the
    embedding track's recipe ({"parts": [{"src": "te3l", "dim": 1024}]}) and this module's full entry."""
    enc = (meta or {}).get("encoder")
    if not enc:
        spec = dict(ENCODERS[BGE_SMALL])
    elif isinstance(enc, str):
        if enc not in ENCODERS:
            raise ValueError(f"the checkpoint names an encoder this build does not know: {enc!r}")
        spec = dict(ENCODERS[enc])
    elif enc.get("name"):
        spec = {**ENCODERS.get(enc["name"], {}), **enc}
    else:
        parts = enc.get("parts") or []
        if len(parts) != 1 or enc.get("adapter"):
            raise ValueError(f"the checkpoint was trained with a composite embedding the product cannot serve: {enc}")
        name = _BY_SRC.get(parts[0].get("src"), "").format(dim=parts[0].get("dim"))
        if name not in ENCODERS:
            raise ValueError(f"the checkpoint names an encoder this build does not know: {enc}")
        spec = dict(ENCODERS[name])
    dim = (meta or {}).get("emb_dim")
    if dim and spec.get("dimensions") and int(dim) != int(spec["dimensions"]):
        raise ValueError(f"the checkpoint's embedding size ({dim}) does not match its encoder {spec['name']} ({spec['dimensions']})")
    return spec


def common_encoder(metas: list[dict]) -> dict:
    """One encoder for an ensemble: every member must have been trained with the same one."""
    specs = [encoder_spec(m) for m in metas]
    names = sorted({s["name"] for s in specs})
    if len(names) != 1:
        raise ValueError(f"the checkpoints were trained with different encoders ({', '.join(names)}); an ensemble needs one")
    return specs[0]


def _unit16(E: np.ndarray) -> np.ndarray:
    """Unit length, then through half precision: the form the training table stores."""
    E = np.asarray(E, dtype=np.float32)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    return E.astype(np.float16).astype(np.float32)


class _Usage:
    def __init__(self):
        self._lock = threading.Lock()
        self.total = {"calls": 0, "texts": 0, "tokens": 0, "usd": 0.0, "seconds": 0.0, "errors": 0}
        self._since = {"calls": 0, "texts": 0, "tokens": 0, "usd": 0.0, "seconds": 0.0, "error": None}

    def add(self, calls=0, texts=0, tokens=0, usd=0.0, seconds=0.0, error=None):
        with self._lock:
            for k, v in (("calls", calls), ("texts", texts), ("tokens", tokens), ("usd", usd), ("seconds", seconds)):
                self.total[k] += v
                self._since[k] += v
            if error:
                self.total["errors"] += 1
                self._since["error"] = error

    def take(self) -> dict:
        with self._lock:
            out, self._since = self._since, {"calls": 0, "texts": 0, "tokens": 0, "usd": 0.0, "seconds": 0.0, "error": None}
        return out


class OpenAIEmbedder:
    """`encode(texts)` -> float32 [n, dimensions], unit length. Texts go out in batches of `batch`, batches in parallel.
    Any failure after the retries raises EmbeddingError: the caller gets no vectors rather than some.
    The key is OPENAI_API_KEY, or injected by a network proxy (then any placeholder is sent)."""

    provider = "OpenAI"
    local = False

    def __init__(self, model: str = "text-embedding-3-large", dimensions: int = 1024, batch: int = 256, timeout: float = 20.0, retries: int = 2,
                 concurrency: int = 4, max_chars: int = 8000, price_per_mtok: float | None = None, api_key: str | None = None,
                 base_url: str | None = None, session=None):
        self.model, self.dim, self.batch, self.timeout, self.retries = model, int(dimensions), max(1, int(batch)), timeout, retries
        self.concurrency, self.max_chars = max(1, int(concurrency)), max_chars
        self.price = price_per_mtok if price_per_mtok is not None else next((e["price_per_mtok"] for e in ENCODERS.values() if e["model"] == model), 0.0)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "proxy-injected"
        self.base_url = (base_url or os.environ.get("XVIRAL_EMBED_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self._session = session
        self.usage = _Usage()

    @classmethod
    def from_spec(cls, spec: dict, env=None) -> "OpenAIEmbedder":
        """XVIRAL_EMBED_TIMEOUT (seconds per call, 20), XVIRAL_EMBED_RETRIES (2), XVIRAL_EMBED_BATCH (256), XVIRAL_EMBED_CONCURRENCY (4)."""
        e = (env or os.environ).get
        return cls(spec["model"], spec["dimensions"], batch=int(e("XVIRAL_EMBED_BATCH") or 256), timeout=float(e("XVIRAL_EMBED_TIMEOUT") or 20),
                   retries=int(e("XVIRAL_EMBED_RETRIES") or 2), concurrency=int(e("XVIRAL_EMBED_CONCURRENCY") or 4),
                   max_chars=spec.get("max_chars", 8000), price_per_mtok=spec.get("price_per_mtok"))

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "dimensions": self.dim, "endpoint": self.base_url}

    def take_usage(self) -> dict:
        return self.usage.take()

    def _post(self, body: dict) -> tuple[int, dict]:
        import requests

        r = (self._session or requests).post(self.base_url + "/embeddings", headers={"Authorization": f"Bearer {self.api_key}"}, json=body, timeout=self.timeout)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"error": {"message": (r.text or "")[:200]}}

    def _one(self, texts: list[str]) -> np.ndarray:
        body = {"model": self.model, "input": texts, "encoding_format": "base64", "dimensions": self.dim}
        t0, err = time.perf_counter(), None
        for attempt in range(self.retries + 1):
            try:
                status, js = self._post(body)
            except Exception as e:      # noqa: BLE001  timeouts and connection errors
                err = f"{type(e).__name__}"
            else:
                if status == 200:
                    try:
                        data = sorted(js["data"], key=lambda d: d["index"])
                        E = np.stack([np.frombuffer(base64.b64decode(d["embedding"]), dtype="<f4") if isinstance(d["embedding"], str)
                                      else np.asarray(d["embedding"], dtype=np.float32) for d in data])
                        tokens = int((js.get("usage") or {}).get("total_tokens", 0))
                    except (KeyError, TypeError, ValueError) as e:
                        err = f"unreadable answer ({type(e).__name__})"
                    else:
                        if E.shape != (len(texts), self.dim):
                            err = f"answer of shape {E.shape}, expected {(len(texts), self.dim)}"
                        else:
                            self.usage.add(calls=1, texts=len(texts), tokens=tokens, usd=tokens * self.price / 1e6, seconds=time.perf_counter() - t0)
                            return E
                else:
                    msg = ((js.get("error") or {}).get("message") if isinstance(js.get("error"), dict) else js.get("error")) or ""
                    err = f"HTTP {status}" + (f": {str(msg)[:120]}" if msg else "")
                    if status not in (408, 409, 429, 500, 502, 503, 504):
                        break               # a refused request will be refused again
            if attempt < self.retries:
                time.sleep(min(0.4 * 2 ** attempt, 4.0))
        self.usage.add(calls=1, seconds=time.perf_counter() - t0, error=err)
        raise EmbeddingError(f"the embedding provider ({self.provider}, {self.model}) did not return vectors: {err}")

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self.dim), np.float32)
        clean = [(t or " ")[:self.max_chars] for t in texts]
        b = batch_size or self.batch
        parts = [clean[i:i + b] for i in range(0, len(clean), b)]
        if len(parts) == 1:
            out = [self._one(parts[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(self.concurrency, len(parts))) as ex:
                out = list(ex.map(self._one, parts))
        return _unit16(np.concatenate(out))


class StubEmbedder(OpenAIEmbedder):
    """An API embedder that never leaves the process, for tests and dry runs. Deterministic: a text's vector depends
    only on the text (signed hashing of words and character 4-grams, so similar texts get similar vectors).
    `fail` (True, or a set of call numbers) makes calls fail; `delay` seconds per call; `calls` keeps every request body."""

    provider = "none (built-in stub, nothing is sent)"

    def __init__(self, dimensions: int = 1024, fail=False, delay: float = 0.0, model: str = "stub-embedding", price_per_mtok: float = 0.13, **kw):
        super().__init__(model, dimensions, price_per_mtok=price_per_mtok, retries=kw.pop("retries", 0), **kw)
        self.base_url, self.fail, self.delay, self.calls = "stub://", fail, delay, []
        self._lock = threading.Lock()

    def vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        t = (text or "").lower()
        words = t.split()
        for f in words + [a + "_" + b for a, b in zip(words, words[1:])] + [t[j:j + 4] for j in range(max(len(t) - 3, 0))]:
            h = int.from_bytes(hashlib.blake2b(f.encode("utf-8", "replace"), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 40) & 1 else -1.0
        if not v.any():
            v[0] = 1.0
        return v / np.linalg.norm(v)

    def _post(self, body: dict) -> tuple[int, dict]:
        with self._lock:
            n = len(self.calls)
            self.calls.append(body)
        if self.delay:
            time.sleep(self.delay)
        if self.fail is True or (isinstance(self.fail, (set, list, tuple)) and n in self.fail):
            return 503, {"error": {"message": "stub failure"}}
        data = [{"index": i, "embedding": base64.b64encode(self.vector(t).astype("<f4").tobytes()).decode()} for i, t in enumerate(body["input"])]
        return 200, {"data": data, "usage": {"total_tokens": sum(max(1, len(t) // 4) for t in body["input"])}}


class LocalEncoder:
    """The sentence-transformers encoder behind the same interface (nothing is sent, nothing is billed)."""

    provider = "local"
    local = True

    def __init__(self, model: str = BGE_SMALL, device: str = "cpu", batch: int = 16):
        from xviral.embed.text_small import SentenceEmbedder

        self.model, self.batch = model, batch
        self._enc = SentenceEmbedder(model, device=device)
        self.dim = self._enc.dim

    def describe(self) -> dict:
        return {"provider": "local", "model": self.model, "dimensions": self.dim, "endpoint": None}

    def take_usage(self) -> dict:
        return {"calls": 0, "texts": 0, "tokens": 0, "usd": 0.0, "seconds": 0.0, "error": None}

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        return self._enc.encode(texts, batch_size=batch_size or self.batch)


def make_encoder(spec: dict, device: str = "cpu", batch: int = 16, env=None):
    """The encoder object for a spec. XVIRAL_EMBED_STUB=1 replaces an API encoder with the stub (dry runs, tests)."""
    if spec["provider"] == "local":
        return LocalEncoder(spec["model"], device=device, batch=batch)
    if spec["provider"] == "openai":
        if ((env or os.environ).get("XVIRAL_EMBED_STUB") or "") in ("1", "true", "yes"):
            return StubEmbedder(spec["dimensions"], model=spec["model"], price_per_mtok=spec.get("price_per_mtok", 0.0))
        return OpenAIEmbedder.from_spec(spec, env)
    raise ValueError(f"unknown embedding provider {spec['provider']!r}")
