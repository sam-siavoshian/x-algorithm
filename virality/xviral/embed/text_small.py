"""Cheap text embeddings for the relevance pre-pass and the policy classifier.

Production: Qwen3-Embedding-0.6B (1024-d, multilingual, Apache-2.0) through
sentence-transformers on a GPU, about 100x cheaper per post than the 8B
multimodal encoder and good enough for a keep/drop decision.

'hashing' is a CPU fallback (signed feature hashing of word and character
n-grams, L2-normalised). It exists so the relevance logic can be exercised
in tests and on machines without a GPU; it is not used for production runs.
"""

from __future__ import annotations

import logging
import re
import zlib

import numpy as np

log = logging.getLogger(__name__)

_WORD = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(self, texts: list[str], batch_size: int = 0, query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = (t or "").lower()
            toks = _WORD.findall(t)
            feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])] + [t[j:j + 4] for j in range(max(len(t) - 3, 0))]
            for f in feats:
                h = zlib.crc32(f.encode())
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


class SentenceEmbedder:
    def __init__(self, model_id: str, device: str = "cuda"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("install the 'encoders' extra (sentence-transformers, torch) on the GPU box") from e
        kwargs = {"model_kwargs": {"dtype": "float32"}} if device == "cpu" else {}
        self.model = SentenceTransformer(model_id, device=device, **kwargs)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 256, query: bool = False) -> np.ndarray:
        """query=True applies the model's query prompt (Qwen3-Embedding is instruction-tuned for queries)."""
        kw = {}
        if query and "query" in (getattr(self.model, "prompts", None) or {}):
            kw["prompt_name"] = "query"
        return self.model.encode([t or "" for t in texts], batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False, **kw).astype(np.float32)


def make_embedder(name: str, device: str = "cuda"):
    if name == "hashing":
        return HashingEmbedder()
    return SentenceEmbedder(name, device=device)
