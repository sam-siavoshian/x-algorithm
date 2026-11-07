"""Sharded fp16 embedding store keyed by post_id.

Layout: <dir>/shard-00000.npy (float16 [n, d]) and shard-00000.ids.npy (int64 [n]).
Shards are memory-mapped on read, so a multi-GB store costs no RAM until rows
are gathered. Writing is append-only and resumable: `missing(ids)` returns the
IDs not yet embedded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class EmbeddingStore:
    def __init__(self, directory: str | Path, dim: int | None = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._index: dict[int, tuple[int, int]] | None = None

    def _shards(self) -> list[Path]:
        return sorted(p for p in self.dir.glob("shard-*.npy") if not p.name.endswith(".ids.npy"))

    def write(self, ids: np.ndarray, vecs: np.ndarray) -> Path:
        ids = np.asarray(ids, dtype=np.int64)
        vecs = np.asarray(vecs, dtype=np.float16)
        assert vecs.ndim == 2 and len(ids) == len(vecs)
        if self.dim is not None and vecs.shape[1] != self.dim:
            raise ValueError(f"dim mismatch {vecs.shape[1]} != {self.dim}")
        k = len(self._shards())
        p = self.dir / f"shard-{k:05d}.npy"
        np.save(self.dir / f"shard-{k:05d}.ids.npy", ids)
        np.save(p, vecs)
        self._index = None
        return p

    def _build_index(self) -> dict[int, tuple[int, int]]:
        if self._index is None:
            idx: dict[int, tuple[int, int]] = {}
            for s, p in enumerate(self._shards()):
                for r, pid in enumerate(np.load(p.with_name(p.name.replace(".npy", ".ids.npy")))):
                    idx[int(pid)] = (s, r)
            self._index = idx
        return self._index

    def ids(self) -> np.ndarray:
        parts = [np.load(p.with_name(p.name.replace(".npy", ".ids.npy"))) for p in self._shards()]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def missing(self, ids) -> list[int]:
        idx = self._build_index()
        return [int(i) for i in ids if int(i) not in idx]

    def get(self, ids, fill_nan: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Returns (vectors float32 [len(ids), d], found mask). Missing rows are NaN or zero."""
        idx = self._build_index()
        shards = [np.load(p, mmap_mode="r") for p in self._shards()]
        if not shards:
            raise FileNotFoundError(f"empty store {self.dir}")
        d = shards[0].shape[1]
        out = np.full((len(ids), d), np.nan if fill_nan else 0.0, dtype=np.float32)
        found = np.zeros(len(ids), dtype=bool)
        for i, pid in enumerate(ids):
            loc = idx.get(int(pid))
            if loc is not None:
                out[i] = shards[loc[0]][loc[1]]
                found[i] = True
        return out, found

    def all(self) -> tuple[np.ndarray, np.ndarray]:
        ps = self._shards()
        ids = self.ids()
        vecs = np.concatenate([np.load(p, mmap_mode="r") for p in ps]) if ps else np.zeros((0, 0), dtype=np.float16)
        return ids, vecs
