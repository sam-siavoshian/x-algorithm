"""An embedding recipe: how the model's post embedding is assembled from raw encoder outputs, identically at
training time (api_table.py) and at serving time (bench_api.py).

A recipe is a JSON file:
    {"parts": [{"src": "bge"}, {"src": "te3l", "dim": 1024}],        # raw vectors, each shortened and renormalised
     "adapter": "adapter.pt" | null,                                  # optional projection network (adapter.py) on the parts
     "keep_parts": true}                                              # with an adapter: output is [parts..., adapted] or adapted alone
Every block of the output is unit length and the whole is divided by sqrt(blocks), so the result is a unit vector,
as the trunk expects (it multiplies by sqrt(dim)).

Sources: "bge" = BAAI/bge-small-en-v1.5 (384), "te3l" = text-embedding-3-large (3072 native), "te3s" = text-embedding-3-small (1536).
"""
from __future__ import annotations

import json
import os

import numpy as np


def _unit(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float32)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)


def load(path: str) -> dict:
    r = json.load(open(path))
    if r.get("adapter") and not os.path.isabs(r["adapter"]):
        r["adapter"] = os.path.join(os.path.dirname(os.path.abspath(path)), r["adapter"])
    return r


def sources(recipe: dict) -> list[str]:
    return sorted({p["src"] for p in recipe["parts"]})


def apply(recipe: dict, raw: dict[str, np.ndarray], batch: int = 65536) -> np.ndarray:
    """raw[src] is float [n, native or larger-than-needed dim]; returns float32 [n, d], unit length."""
    n = len(next(iter(raw.values())))
    net = None
    if recipe.get("adapter"):
        from adapter import load_adapter
        net = load_adapter(recipe["adapter"])
    out = None
    for i in range(0, n, batch):
        blocks = [_unit(np.asarray(raw[p["src"]][i:i + batch], dtype=np.float32)[:, :p.get("dim") or None]) for p in recipe["parts"]]
        if net is not None:
            z = _unit(net.embed(np.hstack(blocks)))
            blocks = blocks + [z] if recipe.get("keep_parts", True) else [z]
        part = np.hstack(blocks) / np.float32(np.sqrt(len(blocks)))
        if out is None:
            out = np.empty((n, part.shape[1]), dtype=np.float32)        # one allocation: a full table at 1,408 dimensions is 2.6 GB
        out[i:i + len(part)] = part
    return out
