#!/usr/bin/env python3
"""The combine track's training table: the broad table (tracks/general/table, 1,154,680 rows) with the post embedding
replaced by OpenAI text-embedding-3-large (first 1,024 dimensions) and semantic IDs reassigned. Everything else is
hard-linked unchanged, so rows, splits, roles and every other input are identical to the general track's table.

    python scripts/combine/build_table.py --emb data/combine/te3l1024_broad.npy --out data/combine/table_te3l_broad

Semantic IDs use the embedding track's codebook (tracks/embed_api/serving/te3l1024/sid_codebook.npz, trained on
training-split rows of the original table), so one codebook serves the models of both tables. Neighbour lists stay
as built with bge-small, as in the embedding track; serving passes none.
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
import numpy as np
from xviral.embed import sid
from xviral.embed.api import ENCODERS

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/tracks/general/table"); ap.add_argument("--emb", default="data/combine/te3l1024_broad.npy")
ap.add_argument("--codebook", default="data/tracks/embed_api/serving/te3l1024/sid_codebook.npz"); ap.add_argument("--out", default="data/combine/table_te3l_broad")
ap.add_argument("--encoder", default="openai:text-embedding-3-large@1024")
a = ap.parse_args()
src, out = Path(a.src).resolve(), Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
E = np.load(a.emb, mmap_mode="r")
meta = json.loads((src / "meta.json").read_text())
assert len(E) == meta["n"], (E.shape, meta["n"])
if not (out / "emb.npy").exists():
    os.link(os.path.abspath(a.emb), out / "emb.npy")
old = np.load(src / "sid.npy", mmap_mode="r")
codes = np.concatenate([sid.phoenix_codes(sid.assign(np.asarray(E[i:i + 50_000], dtype=np.float32), a.codebook)) for i in range(0, len(E), 50_000)])
np.save(out / "sid.npy", codes.astype(old.dtype))
for p in src.glob("*"):
    if p.name not in ("emb.npy", "sid.npy", "meta.json") and p.is_file() and not (out / p.name).exists():
        os.link(p, out / p.name)
shutil.copy(a.codebook, out / "sid_codebook.npz")
enc = dict(ENCODERS[a.encoder])
(out / "meta.json").write_text(json.dumps(meta | {"emb_dim": int(E.shape[1]), "encoder": enc}, indent=2))
(out / "recipe.json").write_text(json.dumps({"parts": enc["parts"]}))
rng = np.random.default_rng(0)
i = np.sort(rng.integers(0, len(E), 3000)); X = np.asarray(E[i], dtype=np.float32); c = (X @ X.T)[np.triu_indices(3000, 1)]
print(f"table written {out}: emb {E.shape}, random-pair cosine {c.mean():.3f} +/- {c.std():.3f}, distinct first-level ids {len(np.unique(codes[:, 0]))}", flush=True)
