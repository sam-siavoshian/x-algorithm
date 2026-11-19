#!/usr/bin/env python3
"""Step 2: a model table whose post embedding follows a recipe (recipe.py), everything else unchanged.

    python research/tracks/embed_api/api_table.py --table data/work/model/table --recipe data/api/recipes/te3l1024.json \
        --raw te3l=data/api/te3l_table_full.npy --out data/api/table_te3l1024 [--pca 512]

The source table's own emb.npy is the "bge" source. --raw files are aligned with the table's rows (texts.parquet).
The semantic-ID codebook is retrained on the new embedding (training-split rows only) and IDs reassigned.
--pca k reduces the assembled embedding to k dimensions with a PCA fitted on training-split rows (mean and components
are written to pca.npz and must be applied at serving). Neighbour lists and neighbour-derived columns stay as built
with bge-small, as in the encoder track: they are the retrieval index and serving passes none.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import recipe as R  # noqa: E402


def fit_pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(0)
    C = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    for i in range(0, len(X), 50_000):
        Z = (X[i:i + 50_000] - mu).astype(np.float64)
        C += Z.T @ Z
    w, V = np.linalg.eigh(C / len(X))
    return mu.astype(np.float32), V[:, ::-1][:, :k].T.astype(np.float32).copy()


def pca_apply(E: np.ndarray, mu: np.ndarray, comp: np.ndarray) -> np.ndarray:
    Z = (E - mu) @ comp.T
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--raw", action="append", default=[], help="src=aligned.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca", type=int, default=0)
    ap.add_argument("--sid-sample", type=int, default=150_000)
    args = ap.parse_args()
    from xviral.embed import sid
    src, out = Path(args.table).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rec = R.load(args.recipe)
    raw = {"bge": np.load(src / "emb.npy", mmap_mode="r")} | {s.split("=")[0]: np.load(s.split("=")[1], mmap_mode="r") for s in args.raw}
    E = R.apply(rec, {k: raw[k] for k in R.sources(rec)})
    train = np.load(src / "split.npy") == 0
    if args.pca:
        mu, comp = fit_pca(E[train], args.pca)
        np.savez(out / "pca.npz", mu=mu, comp=comp)
        E = pca_apply(E, mu, comp).astype(np.float32)
    np.save(out / "emb.npy", E.astype(np.float16))
    cb = out / "sid_codebook.npz"
    rng = np.random.default_rng(0)
    np.save(out / "sid_train.npy", E[rng.choice(np.flatnonzero(train), min(args.sid_sample, int(train.sum())), replace=False)])
    sid.train(out / "sid_train.npy", cb, 6, 256, 25, None)
    os.remove(out / "sid_train.npy")
    codes = np.concatenate([sid.phoenix_codes(sid.assign(E[i:i + 50_000], cb)) for i in range(0, len(E), 50_000)])
    np.save(out / "sid.npy", codes.astype(np.load(src / "sid.npy", mmap_mode="r").dtype))
    for p in src.glob("*"):
        if p.name not in ("emb.npy", "sid.npy", "meta.json") and p.is_file() and not (out / p.name).exists():
            os.link(p, out / p.name)
    meta = json.loads((src / "meta.json").read_text()) | {"emb_dim": int(E.shape[1]), "encoder": rec}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.copy(args.recipe, out / "recipe.json")
    if rec.get("adapter"):
        shutil.copy(rec["adapter"], out / os.path.basename(rec["adapter"]))
    i = rng.integers(0, len(E), 3000)
    c = (E[i] @ E[i].T)[np.triu_indices(3000, 1)]
    print(f"table written {out} emb {E.shape}; random-pair cosine {c.mean():.3f} +/- {c.std():.3f}", flush=True)


if __name__ == "__main__":
    main()
