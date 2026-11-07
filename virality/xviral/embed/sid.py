"""Semantic IDs: 6 x 256 residual-quantization codes of the post embedding.

Training uses this repository's JAX RQ-KMeans trainer, phoenix/reference/sid_codebook.py,
through its own CLI. Assignment uses phoenix/reference/sid_io.assign_sids (NumPy).
Phoenix stores codes +1 so that 0 means "missing" (recsys_batch.py:1259-1261);
`phoenix_codes` applies that shift.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from xviral.embed.mm import REFERENCE_DIR, reference_module

log = logging.getLogger(__name__)


def train(embeddings_npy: str | Path, codebook_out: str | Path, num_levels: int = 6, num_centroids: int = 256,
          iters: int = 25, sample_size: int | None = None, seed: int = 42) -> Path:
    cmd = [sys.executable, str(REFERENCE_DIR / "sid_codebook.py"), "train",
           "--training-data", str(embeddings_npy), "--codebook-out", str(codebook_out),
           "--num-levels", str(num_levels), "--num-centroids", str(num_centroids), "--iters", str(iters), "--seed", str(seed)]
    if sample_size:
        cmd += ["--sample-size", str(sample_size)]
    log.info("sid train: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REFERENCE_DIR))
    return Path(codebook_out)


def assign(embeddings: np.ndarray, codebook_path: str | Path) -> np.ndarray:
    """Codes in [0, 256) per level, shape [N, levels]."""
    return reference_module("sid_io").assign_sids(np.asarray(embeddings, dtype=np.float32), codebook_path)


def phoenix_codes(codes: np.ndarray, found: np.ndarray | None = None) -> np.ndarray:
    out = np.asarray(codes, dtype=np.int32) + 1
    if found is not None:
        out[~found] = 0
    return out.astype(np.uint16)
