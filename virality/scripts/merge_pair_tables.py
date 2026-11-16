#!/usr/bin/env python3
"""Join the per-slice pair tables of build_pair_table.py into one.    python scripts/merge_pair_tables.py <out> <slice dir> [<slice dir> ...]"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

out, parts = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
out.mkdir(parents=True, exist_ok=True)
names = sorted(p.stem for p in parts[0].glob("*.npy"))
offs = np.cumsum([0] + [len(np.load(p / "post_id.npy", mmap_mode="r")) for p in parts])
poffs = np.cumsum([0] + [pl.read_parquet(p / "pairs.parquet").height for p in parts])
for n in names:
    arrs = [np.load(p / f"{n}.npy") for p in parts]
    if n in ("a_start", "a_end"):
        arrs = [a + o for a, o in zip(arrs, offs)]
    if n == "author_idx":
        arrs = [a + o for a, o in zip(arrs, poffs)]
    np.save(out / f"{n}.npy", np.concatenate(arrs))
frames = [pl.read_parquet(p / "pairs.parquet").with_columns(pl.col("row_a") + int(o), pl.col("row_b") + int(o), pl.col("anchor") + int(o)) for p, o in zip(parts, offs)]
pl.concat(frames).write_parquet(out / "pairs.parquet")
meta = json.loads((parts[0] / "meta.json").read_text())
meta.update(n=int(offs[-1]), pairs=int(poffs[-1]))
(out / "meta.json").write_text(json.dumps(meta))
print("merged", int(poffs[-1]), "pairs,", int(offs[-1]), "rows")
