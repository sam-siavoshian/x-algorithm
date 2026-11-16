"""Zenodo 21783628 (193 brand accounts, twitterapi.io pages) -> post schema, with the reader's guards on every file."""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import polars as pl
from xviral.sources import twitterapi_io as R

IN = Path("data/raw/zenodo_brands/x/zenodoUpload/originalData/tweet"); OUT = Path("data/track_data/sources/zenodo_brands")
SNAP = sys.argv[1] if len(sys.argv) > 1 else "2025-11-30"

def one(f):
    tail = f.stem.rsplit("_", 1)[-1]
    try:
        posts, side, rep = R.read_file(f, SNAP, None)
        rep["file_author_id"] = int(tail) if tail.isdigit() else None
        rep["own_rows"] = posts.filter(pl.col("author_id") == rep["file_author_id"]).height
        rep["max_created"] = posts["created_at_ms"].max()
        posts.write_parquet(OUT / "posts" / f"{f.stem}.parquet", compression="zstd"); side.write_parquet(OUT / "threads" / f"{f.stem}.parquet", compression="zstd")
    except Exception as e:
        rep = {"file": f.name, "error": repr(e)[:300]}
    return rep

if __name__ == "__main__":
    (OUT / "posts").mkdir(parents=True, exist_ok=True); (OUT / "threads").mkdir(parents=True, exist_ok=True)
    files = sorted(IN.glob("*.json"), key=lambda f: f.stat().st_size)
    with ProcessPoolExecutor(2) as ex: reps = list(ex.map(one, files))
    json.dump(reps, open(OUT / "file_checks.json", "w"), indent=0)
    print(len(reps), "files;", sum("error" in r for r in reps), "errors")
