#!/usr/bin/env python3
"""A copy of the pairs track's pair table with the src_regime input added, for fine-tuning models trained with it.

    python scripts/expand/pair_table_regime.py --pair-table data/tracks/pairs/pair_table --out data/expand/pair_table_src

History rows take the regime of their source (looked up by post id in the filtered corpus); the two drafts of a pair
take the commonest regime of that pair's history, as model.features.with_src_regime does at serving time.
Every other array is linked, not copied.
"""
import argparse, json, os
from pathlib import Path
import numpy as np, polars as pl
from xviral.model import features
ap = argparse.ArgumentParser(); ap.add_argument("--pair-table", default="data/tracks/pairs/pair_table"); ap.add_argument("--out", default="data/expand/pair_table_src")
ap.add_argument("--posts", default="data/work/filtered/posts.parquet"); args = ap.parse_args()
src, out = Path(args.pair_table), Path(args.out); out.mkdir(parents=True, exist_ok=True)
ids, a_idx = np.load(src / "post_id.npy"), np.load(src / "author_idx.npy")
names = dict(features.CAT_SPEC)["src_regime"]
look = pl.scan_parquet(args.posts).select("post_id", "source").filter(pl.col("post_id").is_in(pl.Series(ids[ids > 0]).unique().implode())).collect()
d = pl.DataFrame({"post_id": ids, "block": a_idx}).join(look, on="post_id", how="left", maintain_order="left")
d = d.with_columns(pl.col("source").replace_strict(features.SRC_REGIME, default=None, return_dtype=pl.String).alias("rg"))
d = d.with_columns(pl.coalesce(pl.col("rg"), pl.col("rg").drop_nulls().mode().first().over("block"), pl.lit("unknown")).alias("rg"))
code = d["rg"].replace_strict({s: i for i, s in enumerate(names)}, default=0, return_dtype=pl.Int32).to_numpy()
np.save(out / "cat.npy", np.concatenate([np.load(src / "cat.npy"), code[:, None].astype(np.int32)], axis=1))
meta = json.loads((src / "meta.json").read_text()); meta["cat_names"] = meta["cat_names"] + ["src_regime"]; meta["cat_cards"] = meta["cat_cards"] + [len(names)]
(out / "meta.json").write_text(json.dumps(meta, indent=2))
for f in src.iterdir():
    if f.name not in ("cat.npy", "meta.json") and not (out / f.name).exists():
        os.symlink(f.resolve(), out / f.name)
print({n: int(c) for n, c in zip(names, np.bincount(code, minlength=len(names)))}, "rows", len(ids), "drafts", int((ids <= 0).sum()))
