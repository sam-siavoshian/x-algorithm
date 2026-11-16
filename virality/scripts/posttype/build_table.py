#!/usr/bin/env python3
"""Extend the trained-on feature table with the post-type inputs (xviral/features/post_type.py).

    python scripts/posttype/build_table.py [--table data/work/model/table] [--out data/work/model/table_pt]

The stored table is reused as it is (embeddings, semantic IDs, neighbours, labels); only num.npy and meta.json are
rewritten, with the new columns appended and normalised from training rows exactly as model/features.py _arrays does.
Trailing values are computed over the account's posts in work/filtered/posts.parquet, the corpus serving draws an
account's history from. Also writes <out>/post_type.parquet (post_id, pt_type and the raw columns).
"""
import argparse, json, os, shutil, warnings
import numpy as np, polars as pl
from xviral.features import post_type as PTF
from xviral.model import features as F

ap = argparse.ArgumentParser()
ap.add_argument("--table", default="data/work/model/table"); ap.add_argument("--out", default="data/work/model/table_pt")
ap.add_argument("--rows", default="data/work/dataset/rows.parquet"); ap.add_argument("--posts", default="data/work/filtered/posts.parquet")
args = ap.parse_args()
rows = pl.read_parquet(args.rows, columns=["post_id", "author_id", "created_at_ms", "split"]).sort("author_id", "created_at_ms", "post_id")
ids = np.load(f"{args.table}/post_id.npy")
assert np.array_equal(rows["post_id"].to_numpy(), ids), "rows.parquet does not match the table order"
d = (pl.scan_parquet(args.posts).select("post_id", "author_id", "created_at_ms", "text", "kind", "likes", "media", "source")
     .filter(pl.col("author_id").is_in(rows["author_id"].unique().implode())).collect())
print("posts of the table's accounts:", d.height, "table rows found there:", d.filter(pl.col("post_id").is_in(rows["post_id"].implode())).height, "of", rows.height)
feats = PTF.post_type_features(d, only=rows["post_id"])
x = rows.join(feats, on="post_id", how="left")
assert x.height == rows.height and np.array_equal(x["post_id"].to_numpy(), ids)
print(x.group_by("split", "pt_type").len().pivot("split", index="pt_type", values="len").sort("train", descending=True))

meta = json.load(open(f"{args.table}/meta.json"))
old = np.load(f"{args.table}/num.npy")
nb = len(meta["num_base"]); assert old.shape[1] == len(meta["num_names"])
how = dict(PTF.NUM_SPEC); names = list(PTF.COLUMNS)
X = x.select([F._num_expr(c, how[c]).alias(c) for c in names]).to_numpy().astype(np.float32)
train = np.load(f"{args.table}/split.npy") == 0
miss = np.isnan(X)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    mean = np.nan_to_num(np.nanmean(np.where(train[:, None], X, np.nan), axis=0)); std = np.nan_to_num(np.nanstd(np.where(train[:, None], X, np.nan), axis=0))
std = np.where(std < 1e-6, 1.0, std)
miss_cols = [i for i in range(X.shape[1]) if 0.001 < miss[train, i].mean()]
Xs = np.nan_to_num((X - mean) / std).clip(-8, 8)
num = np.concatenate([old[:, :nb], Xs, old[:, nb:], miss[:, miss_cols].astype(np.float32)], axis=1)
meta2 = dict(meta)
meta2["num_base"] = meta["num_base"] + names
meta2["num_names"] = meta["num_names"][:nb] + names + meta["num_names"][nb:] + [f"{names[i]}__missing" for i in miss_cols]
meta2["num_mean"] = meta["num_mean"] + mean.tolist(); meta2["num_std"] = meta["num_std"] + std.tolist()
meta2["num_missing_cols"] = meta["num_missing_cols"] + [nb + i for i in miss_cols]
assert num.shape[1] == len(meta2["num_names"])
os.makedirs(args.out, exist_ok=True)
for f in os.listdir(args.table):
    if f in ("num.npy", "meta.json"): continue
    dst = os.path.join(args.out, f)
    if not os.path.exists(dst): os.symlink(os.path.abspath(os.path.join(args.table, f)), dst)
np.save(f"{args.out}/num.npy", num); json.dump(meta2, open(f"{args.out}/meta.json", "w"), indent=2)
x.write_parquet(f"{args.out}/post_type.parquet")
print("wrote", args.out, "numeric inputs", old.shape[1], "->", num.shape[1], "missing indicators for", [names[i] for i in miss_cols])
