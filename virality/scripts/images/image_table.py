#!/usr/bin/env python3
"""Training tables for the image experiment: the accounts that have fetched images, every row of theirs, with and
without the image inputs (otherwise identical, so a model trained on each differs only by the image).

    python scripts/images/image_table.py --min-images 5 --out data/images/table      # writes <out>_img and <out>_ctl

Image inputs (auxiliary inputs of PostEncoder, present only for posts whose first photo was fetched):
  img      SigLIP base embedding of the first photo, unit norm, 768-d
  imgattr  the attributes that survive (see REPORT.md), standardised on the training split, divided by sqrt(dim)
Also writes data/images/aux_img.npz and aux_imgattr.npz (post_id, vec) for the serving path (tac_serve_eval --aux).
"""
import argparse, json, os, shutil
import numpy as np, polars as pl
ap = argparse.ArgumentParser(); ap.add_argument("--src", default="data/work/model/table"); ap.add_argument("--out", default="data/images/table"); ap.add_argument("--min-images", type=int, default=5)
ap.add_argument("--attrs", default="data/images/attributes.parquet"); ap.add_argument("--emb", default="data/images/embeddings.parquet")
a = ap.parse_args()
ATTR = ["aesthetic", "log_sharpness", "log_resolution", "log_aspect", "text_area_share", "log_text_boxes", "has_face", "face_area_share", "brightness", "saturation", "colourfulness", "dark_share", "white_share", "edge_density", "several_images"]
src = a.src; meta = json.load(open(f"{src}/meta.json"))
pid = np.load(f"{src}/post_id.npy"); aidx = np.load(f"{src}/author_idx.npy"); split = np.load(f"{src}/split.npy")
E = pl.read_parquet(a.emb); A = pl.read_parquet(a.attrs)
A = A.with_columns(pl.col("sharpness").log1p().alias("log_sharpness"), (pl.col("orig_w").cast(pl.Float64) * pl.col("orig_h")).log1p().alias("log_resolution"),
                   (pl.col("orig_w").cast(pl.Float64) / pl.col("orig_h")).log().alias("log_aspect"), pl.col("text_boxes").cast(pl.Float64).log1p().alias("log_text_boxes"),
                   (pl.col("faces") > 0).cast(pl.Float64).alias("has_face"), (pl.col("n_media") >= 2).cast(pl.Float64).alias("several_images"))
have = set(E["post_id"].to_list())
m = np.isin(pid, np.fromiter(have, np.int64, len(have)))
cnt = np.bincount(aidx, weights=m, minlength=aidx.max() + 1)
keep_a = cnt >= a.min_images
keep = keep_a[aidx]
print(f"accounts with {a.min_images}+ fetched images: {keep_a.sum()} of {len(keep_a)}; rows kept {keep.sum()} of {len(keep)}; rows with an image {m[keep].sum()}")
sel = np.flatnonzero(keep); n = len(sel)
pos = np.full(len(pid), -1, np.int64); pos[sel] = np.arange(n)
def subset(out, with_img):
    if os.path.exists(out): shutil.rmtree(out)
    os.makedirs(out)
    for f in os.listdir(src):
        if not f.endswith(".npy"): continue
        k = f[:-4]
        if k in ("author_idx", "a_start", "a_end", "nbr_idx"): continue
        arr = np.load(f"{src}/{f}", mmap_mode="r"); np.save(f"{out}/{f}", np.asarray(arr[sel]))
    au = aidx[sel]; change = np.r_[True, au[1:] != au[:-1]]; starts = np.flatnonzero(change); ends = np.r_[starts[1:], n]; ai = np.cumsum(change) - 1
    np.save(f"{out}/author_idx.npy", ai.astype(np.int32)); np.save(f"{out}/a_start.npy", starts[ai].astype(np.int64)); np.save(f"{out}/a_end.npy", ends[ai].astype(np.int64))
    if os.path.exists(f"{src}/nbr_idx.npy"):
        nb = np.load(f"{src}/nbr_idx.npy", mmap_mode="r")[sel]; nb2 = np.where(nb >= 0, pos[np.maximum(nb, 0)], -1).astype(np.int32)
        np.save(f"{out}/nbr_idx.npy", nb2); cos = np.load(f"{out}/nbr_cos.npy"); cos[nb2 < 0] = 0; np.save(f"{out}/nbr_cos.npy", cos)
    if os.path.exists(f"{src}/texts.parquet"): pl.read_parquet(f"{src}/texts.parquet")[sel.tolist()].write_parquet(f"{out}/texts.parquet")
    sp = np.load(f"{out}/split.npy"); counts = np.load(f"{out}/counts.npy")
    mt = {**meta, "n": n, "n_authors": int(len(starts)), "splits": {k: int((sp == v).sum()) for k, v in {"train": 0, "val": 1, "test": 2}.items()},
          "has_views": int((counts[:, 4] >= 0).sum()), "has_likes": int((counts[:, 0] >= 0).sum()), "audience_rows": int(np.load(f"{out}/audience.npy").sum()), "aux_dims": {}}
    if with_img:
        ids = pid[sel]; J = pl.DataFrame({"post_id": ids}).with_row_index("r")
        je = J.join(E, on="post_id", how="inner"); V = np.zeros((n, 768), np.float16); V[je["r"].to_numpy()] = np.stack(je["siglip"].to_numpy()).astype(np.float16)
        h = np.zeros(n, bool); h[je["r"].to_numpy()] = True
        ja = J.join(A.select("post_id", *ATTR), on="post_id", how="inner"); X = ja.select(ATTR).to_numpy().astype(np.float32)
        tr = np.isin(ja["r"].to_numpy(), np.flatnonzero(sp == 0)); mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0) + 1e-6
        Xs = np.nan_to_num((X - mu) / sd).clip(-5, 5) / np.sqrt(len(ATTR)); W = np.zeros((n, len(ATTR)), np.float16); W[ja["r"].to_numpy()] = Xs
        np.save(f"{out}/aux_img.npy", V); np.save(f"{out}/has_aux_img.npy", h); np.save(f"{out}/aux_imgattr.npy", W); np.save(f"{out}/has_aux_imgattr.npy", h)
        mt["aux_dims"] = {"img": 768, "imgattr": len(ATTR)}; mt["imgattr_names"] = ATTR; mt["imgattr_mean"] = mu.tolist(); mt["imgattr_std"] = sd.tolist()
        # serving files over every image we have (not only this subset), with the same transforms
        Xa = A.select(ATTR).to_numpy().astype(np.float32); Xa = np.nan_to_num((Xa - mu) / sd).clip(-5, 5) / np.sqrt(len(ATTR))
        np.savez("data/images/aux_imgattr.npz", post_id=A["post_id"].to_numpy(), vec=Xa.astype(np.float16))
        np.savez("data/images/aux_img.npz", post_id=E["post_id"].to_numpy(), vec=np.stack(E["siglip"].to_numpy()).astype(np.float16))
        print("image rows by split:", {k: int((h & (sp == v)).sum()) for k, v in {"train": 0, "val": 1, "test": 2}.items()})
    json.dump(mt, open(f"{out}/meta.json", "w"), indent=1)
subset(a.out + "_ctl", False); subset(a.out + "_img", True)
print("tables written")
