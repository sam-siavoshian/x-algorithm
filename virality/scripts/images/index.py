#!/usr/bin/env python3
"""The shared image index and the derived files other workers reuse.

    python scripts/images/index.py     # after fetch.py, represent.py and label.py

Writes, under data/images/:
  index.parquet       post_id, media_idx, url, status, width, height (original, from the corpus), bytes, priority, fetched_at shard
  embeddings.parquet  post_id, media_idx, siglip (768, fp16 list), clip (512) for every image represented
  attributes.parquet  post_id, every pixel attribute, aesthetic, zero-shot type and scores, AI scores, tertiles, confident flags
"""
import glob, json, os
import numpy as np, polars as pl
S = pl.read_parquet("data/images/selection.parquet").select("post_id", "author_id", "media_idx", "url", pl.col("width").alias("orig_w"), pl.col("height").alias("orig_h"), "n_media", "priority")
R = pl.concat([pl.read_parquet(f, columns=["post_id", "media_idx", "status", "bytes"]).with_columns(pl.lit(os.path.basename(f)).alias("shard")) for f in sorted(glob.glob("data/images/raw/img-*.parquet"))])
I = S.join(R, on=["post_id", "media_idx"], how="inner")
I.write_parquet("data/images/index.parquet")
print("index:", I.height, "requested;", I.filter(pl.col("status") == 200).height, "fetched;", I.group_by("status").len().sort("len", descending=True).head(5).to_dicts())
ids, si, cl = [], [], []
for f in sorted(glob.glob("data/images/rep/img-*.npz")):
    z = np.load(f); ids.append(z["post_id"]); si.append(z["siglip"]); cl.append(z["clip"])
ids = np.concatenate(ids); si = np.concatenate(si); cl = np.concatenate(cl)
P = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/images/rep/img-*.parquet"))], how="diagonal_relaxed")
assert P.height == len(ids) and (P["post_id"].to_numpy() == ids).all()
E = pl.DataFrame({"post_id": ids, "media_idx": P["media_idx"], "siglip": si.astype(np.float32), "clip": cl.astype(np.float32)})
E.write_parquet("data/images/embeddings.parquet")
L = pl.read_parquet("data/images/labels.parquet")
A = P.join(S.select("post_id", "author_id", "orig_w", "orig_h", "n_media", "priority"), on="post_id", how="left").join(L.drop("media_idx"), on="post_id", how="left")
if os.path.exists("data/images/ai_detector.parquet"):
    A = A.drop("ai_detector_p").join(pl.read_parquet("data/images/ai_detector.parquet"), on="post_id", how="left")
A = A.with_columns((pl.col("orig_w").cast(pl.Float64) * pl.col("orig_h")).alias("resolution"))
# Quality tertiles are cut inside each image type (a screenshot is compared with screenshots), text cover over all images.
for q, by in (("aesthetic", "itype"), ("sharpness", "itype"), ("resolution", "itype"), ("colourfulness", "itype"), ("text_area_share", None)):
    r = (pl.col(q).rank("ordinal") / pl.col(q).count()) if by is None else (pl.col(q).rank("ordinal").over(by) / pl.col(q).count().over(by))
    A = A.with_columns(pl.when(pl.col(q).is_null()).then(None).when(r <= 1 / 3).then(0).when(r <= 2 / 3).then(1).otherwise(2).alias(f"{q}_tert"))
med = A.group_by("itype").agg(pl.col("margin").median().alias("_m"))
A = A.join(med, on="itype").with_columns((pl.col("margin") >= pl.col("_m")).alias("confident")).drop("_m")
# "AI-generated" confident subset: illustration/render class and detector p >= 0.9999 (74% AI among 23 eye-labelled
# images that pass, threshold read off the same 38 labels, see data/images/ai_validation.json). Not a reliable label.
A = A.with_columns(((pl.col("itype_raw") == "illustration_render") & (pl.col("ai_detector_p").fill_null(0) >= 0.9999)).alias("ai_confident"))
A.write_parquet("data/images/attributes.parquet")
json.dump(sorted(L["itype"].unique().to_list()), open("data/images/type_names.json", "w"))
print("attributes:", A.height, "images,", A["author_id"].n_unique(), "accounts")
print(A.group_by("itype").agg(pl.len(), pl.col("confident").sum(), pl.col("aesthetic").median(), pl.col("text_area_share").median()).sort("len", descending=True))
