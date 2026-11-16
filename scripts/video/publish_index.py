#!/usr/bin/env python3
"""media/videos/index.parquet for the shared dataset: one row per target video post. Video files were not acquired
(see tracks/video/REPORT.md), so duration, bitrate and has_audio are null and status says what exists for the post."""
import json, os, polars as pl
from huggingface_hub import HfApi
T = pl.read_parquet("data/video/targets.parquet")
if os.path.exists("data/video/targets2.parquet"): T = pl.concat([T, pl.read_parquet("data/video/targets2.parquet")], how="diagonal_relaxed").unique("post_id", keep="first")
th = pl.DataFrame([json.loads(x) for x in open("data/video/thumbs.jsonl")], infer_schema_length=None).unique("post_id", keep="last").select("post_id", pl.col("status").alias("thumb_http"), "thumb_w", "thumb_h")
F = set(pl.read_parquet("data/video/thumb_feats.parquet")["post_id"].to_list()) if os.path.exists("data/video/thumb_feats.parquet") else set()
I = T.join(th, on="post_id", how="left").with_columns(
    pl.when(pl.col("thumb_http") == 200).then(pl.lit("thumbnail_only")).when(pl.col("thumb_http").is_null()).then(pl.lit("pending")).otherwise(pl.lit("thumbnail_failed")).alias("status"),
    pl.lit(None, dtype=pl.Float64).alias("duration"), pl.col("thumb_w").alias("width"), pl.col("thumb_h").alias("height"), pl.lit(None, dtype=pl.Int64).alias("bitrate"), pl.lit(None, dtype=pl.Boolean).alias("has_audio"),
    pl.when(pl.col("post_id").is_in(list(F))).then(pl.lit("media/videos/thumb_features.parquet")).otherwise(pl.lit(None, dtype=pl.String)).alias("features_path"), pl.lit(None, dtype=pl.String).alias("audio_path"))
I = I.select("post_id", "status", "duration", "width", "height", "bitrate", "has_audio", "features_path", "audio_path", "author_id", "mtype", "thumb_url", "thumb_http", "akind", "ptype", "prio", "source")
os.makedirs("data/video/pub", exist_ok=True); I.write_parquet("data/video/pub/index.parquet"); print(I.group_by("status").len())
api = HfApi(); R = "sam-siavoshian/xviral-work"
note = """# media/videos (video track)\n\nVideo FILES WERE NOT ACQUIRED. `index.parquet` lists the video and GIF posts this track covers; `duration`, `bitrate`, `has_audio`, `audio_path` are null for every row.\n`width`/`height` are the poster frame's pixel size (the video's aspect ratio). There is no `audio/` folder. See `tracks/video/REPORT.md`.\n\n- `index.parquet`: post_id, status (thumbnail_only | thumbnail_failed | pending), features_path, target metadata.\n- `thumb_features.parquet`: per poster frame: zero-shot scores, first-frame class, faces, text boxes, pixel statistics.\n- `thumb_emb_siglip2_base256.npy` + `thumb_emb_ids.parquet`: SigLIP-2 base (768-d, float16) embedding of the poster frame, row-aligned.\n"""
open("data/video/pub/README.md", "w").write(note)
up = [("data/video/pub/index.parquet", "media/videos/index.parquet"), ("data/video/pub/README.md", "media/videos/README.md")]
if os.path.exists("data/video/pub/thumb_features.parquet"): up += [("data/video/pub/thumb_features.parquet", "media/videos/thumb_features.parquet"), ("data/video/pub/thumb_emb_siglip2_base256.npy", "media/videos/thumb_emb_siglip2_base256.npy"), ("data/video/pub/thumb_emb_ids.parquet", "media/videos/thumb_emb_ids.parquet")]
for a, b in up: api.upload_file(path_or_fileobj=a, path_in_repo=b, repo_id=R, repo_type="dataset", commit_message="video track: " + os.path.basename(b))
print("published", [b for _, b in up])
