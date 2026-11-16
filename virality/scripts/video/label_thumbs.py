#!/usr/bin/env python3
"""First-frame class and attributes of each poster frame, from data/video/thumb_feats.parquet (thumb_features.py).

    python scripts/video/label_thumbs.py      # writes data/video/labels.parquet

Class rule (fixed after reading contact sheets of the first 900 frames, before any effect was computed):
  the prompt group with the highest SigLIP-2 score, with logo merged into title, game/film/meme/generated imagery
  merged into graphic, 'blank' into other; a frame scored graphic with a detected face at least 20% of the frame height,
  or scored real with a detected face at least 10% of the frame height, becomes face; a frame scored face with no
  detected face and a ui score within 1.0 of the face score becomes ui;
  empty  (class other, attribute empty_first) a frame with no readable text that is near black (mean luma under 16,
  luma spread under 12) or one flat colour (90% of pixels in one colour bin): a fade-in or blank opening frame.
"""
import json, numpy as np, polars as pl
F = pl.read_parquet("data/video/thumb_feats.parquet").unique("post_id", keep="last")
G = ["ui", "face", "title", "logo", "graphic", "real", "chart", "game", "blank"]
z = F.select(["z_" + g for g in G]).to_numpy(); top = np.array(G)[z.argmax(1)]
MERGE = {"logo": "title", "game": "graphic", "blank": "other"}
cls = np.array([MERGE.get(t, t) for t in top], dtype=object)
fh = F["face_max_h"].fill_null(0).to_numpy(); nf = F["faces"].fill_null(0).to_numpy(); luma = F["mean_luma"].to_numpy()
cls[(cls == "graphic") & (fh >= 0.20)] = "face"; cls[(cls == "real") & (fh >= 0.10)] = "face"
cls[(cls == "face") & (nf == 0) & (z[:, 0] >= z[:, 1] - 1.0)] = "ui"
tb = F["text_boxes"].to_numpy(); empty = (tb == 0) & (((luma < 16) & (F["std_luma"].to_numpy() < 12)) | (F["flat_share"].to_numpy() >= 0.90))
cls[empty] = "other"
T = pl.concat([pl.read_parquet("data/video/targets.parquet"), pl.read_parquet("data/video/targets2.parquet")], how="diagonal_relaxed").unique("post_id", keep="first").select("post_id", "mtype", "thumb_url", "meta_w", "meta_h", "n_videos")
L = F.with_columns(pl.Series("empty_first", empty.tolist()), pl.Series("cls", cls.tolist()), pl.Series("top_prompt_group", top.tolist())).join(T, on="post_id", how="left")
ar = pl.col("w") / pl.col("h"); short = pl.min_horizontal("meta_w", "meta_h")
L = L.with_columns(
    (pl.col("mean_luma") < 40).alias("dark_first"), (pl.col("faces").fill_null(0) > 0).alias("has_face"), (pl.col("face_max_h").fill_null(0) >= 0.25).alias("big_face"),
    (pl.col("text_boxes") >= 2).alias("text_on_frame"), (pl.col("text_area") >= 0.08).alias("text_heavy"), (pl.col("text_area") < 0.02).alias("text_light"), (pl.col("text_hmax") >= 0.08).alias("big_headline"),
    (pl.col("text_area_lower") >= 0.02).alias("lower_text"),
    (ar < 0.8).alias("portrait"), ((ar >= 0.8) & (ar <= 1.2)).alias("square"), (ar > 1.2).alias("landscape"), ((ar >= 1.7) & (ar <= 1.85)).alias("wide_16_9"),
    (short >= 1080).fill_null(False).alias("hd1080"), (short < 720).fill_null(False).alias("sub720"),
    pl.when(pl.col("thumb_url").str.contains("amplify_video_thumb")).then(pl.lit("amplify")).when(pl.col("thumb_url").str.contains("ext_tw_video_thumb")).then(pl.lit("ext")).when(pl.col("thumb_url").str.contains("tweet_video_thumb")).then(pl.lit("gif")).otherwise(pl.lit("other")).alias("route"))
L.write_parquet("data/video/labels.parquet")
print(L.height, L.group_by("cls").len().sort("len", descending=True).rows())
print({k: round(float(L[k].mean()), 3) for k in ["empty_first", "dark_first", "has_face", "big_face", "text_on_frame", "text_heavy", "big_headline", "lower_text", "portrait", "square", "wide_16_9", "hd1080", "sub720"]}, L.group_by("route").len().rows())
