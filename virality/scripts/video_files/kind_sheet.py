#!/usr/bin/env python3
"""Contact sheets to validate a label by eye: 40 random videos with the label, each as its 5 body frames in a row.

    python scripts/video_files/kind_sheet.py <column>=<value> [seed] [caption]    # e.g. kind=screen, captions=true
        writes data/vf/sheets/val_<column>_<value>.jpg and .csv (row number, post_id)
With `caption` as third argument each video is shown as the lower 45% of its 5 body frames (for the caption check).
"""
import sys, subprocess, numpy as np, polars as pl
from PIL import Image, ImageDraw
col, val = sys.argv[1].split("="); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3; cap = len(sys.argv) > 3
V = pl.read_parquet("data/vf/video_features.parquet"); T = pl.read_parquet("data/launch/table.parquet", columns=["post_id", "lift_likes", "source"]).filter(pl.col("lift_likes").is_not_null())
V = V.join(T, on="post_id", how="semi").filter(pl.col(col).cast(pl.String).str.to_lowercase() == val.lower())
S = V.sample(min(40, V.height), seed=seed).sort("post_id")
TH, TW = (60, 210) if cap else (84, 150); cols = 2; RW = 5 * (TW + 2) + 34
sheet = Image.new("RGB", (cols * RW, ((S.height + 1) // 2) * (TH + 4)), "white"); D = ImageDraw.Draw(sheet)
for k, r in enumerate(S.iter_rows(named=True)):
    n = r["n_frames"]; idx = sorted({min(n - 1, int((j + 0.5) * n / 5)) for j in range(5)})
    x0, y0 = (k % cols) * RW, (k // cols) * (TH + 4); D.text((x0 + 2, y0 + 2), str(k + 1), fill="black")
    for j, i in enumerate(idx):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"data/vf/preview_x/preview/{r['post_id']}.mp4", "-vf", f"select=eq(n\\,{i})", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"], capture_output=True).stdout
        import io
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        if cap: im = im.crop((0, int(im.height * 0.55), im.width, im.height))
        im.thumbnail((TW, TH)); sheet.paste(im, (x0 + 30 + j * (TW + 2), y0 + 2))
name = f"data/vf/sheets/val_{col}_{val}"
sheet.save(name + ".jpg", quality=85); S.select("post_id", "kind", "body_classes", "cuts_per_min", "captions", "caption_sample").with_row_index("row", 1).write_csv(name + ".csv"); print(name, S.height, "of", V.height)
