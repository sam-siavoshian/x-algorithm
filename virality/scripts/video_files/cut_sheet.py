#!/usr/bin/env python3
"""Contact sheets to set and check the cut detector by eye: the first 48 frames (24 s) of a video in a grid, each with
its difference to the previous frame (d = grey difference, hd = colour-histogram distance); detected cuts framed red.

    python scripts/video_files/cut_sheet.py <seed> <n videos>      # writes data/vf/sheets/cuts_<seed>_<i>.jpg
"""
import sys, glob, subprocess, json, numpy as np, polars as pl
from PIL import Image, ImageDraw
sys.path.insert(0, "scripts/video_files")
from label_videos import is_cut
seed, n = int(sys.argv[1]), int(sys.argv[2])
F = pl.concat([pl.read_parquet(f, columns=["post_id", "n_frames", "w", "h", "d", "hd"]) for f in glob.glob("data/vf/frames/part_*.parquet")], how="diagonal_relaxed").filter(pl.col("n_frames") >= 12).sample(n, seed=seed)
for k, r in enumerate(F.iter_rows(named=True)):
    w, h = r["w"], r["h"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"data/vf/preview_x/preview/{r['post_id']}.mp4", "-frames:v", "48", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3); cut = is_cut(np.array(r["d"]), np.array(r["hd"]))
    tw, th = int(w * 112 / h), 112; cols = 8; rows = (len(fr) + cols - 1) // cols
    S = Image.new("RGB", (cols * (tw + 4), rows * (th + 16)), "white"); D = ImageDraw.Draw(S)
    for i, f in enumerate(fr):
        x, y = (i % cols) * (tw + 4), (i // cols) * (th + 16)
        S.paste(Image.fromarray(f).resize((tw, th)), (x + 2, y + 14))
        c = i > 0 and bool(cut[i - 1])
        D.text((x + 2, y + 1), f"{i/2:.1f}s d{r['d'][i-1]:.0f} h{r['hd'][i-1]:.2f}" if i else "0s", fill="red" if c else "black")
        if c: D.rectangle([x, y + 12, x + tw + 3, y + th + 15], outline="red", width=2)
    S.save(f"data/vf/sheets/cuts_{seed}_{k}.jpg", quality=80); print(k, r["post_id"], int(cut[:47].sum()), "cuts in sheet")
