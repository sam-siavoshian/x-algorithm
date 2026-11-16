#!/usr/bin/env python3
"""Contact sheets for checking labels by eye: python scripts/video/contact_sheet.py <labels.parquet with post_id,label> <outdir> [n per class] [seed]"""
import os, sys, polars as pl
from PIL import Image, ImageDraw
L = pl.read_parquet(sys.argv[1]); out = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 40; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
os.makedirs(out, exist_ok=True); C, S = 8, 200
for (lab,), g in L.group_by("label"):
    g = g.sample(min(n, g.height), seed=seed); rows = (g.height + C - 1) // C
    sheet = Image.new("RGB", (C * S, rows * (S + 14)), "white"); d = ImageDraw.Draw(sheet)
    for k, pid in enumerate(g["post_id"].to_list()):
        im = Image.open(f"data/video/thumbs/{pid}.jpg").convert("RGB"); im.thumbnail((S - 4, S - 4))
        x, y = (k % C) * S, (k // C) * (S + 14); sheet.paste(im, (x + 2, y + 14)); d.text((x + 2, y), f"{k}", fill="black")
    sheet.save(f"{out}/{lab}.jpg", quality=80); g.select("post_id").write_csv(f"{out}/{lab}.csv")
    print(lab, g.height, L.filter(pl.col("label") == lab).height)
