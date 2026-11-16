#!/usr/bin/env python3
""""AI-generated" scores for the image-like classes, and sheets for labelling a validation set by eye.

    python scripts/images/ai_check.py --detect          # open detector on every photo / illustration image -> data/images/ai_detector.parquet
    python scripts/images/ai_check.py --sheet N --seed S # a random sheet of N photo / illustration images, with both scores in the key

The detector is Ateeqq/ai-vs-human-image-detector (a SigLIP fine-tune trained on AI art against human photos and art);
the zero-shot score is SigLIP base, AI prompts against real-photo / designed prompts (scripts/images/prompts.py).
"""
import argparse, glob, io, os
import numpy as np, polars as pl, torch
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser(); ap.add_argument("--detect", action="store_true"); ap.add_argument("--sheet", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--classes", default="illustration_render,people_photo,other_photo,meme"); ap.add_argument("--name", default="ai"); ap.add_argument("--only-sheets", action="store_true", help="detect only images on existing sheets")
a = ap.parse_args()
L = pl.read_parquet("data/images/labels.parquet").filter(pl.col("itype_raw").is_in(a.classes.split(",")))
raw = {}
for f in sorted(glob.glob("data/images/raw/img-*.parquet")):
    d = pl.read_parquet(f, columns=["post_id", "jpeg", "status"]).filter((pl.col("status") == 200) & pl.col("post_id").is_in(L["post_id"].implode()))
    raw.update(zip(d["post_id"].to_list(), d["jpeg"].to_list()))
L = L.filter(pl.col("post_id").is_in(list(raw)))
if a.detect:
    torch.set_num_threads(4)
    from transformers import AutoModelForImageClassification
    dm = AutoModelForImageClassification.from_pretrained("Ateeqq/ai-vs-human-image-detector").eval(); k = [i for i, v in dm.config.id2label.items() if v == "ai"][0]
    done = pl.read_parquet("data/images/ai_detector.parquet") if os.path.exists("data/images/ai_detector.parquet") else pl.DataFrame(schema={"post_id": pl.Int64, "ai_detector_p": pl.Float32})
    want = L["post_id"].to_list()
    if a.only_sheets:
        import glob as _g
        ss = set(); [ss.update(pl.read_csv(f)["post_id"].to_list()) for f in _g.glob("data/images/sheets/*.csv")]; want = [p for p in want if p in ss]
    seen = set(done["post_id"].to_list()); todo = [p for p in want if p not in seen]
    out = []
    for i in range(0, len(todo), 32):
        ims = [Image.open(io.BytesIO(raw[p])).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC) for p in todo[i:i + 32]]
        x = torch.from_numpy(((np.stack([np.asarray(im, np.float32) / 255 for im in ims]) - 0.5) / 0.5).transpose(0, 3, 1, 2).copy())
        with torch.no_grad(): out += torch.softmax(dm(pixel_values=x).logits, -1)[:, k].tolist()
    done = pl.concat([done, pl.DataFrame({"post_id": todo, "ai_detector_p": out}, schema={"post_id": pl.Int64, "ai_detector_p": pl.Float32})])
    done.write_parquet("data/images/ai_detector.parquet"); print("detector scores:", done.height)
if a.sheet:
    D = pl.read_parquet("data/images/ai_detector.parquet") if os.path.exists("data/images/ai_detector.parquet") else None
    s = L.sample(n=min(a.sheet, L.height), seed=a.seed, shuffle=True)
    if D is not None: s = s.join(D, on="post_id", how="left")
    cols, th = 6, 220; rows = (s.height + cols - 1) // cols
    sheet = Image.new("RGB", (cols * th, rows * (th + 18)), "white"); dr = ImageDraw.Draw(sheet)
    for j, pid in enumerate(s["post_id"].to_list()):
        im = Image.open(io.BytesIO(raw[pid])).convert("RGB"); im.thumbnail((th, th)); x, y = (j % cols) * th, (j // cols) * (th + 18)
        sheet.paste(im, (x + (th - im.width) // 2, y + 18 + (th - im.height) // 2)); dr.text((x + 3, y + 3), f"{j}", fill="red")
    os.makedirs("data/images/sheets", exist_ok=True)
    sheet.save(f"data/images/sheets/{a.name}_{a.seed}.jpg", quality=88)
    s.drop([c for c in s.columns if c.startswith("s_")]).with_row_index("k").write_csv(f"data/images/sheets/{a.name}_{a.seed}.csv"); print("sheet written", s.height)
