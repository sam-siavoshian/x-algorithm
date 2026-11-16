#!/usr/bin/env python3
"""Zero-shot type labels from SigLIP base embeddings, plus contact sheets for checking them by eye.

    python scripts/images/label.py                      # writes data/images/labels.parquet
    python scripts/images/label.py --sheets 30 --seed 0 # also writes data/images/sheets/<class>_<seed>.jpg (+ .csv key)

Per image: the score of every class (SigLIP sigmoid logit of the mean prompt embedding), the argmax class, its margin
over the runner-up, and an "AI-generated" zero-shot score (ai prompts minus real prompts).
"""
import argparse, glob, io, os, sys
import numpy as np, polars as pl, torch
sys.path.insert(0, os.path.dirname(__file__))
from prompts import CLASSES, AI_PROMPTS, GROUP
ap = argparse.ArgumentParser(); ap.add_argument("--rep", default="data/images/rep"); ap.add_argument("--raw", default="data/images/raw"); ap.add_argument("--out", default="data/images/labels.parquet")
ap.add_argument("--sheets", type=int, default=0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--min-margin", type=float, default=0.0)
ap.add_argument("--sheet-filter", default="")
a = ap.parse_args()
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224"); m = AutoModel.from_pretrained("google/siglip-base-patch16-224").eval()
def txt(ps):
    with torch.no_grad():
        o = m.get_text_features(**tok(ps, padding="max_length", max_length=64, return_tensors="pt", truncation=True))
        e = o if torch.is_tensor(o) else o.pooler_output
    e = torch.nn.functional.normalize(e, dim=-1).mean(0); return torch.nn.functional.normalize(e, dim=0).numpy()
names = list(CLASSES); W = np.stack([txt(CLASSES[c]) for c in names]); AIW = np.stack([txt(AI_PROMPTS["ai"]), txt(AI_PROMPTS["real"])])
scale, bias = float(m.logit_scale.exp()), float(m.logit_bias)
ids, idx, E = [], [], []
for f in sorted(glob.glob(f"{a.rep}/img-*.npz")):
    z = np.load(f); ids.append(z["post_id"]); E.append(z["siglip"].astype(np.float32))
    A = pl.read_parquet(f[:-4] + ".parquet", columns=["post_id", "media_idx"]); idx.append(A["media_idx"].to_numpy())
ids, idx, E = np.concatenate(ids), np.concatenate(idx), np.concatenate(E)
S = E @ W.T * scale + bias
o = np.argsort(-S, 1); top, second = o[:, 0], o[:, 1]
ai = (E @ AIW.T * scale)
L = pl.DataFrame({"post_id": ids, "media_idx": idx, "itype_raw": np.array(names)[top], "itype": [GROUP.get(x, x) for x in np.array(names)[top]], "itype2": np.array(names)[second], "margin": S[np.arange(len(S)), top] - S[np.arange(len(S)), second],
                  "ai_zeroshot": ai[:, 0] - ai[:, 1], **{f"s_{c}": S[:, j].astype(np.float32) for j, c in enumerate(names)}})
L.write_parquet(a.out)
print(L.group_by("itype_raw").agg(pl.len(), pl.col("margin").median()).sort("len", descending=True))
if a.sheets:
    from PIL import Image, ImageDraw
    os.makedirs("data/images/sheets", exist_ok=True)
    raw = {}
    for f in sorted(glob.glob(f"{a.raw}/img-*.parquet")):
        d = pl.read_parquet(f, columns=["post_id", "jpeg", "status"]).filter(pl.col("status") == 200)
        raw.update(zip(d["post_id"].to_list(), d["jpeg"].to_list()))
    Q = L.filter(pl.col("margin") >= a.min_margin)
    if a.sheet_filter: Q = Q.filter(pl.sql_expr(a.sheet_filter))
    for c in names:
        s = Q.filter((pl.col("itype_raw") == c) & pl.col("post_id").is_in(list(raw))).sample(n=min(a.sheets, Q.filter(pl.col("itype_raw") == c).height), seed=a.seed, shuffle=True) if Q.filter(pl.col("itype_raw") == c).height else None
        if s is None or s.height == 0: continue
        cols, th = 6, 200; rows = (s.height + cols - 1) // cols
        sheet = Image.new("RGB", (cols * th, rows * (th + 18)), "white"); dr = ImageDraw.Draw(sheet)
        for k, pid in enumerate(s["post_id"].to_list()):
            im = Image.open(io.BytesIO(raw[pid])).convert("RGB"); im.thumbnail((th, th))
            x, y = (k % cols) * th, (k // cols) * (th + 18); sheet.paste(im, (x + (th - im.width) // 2, y + 18 + (th - im.height) // 2)); dr.text((x + 3, y + 3), f"{k}", fill="red")
        sheet.save(f"data/images/sheets/{c}_{a.seed}.jpg", quality=85); s.select("post_id", "itype_raw", "itype2", "margin", "ai_zeroshot").with_row_index("k").write_csv(f"data/images/sheets/{c}_{a.seed}.csv")
    print("sheets written")
