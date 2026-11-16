#!/usr/bin/env python3
"""Embed the new rows with the encoder of the existing table (BAAI/bge-small-en-v1.5), the same rendered text
(model.features.render_texts), under bf16 autocast on the CPU (encoder track: same vectors, 2.3x faster).

    python scripts/expand/embed_rows.py --rows data/expand/new_rows.parquet --out data/expand/emb [--check-table data/work/model/table]

Resumable: vectors are written in shards of 20,000 posts, sorted by text length so batches pad little.
--check-table re-embeds 2,000 rows of the existing table first and reports the cosine to the stored vectors.
"""
import argparse, logging, time
from pathlib import Path
import numpy as np, polars as pl, torch
from xviral.embed.text_small import SentenceEmbedder
from xviral.model import features

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True); ap.add_argument("--check-table", default=None); ap.add_argument("--batch", type=int, default=128)
args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s"); log = logging.getLogger("embed")
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
enc = SentenceEmbedder("BAAI/bge-small-en-v1.5", device="cpu")

def encode(texts):
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        return enc.encode(texts, batch_size=args.batch)

if args.check_table:
    d = Path(args.check_table)
    idx = np.sort(np.random.default_rng(0).choice(len(np.load(d / "post_id.npy")), 2000, replace=False))
    tx = pl.read_parquet(d / "texts.parquet")["text"].gather(idx).to_list()
    ref = np.asarray(np.load(d / "emb.npy", mmap_mode="r")[idx], np.float32)
    v = encode(tx)
    cos = (v * ref).sum(1) / (np.linalg.norm(v, axis=1) * np.linalg.norm(ref, axis=1) + 1e-9)
    msg = f"check against stored vectors of the existing table, 2000 rows: cosine mean {cos.mean():.5f}, min {cos.min():.5f}, share above 0.999: {(cos > 0.999).mean():.4f}"
    log.info(msg); (out / "check.txt").write_text(msg + "\n")

rows = pl.read_parquet(args.rows, columns=["post_id", "text", "text_note", "author_handle", "author_display_name", "quoted_text"])
texts = features.render_texts(rows)
ids = rows["post_id"].to_numpy()
order = np.argsort([len(t) for t in texts], kind="stable")
t0 = time.time(); done = 0
for k in range(0, len(order), 20_000):
    f = out / f"part-{k // 20_000:04d}.npz"
    sel = order[k:k + 20_000]
    if f.exists():
        continue
    v = encode([texts[i] for i in sel])
    np.savez(f, post_id=ids[sel], emb=v.astype(np.float16))
    done += len(sel)
    log.info("embedded %d/%d, %.0f posts/s this run", k + len(sel), len(order), done / (time.time() - t0))
print("EMBED_DONE")
