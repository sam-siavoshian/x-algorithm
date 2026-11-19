#!/usr/bin/env python3
"""The pairs track's pair table (serving-path rows of 13,518 controlled pairs) for models trained on the combine table:
the post embedding replaced by text-embedding-3-large (1,024 dimensions), semantic IDs reassigned, and the two source
inputs (src_regime, src_name) added as features.transform derives them at serving time. Everything else is linked.

    python scripts/combine/pair_table_te3l.py --count-only           # how many texts are not in the store, and their size
    python scripts/combine/pair_table_te3l.py --max-dollars 20.5     # absolute cap on the store's ledger

Texts are rendered from the corpus by post id (a draft row is the real post it stands for, taken from pairs.parquet).
--check N re-embeds N rendered texts with bge-small and compares them with the table's stored embedding.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "research/tracks/embed_api")
from openai_embed import MAX_CHARS, ApiEmbedder, key_of  # noqa: E402
from xviral.embed import sid as sidmod  # noqa: E402
from xviral.model import features  # noqa: E402
from xviral.schema import conform  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pair-table", default="data/tracks/pairs/pair_table"); ap.add_argument("--out", default="data/combine/pair_table_te3l")
ap.add_argument("--posts", default="data/work/filtered/posts.parquet"); ap.add_argument("--store", default="data/tracks/embed_api/store/te3large_3072")
ap.add_argument("--codebook", default="data/combine/serving_te3l/sid_codebook.npz"); ap.add_argument("--dim", type=int, default=1024)
ap.add_argument("--count-only", action="store_true"); ap.add_argument("--max-dollars", type=float, default=None); ap.add_argument("--check", type=int, default=0)
a = ap.parse_args()
src, out = Path(a.pair_table), Path(a.out)
ids, block = np.load(src / "post_id.npy"), np.load(src / "author_idx.npy")
P = pl.read_parquet(src / "pairs.parquet")
real = ids.copy()
real[P["row_a"].to_numpy()], real[P["row_b"].to_numpy()] = P["post_id"].to_numpy(), P["post_id_b"].to_numpy()
assert (real > 0).all(), int((real <= 0).sum())
COLS = ["post_id", "author_id", "author_handle", "author_display_name", "created_at_ms", "text", "kind", "media", "urls", "quoted_post_id", "quoted_text", "source"]
d = pl.scan_parquet(a.posts).select(COLS).filter(pl.col("post_id").is_in(pl.Series(np.unique(real)).implode())).collect()
d = d.with_columns(pl.Series("rendered", features.render_texts(conform(d))))
look = pl.DataFrame({"post_id": real}).join(d.select("post_id", "rendered", "source"), on="post_id", how="left", maintain_order="left")
assert look["rendered"].null_count() == 0, look["rendered"].null_count()
texts = look["rendered"].to_list()
if a.check:
    from xviral.embed.text_small import SentenceEmbedder
    rng = np.random.default_rng(0); pick = np.sort(rng.choice(len(texts), a.check, replace=False))
    E0 = np.asarray(np.load(src / "emb.npy", mmap_mode="r")[pick], dtype=np.float32)
    E1 = SentenceEmbedder("BAAI/bge-small-en-v1.5", device="cpu").encode([texts[i] for i in pick])
    cos = (E0 * E1).sum(1) / (np.linalg.norm(E0, axis=1) * np.linalg.norm(E1, axis=1) + 1e-9)
    print(json.dumps({"checked": int(a.check), "cosine_min": float(cos.min()), "cosine_median": float(np.median(cos)), "below_0.99": int((cos < 0.99).sum()),
                      "drafts_checked": int((ids[pick] <= 0).sum()), "drafts_below_0.99": int((cos[ids[pick] <= 0] < 0.99).sum())}), flush=True)
enc = ApiEmbedder("text-embedding-3-large", a.store, workers=8, max_dollars=a.max_dollars)
norm = [(t or " ")[:MAX_CHARS] for t in texts]
keys = [key_of(t) for t in norm]
miss = {k: t for k, t in zip(keys, norm) if k not in enc._pos}
print(json.dumps({"rows": len(texts), "distinct_texts": len(set(keys)), "not_in_store": len(miss), "chars_to_send": sum(len(t) for t in miss.values()),
                  "projected_dollars_at_0.25_tokens_per_char": sum(len(t) for t in miss.values()) * 0.25 * 0.13 / 1e6}), flush=True)
if a.count_only:
    sys.exit(0)
todo = list(miss.values())
for i in range(0, len(todo), 40_000):
    enc.encode(todo[i:i + 40_000])
out.mkdir(parents=True, exist_ok=True)
E = np.lib.format.open_memmap(out / "emb.npy", mode="w+", dtype=np.float16, shape=(len(texts), a.dim))
codes = []
for i in range(0, len(texts), 20_000):
    blk = np.stack([np.asarray(enc._vecs[s][r][:a.dim], dtype=np.float32) for s, r in (enc._pos[k] for k in keys[i:i + 20_000])])
    blk = (blk / np.maximum(np.linalg.norm(blk, axis=1, keepdims=True), 1e-9)).astype(np.float16)
    E[i:i + len(blk)] = blk
    codes.append(sidmod.phoenix_codes(sidmod.assign(blk.astype(np.float32), a.codebook)))
E.flush()
np.save(out / "sid.npy", np.concatenate(codes).astype(np.load(src / "sid.npy", mmap_mode="r").dtype))
# the two source inputs, as at serving: history rows by their source, a draft takes its pair's commonest regime and no source name
rg_names, sn_names = dict(features.CAT_SPEC)["src_regime"], dict(features.CAT_SPEC)["src_name"]
f = pl.DataFrame({"draft": ids <= 0, "block": block, "source": look["source"]}).with_columns(
    pl.when(pl.col("draft")).then(None).otherwise(pl.col("source").replace_strict(features.SRC_REGIME, default=None, return_dtype=pl.String)).alias("rg"),
    pl.when(pl.col("draft")).then(None).otherwise(pl.col("source").cast(pl.String).str.split("+").list.first()).alias("sn"))
f = f.with_columns(pl.coalesce(pl.col("rg"), pl.col("rg").drop_nulls().mode().first().over("block"), pl.lit("unknown")).alias("rg"))
rg = f["rg"].replace_strict({s: i for i, s in enumerate(rg_names)}, default=0, return_dtype=pl.Int32).to_numpy()
sn = f["sn"].replace_strict({s: i for i, s in enumerate(sn_names)}, default=0, return_dtype=pl.Int32).to_numpy()
cat0 = np.load(src / "cat.npy")
np.save(out / "cat.npy", np.concatenate([cat0, rg[:, None].astype(cat0.dtype), sn[:, None].astype(cat0.dtype)], axis=1))
meta = json.loads((src / "meta.json").read_text())
meta["cat_names"] = meta["cat_names"] + ["src_regime", "src_name"]; meta["cat_cards"] = meta["cat_cards"] + [len(rg_names), len(sn_names)]
meta["emb_dim"] = a.dim; meta["encoder"] = {"name": "openai:text-embedding-3-large@1024", "parts": [{"src": "te3l", "dim": 1024}]}
(out / "meta.json").write_text(json.dumps(meta, indent=2))
for p in src.iterdir():
    if p.name not in ("cat.npy", "meta.json", "emb.npy", "sid.npy") and not (out / p.name).exists():
        os.symlink(p.resolve(), out / p.name)
print("written", out, {"regime": {n: int(c) for n, c in zip(rg_names, np.bincount(rg, minlength=len(rg_names))) if c}, "source": {n: int(c) for n, c in zip(sn_names, np.bincount(sn, minlength=len(sn_names))) if c}},
      "ledger", json.dumps(enc.ledger), flush=True)
