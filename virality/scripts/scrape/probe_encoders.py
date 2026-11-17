#!/usr/bin/env python3
"""What can be established about the three undocumented embedding files: shapes, norms, whether rows without a picture
description share one vector, and whether any common open 768-d text encoder reproduces the stored text vectors
(all inference is local; nothing is sent anywhere)."""
import json, sys
import numpy as np, polars as pl
D = "data/scrape/converted/"
raw = pl.read_parquet(D + "posts.parquet", columns=["id_str", "text", "post_image_description", "has_media"])
out = {}
E = {k: (np.load(D + k + ".npy", mmap_mode="r"), np.load(D + k + "_has.npy")) for k in ("text_emb", "img_emb", "imgdesc_emb")}
for k, (e, h) in E.items():
    x = np.asarray(e[:30000], np.float32)
    out[k] = {"shape": list(e.shape), "dtype": str(e.dtype), "rows_with_vector": int(h.sum()), "compact": bool(e.shape[0] == h.sum()),
              "norm_mean": float(np.linalg.norm(x, axis=1).mean()), "mean_abs_component": float(np.abs(x).mean()), "mean_cosine_between_random_rows": float((x[:2000] @ x[2000:4000].T).mean())}
# description vectors: rows whose description is the string "NaN"
e, h = E["imgdesc_emb"]
pos = np.cumsum(h) - 1
desc = raw["post_image_description"].to_list()
nan_rows = [i for i in range(len(desc)) if h[i] and (desc[i] is None or desc[i].strip().lower() == "nan")][:3000]
real_rows = [i for i in range(len(desc)) if h[i] and desc[i] and desc[i].strip().lower() != "nan"][:3000]
xn = np.asarray(e[pos[nan_rows]], np.float32); xr = np.asarray(e[pos[real_rows]], np.float32)
out["imgdesc_emb"]["rows_without_a_description_all_share_one_vector"] = bool(np.abs(xn - xn[0]).max() < 1e-3)
out["imgdesc_emb"]["distinct_vectors_among_3000_rows_without_description"] = int(len(np.unique(xn.round(3), axis=0)))
out["imgdesc_emb"]["rows_with_real_description_and_vector"] = int(sum(1 for i in range(len(desc)) if h[i] and desc[i] and desc[i].strip().lower() != "nan"))
te = E["text_emb"][0]
tn = [i for i, t in enumerate(raw["text"].to_list()) if t and t.strip().lower() == "nan"][:5]
out["text_and_description_same_space"] = {"cosine_text_vs_own_description_mean": float((np.asarray(te[real_rows], np.float32) * xr).sum(1).mean()),
                                          "cosine_text_vs_other_description_mean": float((np.asarray(te[real_rows], np.float32) * np.roll(xr, 1, 0)).sum(1).mean())}
ie, ih = E["img_emb"]; ipos = np.cumsum(ih) - 1
both = [i for i in real_rows if ih[i]][:2000]
out["img_emb"]["has_vector_but_no_media_flag"] = int((ih & ~raw["has_media"].to_numpy()).sum())
out["img_emb"]["media_posts_without_vector"] = int((~ih & raw["has_media"].to_numpy()).sum())
# candidate open text encoders
if len(sys.argv) > 1:
    from sentence_transformers import SentenceTransformer
    idx = real_rows[:40]
    texts = [raw["text"][i] for i in idx]; descs = [desc[i] for i in idx]
    ref_t = np.asarray(te[idx], np.float32); ref_d = np.asarray(e[pos[idx]], np.float32)
    res = {}
    for name, kw in [("sentence-transformers/all-mpnet-base-v2", {}), ("BAAI/bge-base-en-v1.5", {}), ("intfloat/e5-base-v2", {"prefix": "passage: "}), ("thenlper/gte-base", {}),
                     ("nomic-ai/nomic-embed-text-v1.5", {"prefix": "search_document: ", "trust": True}), ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", {}),
                     ("sentence-transformers/multi-qa-mpnet-base-dot-v1", {}), ("sentence-transformers/all-distilroberta-v1", {})]:
        try:
            m = SentenceTransformer(name, device="cpu", trust_remote_code=kw.get("trust", False))
            best = {}
            for pf in {"", kw.get("prefix", "")}:
                v = m.encode([pf + t for t in texts], normalize_embeddings=True); w = m.encode([pf + t for t in descs], normalize_embeddings=True)
                best[pf or "none"] = {"text": float((v * ref_t).sum(1).mean()), "description": float((w * ref_d).sum(1).mean())}
            res[name] = best
        except Exception as ex:
            res[name] = {"error": str(ex)[:200]}
        print(name, res[name], flush=True)
    out["candidate_text_encoders_mean_cosine_to_stored"] = res
json.dump(out, open("data/scrape_track/encoder_probe.json", "w"), indent=1)
print(json.dumps(out, indent=1))
