#!/usr/bin/env python3
"""Represent every fetched image on CPU, one output file per fetch shard (resumable):

    python scripts/images/represent.py --raw data/images/raw --out data/images/rep [--watch]

Per image: SigLIP base embedding (768, unit norm), CLIP ViT-B/32 embedding (512, unit norm) and the LAION linear
aesthetic score on it, the probability of "AI-generated" from an open detector (Ateeqq/ai-vs-human-image-detector, a
SigLIP fine-tune), and pixel attributes measured on the 680 px rendition: sharpness (variance of the Laplacian), edge
density, text boxes and the share of the image they cover (RapidOCR text detector, no recognition), faces (YuNet),
brightness, saturation, colourfulness, share of dark and of white pixels, three dominant colours.
"""
import argparse, glob, io, os, time, warnings
import cv2, numpy as np, polars as pl, torch
from PIL import Image
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True); ap.add_argument("--watch", action="store_true"); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--no-ocr", action="store_true"); ap.add_argument("--ai", action="store_true"); ap.add_argument("--ocr-shards", type=int, default=10**6, help="text detection only on shards below this index (time budget)"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.set_num_threads(a.threads); cv2.setNumThreads(a.threads)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoProcessor, AutoModelForImageClassification, CLIPVisionModelWithProjection
sp = AutoProcessor.from_pretrained("google/siglip-base-patch16-224"); sm = AutoModel.from_pretrained("google/siglip-base-patch16-224").eval()
cm = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").eval()
sd = load_file(hf_hub_download("shunk031/aesthetics-predictor-v1-vit-base-patch32", "model.safetensors")); AW, AB = sd["predictor.weight"][0], sd["predictor.bias"][0]
dm = AutoModelForImageClassification.from_pretrained("Ateeqq/ai-vs-human-image-detector").eval(); AI_IDX = [k for k, v in dm.config.id2label.items() if v == "ai"][0]
yn = cv2.FaceDetectorYN.create(hf_hub_download("opencv/face_detection_yunet", "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.7, 0.3, 50)
ocr = None
if not a.no_ocr:
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
CM, CS = np.array([0.48145466, 0.4578275, 0.40821073], np.float32), np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
def prep(ims, mean, std):
    x = np.stack([np.asarray(im.resize((224, 224), Image.Resampling.BICUBIC), np.float32) / 255 for im in ims]); return torch.from_numpy(((x - mean) / std).transpose(0, 3, 1, 2).copy())
def pixel(im, use_ocr=True):
    rgb = np.asarray(im); h, w = rgb.shape[:2]; g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F); ed = cv2.Canny(g, 100, 200)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); R, G, B = (rgb[..., i].astype(np.float32) for i in range(3)); rg, yb = R - G, 0.5 * (R + G) - B
    small = cv2.resize(rgb, (48, 48), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    _, lab, cen = cv2.kmeans(small, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 1, cv2.KMEANS_PP_CENTERS)
    sh = np.bincount(lab.ravel(), minlength=3) / len(small); o = np.argsort(-sh)
    r = {"fetched_w": w, "fetched_h": h, "sharpness": float(lap.var()), "edge_density": float((ed > 0).mean()), "brightness": float(g.mean() / 255), "contrast": float(g.std() / 255), "saturation": float(hsv[..., 1].mean() / 255),
         "colourfulness": float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)), "dark_share": float((g < 40).mean()), "white_share": float((g > 235).mean()),
         "dom_rgb": [int(v) for k in o for v in cen[k]], "dom_share": [float(sh[k]) for k in o]}
    yn.setInputSize((w, h)); _, f = yn.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    r["faces"] = 0 if f is None else int(len(f)); r["face_area_share"] = 0.0 if f is None else float(max(x[2] * x[3] for x in f) / (w * h))
    if ocr is not None and use_ocr:
        try:
            k = min(1.0, 512 / max(w, h)); bx, _ = ocr.text_detector(rgb if k == 1 else cv2.resize(rgb, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA))
            bx = None if bx is None else np.asarray(bx, np.float32) / k
        except Exception:  # noqa: BLE001
            bx = None
        if bx is not None and len(bx):
            ar = [cv2.contourArea(b) for b in bx]; hs = [float(np.linalg.norm(b[0] - b[3])) for b in bx]
            r.update(text_boxes=len(bx), text_area_share=float(sum(ar) / (w * h)), text_max_height_share=float(max(hs) / h))
        else: r.update(text_boxes=0, text_area_share=0.0, text_max_height_share=0.0)
    return r
def shard(f, out):
    d = pl.read_parquet(f).filter(pl.col("status") == 200)
    if a.limit: d = d.head(a.limit)
    ids, idx, ims, T = [], [], [], {}
    for pid, mi, b in zip(d["post_id"], d["media_idx"], d["jpeg"]):
        try: im = Image.open(io.BytesIO(b)).convert("RGB")
        except Exception: continue  # noqa: BLE001
        ids.append(pid); idx.append(mi); ims.append(im)
    t = time.time(); use_ocr = int(os.path.basename(f)[4:9]) < a.ocr_shards; rows = [pixel(im, use_ocr) for im in ims]; T["pixel"] = time.time() - t
    E, C, AE, AI = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(ims), 32):
            b = ims[i:i + 32]; t = time.time(); xs = prep(b, 0.5, 0.5)
            o = sm.get_image_features(pixel_values=xs); e = o if torch.is_tensor(o) else o.pooler_output; E.append(torch.nn.functional.normalize(e, dim=-1).numpy()); T["siglip"] = T.get("siglip", 0) + time.time() - t
            t = time.time(); c = torch.nn.functional.normalize(cm(pixel_values=prep(b, CM, CS)).image_embeds, dim=-1); C.append(c.numpy()); AE.append((c @ AW + AB).numpy()); T["clip"] = T.get("clip", 0) + time.time() - t
            t = time.time(); AI.append(torch.softmax(dm(pixel_values=xs).logits, -1)[:, AI_IDX].numpy() if a.ai else np.full(len(b), np.nan, np.float32)); T["ai"] = T.get("ai", 0) + time.time() - t
    A = pl.DataFrame(rows).with_columns(pl.Series("post_id", ids, dtype=pl.Int64), pl.Series("media_idx", idx, dtype=pl.Int64), pl.Series("aesthetic", np.concatenate(AE).astype(np.float32)), pl.Series("ai_detector_p", np.concatenate(AI).astype(np.float32)))
    np.savez(out + ".tmp.npz", post_id=np.array(ids, np.int64), siglip=np.concatenate(E).astype(np.float16), clip=np.concatenate(C).astype(np.float16)); A.write_parquet(out + ".parquet"); os.replace(out + ".tmp.npz", out + ".npz")
    print(os.path.basename(f), len(ims), {k: round(v, 1) for k, v in T.items()}, flush=True)
idle = 0
while True:
    fs = [f for f in sorted(glob.glob(f"{a.raw}/img-*.parquet")) if not os.path.exists(f"{a.out}/{os.path.basename(f)[:-8]}.npz")]
    for f in fs: shard(f, f"{a.out}/{os.path.basename(f)[:-8]}"); idle = 0
    if not a.watch or a.limit: break
    if not fs:
        idle += 1
        if idle > 20: break
        time.sleep(30)
print("REPRESENT_DONE", flush=True)
