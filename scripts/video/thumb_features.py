#!/usr/bin/env python3
"""Features of the poster frame (the first thing the feed shows for a video post), CPU only.

    python scripts/video/thumb_features.py          # reads data/video/thumbs/*.jpg, writes data/video/thumb_feats.parquet + thumb_emb.npy

  SigLIP-2 image embedding (xviral.embed.siglip.SiglipEncoder, base model on CPU) and zero-shot scores against the
  prompts in PROMPTS; pixel statistics (dark frame, flat background share, colourfulness, edge density);
  faces (OpenCV YuNet): count, largest face as a share of the frame; text (RapidOCR detector + recogniser):
  number of text boxes, share of frame area covered by text, the same for the lower third, largest text height.
Resumable: posts already in the output are skipped.
"""
import glob, io, os, sys, numpy as np, polars as pl
sys.path.insert(0, ".")
MODEL = os.environ.get("SIGLIP", "google/siglip2-base-patch16-256")
PROMPTS = {
    "ui": ["a screenshot of a software application interface", "a screenshot of a website or web app", "a screen recording of a code editor or terminal", "a screenshot of a chat interface with an AI assistant", "a screenshot of a dashboard with charts and tables", "a screenshot of a mobile app on a phone screen"],
    "face": ["a person talking to the camera", "a close-up of a person's face", "a selfie video of a person speaking", "a person being interviewed, sitting in a room"],
    "title": ["large text on a plain background, a title card", "a slide with a headline in big letters", "a text announcement on a solid color background"],
    "logo": ["a company logo centered on a plain background", "a brand logo animation frame"],
    "graphic": ["an abstract motion graphics animation", "a 3D rendered product animation", "a colorful gradient background with shapes", "a cartoon or illustrated animation"],
    "real": ["a photo of a real-world outdoor scene", "a robot or hardware device in a lab", "people on a stage at a conference", "a group of people in an office", "a video of a physical product on a table", "footage of a car, drone or machine"],
    "chart": ["a chart or graph with data", "a diagram with boxes and arrows", "a presentation slide with bullet points"],
    "game": ["a frame from a video game", "an AI-generated cinematic image", "a frame from a movie or TV show", "a meme or reaction image"],
    "blank": ["a completely black image", "a blank white image"],
}
def pix_stats(im):
    import cv2
    a = np.asarray(im.convert("RGB").resize((160, 160))).astype(np.float32); g = a.mean(2)
    q = (a // 32).astype(np.int32); code = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]; cnt = np.bincount(code.ravel(), minlength=512)
    rg = a[..., 0] - a[..., 1]; yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    e = cv2.Canny(g.astype(np.uint8), 80, 160)
    return {"mean_luma": float(g.mean()), "std_luma": float(g.std()), "flat_share": float(cnt.max() / cnt.sum()), "colorfulness": float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)),
            "edge_density": float((e > 0).mean()), "dark_share": float((g < 24).mean()), "light_share": float((g > 232).mean())}
def main():
    import cv2, torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    from rapidocr_onnxruntime import RapidOCR
    torch.set_num_threads(4)
    files = sorted(glob.glob("data/video/thumbs/*.jpg")); OUT = "data/video/thumb_feats.parquet"; EMB = "data/video/thumb_emb.npy"
    old = pl.read_parquet(OUT) if os.path.exists(OUT) else None; olde = np.load(EMB) if old is not None else None
    have = set(old["post_id"].to_list()) if old is not None else set()
    files = [f for f in files if int(os.path.basename(f)[:-4]) not in have]
    if len(sys.argv) > 1: files = files[:int(sys.argv[1])]
    print("to do", len(files), flush=True)
    proc = AutoProcessor.from_pretrained(MODEL); model = AutoModel.from_pretrained(MODEL).eval()
    keys = [(k, p) for k, ps in PROMPTS.items() for p in ps]
    with torch.no_grad():
        ti = proc(text=[p for _, p in keys], padding="max_length", max_length=64, return_tensors="pt"); tf = model.get_text_features(**ti)
        tf = getattr(tf, "pooler_output", tf); tf = tf / tf.norm(dim=-1, keepdim=True)
    ypath = "data/video/yunet.onnx"
    fd = cv2.FaceDetectorYN.create(ypath, "", (320, 320), 0.7, 0.3, 50) if os.path.exists(ypath) else None
    ocr = RapidOCR(); ocr.use_angle_cls = False                 # frames are at most 448 px: no upscaling to 736, no angle classifier
    for op in ocr.text_detector.preprocess_op:
        if hasattr(op, "limit_side_len"): op.limit_side_len, op.limit_type = 448, "max"
    rows, embs = [], []
    for b in range(0, len(files), 32):
        fs = files[b:b + 32]; ims = [Image.open(f).convert("RGB") for f in fs]
        with torch.no_grad():
            f = model.get_image_features(**proc(images=ims, return_tensors="pt")); f = getattr(f, "pooler_output", f); f = f / f.norm(dim=-1, keepdim=True)
            logit = (f @ tf.T) * model.logit_scale.exp() + model.logit_bias
        embs.append(f.numpy().astype(np.float16)); lg = logit.numpy()
        for j, (fp, im) in enumerate(zip(fs, ims)):
            r = {"post_id": int(os.path.basename(fp)[:-4]), "w": im.size[0], "h": im.size[1]} | pix_stats(im)
            for k in PROMPTS: r["z_" + k] = float(max(lg[j, i] for i, (kk, _) in enumerate(keys) if kk == k))
            bgr = np.asarray(im)[:, :, ::-1].copy(); H, W = bgr.shape[:2]
            if fd is not None:
                fd.setInputSize((W, H)); _, faces = fd.detect(bgr); faces = [] if faces is None else faces
                r["faces"] = len(faces); r["face_max_share"] = float(max([x[2] * x[3] for x in faces], default=0) / (W * H)); r["face_max_h"] = float(max([x[3] for x in faces], default=0) / H)
            try: res, _ = ocr(bgr)
            except Exception: res = None  # noqa: BLE001
            res = [x for x in (res or []) if float(x[2]) >= 0.6 and len(x[1].strip()) >= 2]
            area = low = 0.0; hmax = 0.0; words = 0; txt = []
            for box, t, _c in res:
                p = np.array(box, dtype=np.float32); bw = np.linalg.norm(p[1] - p[0]); bh = np.linalg.norm(p[3] - p[0]); cy = p[:, 1].mean()
                area += bw * bh; hmax = max(hmax, bh / H); words += len(t.split()); txt.append(t)
                if cy > H * 2 / 3: low += bw * bh
            r |= {"text_boxes": len(res), "text_words": words, "text_area": float(area / (W * H)), "text_area_lower": float(low / (W * H / 3)), "text_hmax": float(hmax), "text": " | ".join(txt)[:300]}
            rows.append(r)
        if (b // 32) % 10 == 0: print(b + len(fs), "/", len(files), flush=True)
    if not rows: return
    new = pl.DataFrame(rows, infer_schema_length=None); e = np.concatenate(embs)
    if old is not None: new = pl.concat([old, new], how="diagonal_relaxed"); e = np.concatenate([olde, e])
    new.write_parquet(OUT); np.save(EMB, e); print("FEATS_DONE", new.height, flush=True)
main()
