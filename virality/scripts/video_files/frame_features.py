#!/usr/bin/env python3
"""Whole-video visual features from the reduced previews (2 frames a second, 224 px high, first 180 s, no audio).

    python scripts/video_files/frame_features.py <shard> <of> [limit]     # run <of> copies, one per core
        reads  data/vf/preview_x/preview/<post_id>.mp4, data/vf/order.parquet (post_id in processing order)
        writes data/vf/frames/part_<shard>_<k>.parquet   one row per video (scalars + per-frame lists)
               data/vf/frames/emb_<shard>_<k>.npy        pooled SigLIP-2 embedding per video, row-aligned

Per video, on every frame (2 a second):
  d[t]      mean absolute difference of consecutive 64x36 grey frames (0..255)
  hd[t]     L1 distance of consecutive 8x4x4 HSV histograms (0..2)
  face[t]   YuNet: number of faces, height of the largest as a share of the frame height
On SAMPLED frames (the frame at 2.5 s, plus the mid-points of 5 equal parts of the preview; the frame at 0 s is the
poster frame the video track already embedded and scored, media/videos/thumb_features.parquet):
  SigLIP-2 base image embedding and zero-shot scores against the prompt groups of scripts/video/thumb_features.py;
  on the 5 body frames, text detection and recognition (RapidOCR) of the lower 40% of the frame, upscaled 2x.
SigLIP costs 0.44 s a frame on one core, which is what limits the sample to 6 frames a video.
Nothing here labels a video; scripts/video_files/label_videos.py turns these into classes and attributes, so every
threshold can be changed without decoding again.
"""
import os, sys
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
import subprocess, json, numpy as np, polars as pl
sys.path.insert(0, "scripts/video")
SHARD, OF = int(sys.argv[1]), int(sys.argv[2]); LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None
MODEL = "google/siglip2-base-patch16-256"; SRC = "data/vf/preview_x/preview"; OUT = "data/vf/frames"; CHUNK = 100
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
G = list(PROMPTS)


def decode(path):
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", path], capture_output=True, text=True)
    st = json.loads(pr.stdout or "{}").get("streams") or [{}]
    w, h = int(st[0].get("width") or 0), int(st[0].get("height") or 0)
    if not w or not h: return None
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3) if n else None


def sample_idx(n):
    first = [min(5, n - 1)]
    body = sorted({min(n - 1, int((k + 0.5) * n / 5)) for k in range(5)})
    return first, body


def main():
    import cv2, torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    from rapidocr_onnxruntime import RapidOCR
    torch.set_num_threads(1); cv2.setNumThreads(1)
    os.makedirs(OUT, exist_ok=True)
    ids = pl.read_parquet("data/vf/order.parquet")["post_id"].to_list()
    ids = [p for i, p in enumerate(ids) if i % OF == SHARD]
    done = set()
    for f in os.listdir(OUT):
        if f.startswith(f"part_{SHARD}_"): done |= set(pl.read_parquet(os.path.join(OUT, f), columns=["post_id"])["post_id"].to_list())
    ids = [p for p in ids if p not in done][:LIMIT]
    k0 = len([f for f in os.listdir(OUT) if f.startswith(f"part_{SHARD}_")])
    print("to do", len(ids), flush=True)
    proc = AutoProcessor.from_pretrained(MODEL); model = AutoModel.from_pretrained(MODEL).eval()
    keys = [(g, p) for g, ps in PROMPTS.items() for p in ps]
    with torch.no_grad():
        tf = model.get_text_features(**proc(text=[p for _, p in keys], padding="max_length", max_length=64, return_tensors="pt"))
        tf = getattr(tf, "pooler_output", tf); tf = tf / tf.norm(dim=-1, keepdim=True)
    gi = [[i for i, (g, _) in enumerate(keys) if g == gg] for gg in G]
    fd = cv2.FaceDetectorYN.create("data/vf/models/yunet.onnx", "", (320, 320), 0.7, 0.3, 50)
    ocr = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1); ocr.use_angle_cls = False
    for op in ocr.text_detector.preprocess_op:
        if hasattr(op, "limit_side_len"): op.limit_side_len, op.limit_type = 960, "max"
    rows, embs = [], []

    def flush():
        nonlocal rows, embs, k0
        if not rows: return
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(f"{OUT}/part_{SHARD}_{k0}.parquet"); np.save(f"{OUT}/emb_{SHARD}_{k0}.npy", np.stack(embs)); k0 += 1; rows, embs = [], []

    for n_done, pid in enumerate(ids):
        fr = decode(f"{SRC}/{pid}.mp4")
        if fr is None:
            rows.append({"post_id": pid, "n_frames": 0}); embs.append(np.zeros(768, np.float16)); continue
        n, H, W = fr.shape[:3]
        small = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (64, 36), interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32)
        d = np.abs(np.diff(small, axis=0)).mean((1, 2)) if n > 1 else np.zeros(0, np.float32)
        hs = []
        for f in fr:
            hsv = cv2.cvtColor(cv2.resize(f, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2HSV)
            hh = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel(); hs.append(hh / hh.sum())
        hs = np.stack(hs); hd = np.abs(np.diff(hs, axis=0)).sum(1) if n > 1 else np.zeros(0, np.float32)
        luma = small.mean((1, 2))
        fd.setInputSize((W, H)); fcount, fh = [], []
        for f in fr:
            _, faces = fd.detect(np.ascontiguousarray(f[:, :, ::-1])); faces = [] if faces is None else faces
            fcount.append(len(faces)); fh.append(float(max([x[3] for x in faces], default=0) / H))
        first, body = sample_idx(n); samp = sorted(set(first) | set(body))
        with torch.no_grad():
            e = model.get_image_features(**proc(images=[Image.fromarray(fr[i]) for i in samp], return_tensors="pt")); e = getattr(e, "pooler_output", e); e = e / e.norm(dim=-1, keepdim=True)
            lg = ((e @ tf.T) * model.logit_scale.exp() + model.logit_bias).numpy()
        z = np.stack([lg[:, ix].max(1) for ix in gi], 1)          # sampled frame x group
        e = e.numpy(); bi = [samp.index(i) for i in body]
        pooled = e[bi].mean(0); pooled /= np.linalg.norm(pooled) + 1e-9
        low = []                                                   # lower-40% text per body frame
        y0 = int(H * 0.6)
        for i in body:
            crop = cv2.resize(np.ascontiguousarray(fr[i][y0:, :, ::-1]), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            try: res, _ = ocr(crop)
            except Exception: res = None  # noqa: BLE001
            items = []
            for box, t, c in (res or []):
                if float(c) < 0.5 or len(t.strip()) < 2: continue
                p = np.array(box, dtype=np.float32) / 2.0
                items.append({"cx": float(p[:, 0].mean() / W), "cy": float((p[:, 1].mean() + y0) / H), "w": float((p[:, 0].max() - p[:, 0].min()) / W), "h": float((p[:, 1].max() - p[:, 1].min()) / H), "t": t.strip()[:80], "c": round(float(c), 2)})
            low.append(items)
        rows.append({"post_id": pid, "n_frames": n, "w": W, "h": H, "d": d.round(2).tolist(), "hd": hd.round(3).tolist(), "luma": luma.round(1).tolist(), "faces": fcount, "face_h": [round(x, 3) for x in fh],
                     "samp_idx": samp, "first_idx": first, "body_idx": body, "z": z.round(2).ravel().tolist(), "lower_text": json.dumps(low)})
        embs.append(pooled.astype(np.float16))
        if len(rows) >= CHUNK: flush(); print(n_done + 1, "/", len(ids), flush=True)
    flush(); print("FRAMES_DONE", SHARD, flush=True)


main()
