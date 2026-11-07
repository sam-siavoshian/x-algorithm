"""Phase-1 orchestration with fake encoders: media fetch, video sampling, ASR, the X-faithful
embedding, semantic IDs through the repo's own codebook loader, SigLIP, and neighbours.
Checks the glue between stages; the models themselves need the GPU box."""

import io
import shutil
import subprocess
import time

import numpy as np
import polars as pl
import pytest

from tests.test_end_to_end import build_sources
from xviral import pipeline
from xviral.config import Config
from xviral.embed import mm, sid
from xviral.embed.store import EmbeddingStore


class FakeST:
    """Stands in for the SentenceTransformer that mm_encoder loads; returns a deterministic 4096-d vector."""

    def encode(self, inputs, prompt=None, batch_size=None):
        out = []
        for x in inputs:
            seed = abs(hash(str(x)[:200])) % (2**32)
            out.append(np.random.default_rng(seed).normal(size=4096).astype(np.float32))
        return np.stack(out)


class FakeSiglip:
    def __init__(self, *a, **k):
        pass

    def encode(self, images):
        return np.stack([np.random.default_rng(len(b)).normal(size=16).astype(np.float32) for b in images]) if images else np.zeros((0, 16))


class FakeResp:
    def __init__(self, content, status=200):
        self.content, self.status_code = content, status


class FakeSession:
    headers: dict = {}

    def get(self, url, timeout=None):
        from PIL import Image

        b = io.BytesIO()
        Image.new("RGB", (900, 600), (len(url) % 255, 10, 10)).save(b, format="JPEG")
        return FakeResp(b.getvalue())


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_phase1_glue(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    usc_root, ca, scrape = build_sources(tmp_path / "raw", rng)
    import os

    future = time.time() + 400 * 86400
    os.utime(scrape, (future, future))
    cfg = Config.model_validate({
        "paths": {"work": str(tmp_path / "work"), "usc_root": str(usc_root), "community_archive": str(ca), "user_scrape": [str(scrape)]},
        "relevance": {"embedder": "hashing", "negatives_sample": 5000},
        "media": {"max_images": 20, "max_videos": 2, "requests_per_second": 1000},
        "embed": {"device": "cpu", "sid_levels": 2, "sid_centroids": 4},
        "workers": 1,
    })
    pipeline.phase0(cfg)
    rows_path = cfg.paths.work / "dataset" / "rows.parquet"
    rows = pl.read_parquet(rows_path)
    # Give a few kept posts an image and a video so the media stages have work.
    ids = rows["post_id"].head(6).to_list()
    photo = {"type": "photo", "url": "https://pbs.twimg.com/media/A.jpg", "video_url": None, "width": 900, "height": 600, "duration_ms": None, "alt": None}
    vid = {"type": "video", "url": "https://pbs.twimg.com/t.jpg", "video_url": "https://video.twimg.com/v.mp4", "width": 320, "height": 568, "duration_ms": 12000, "alt": None}
    media = [[photo] if i in ids[:4] else [vid] if i in ids[4:] else m for i, m in zip(rows["post_id"].to_list(), rows["media"].to_list())]
    rows = rows.with_columns(pl.Series("media", media, dtype=rows.schema["media"]))
    rows.write_parquet(rows_path)

    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=12:size=320x568:rate=10", "-f", "lavfi",
                    "-i", "sine=frequency=300:duration=12", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(clip)], check=True)

    def fake_download(url, timeout=60.0, max_bytes=0):
        p = tmp_path / f"dl_{time.monotonic_ns()}.mp4"
        shutil.copy(clip, p)
        return p

    import xviral.media.asr as asr_mod
    import xviral.media.fetch as fetch_mod
    import xviral.media.video as video_mod

    monkeypatch.setattr(video_mod, "download", fake_download)
    monkeypatch.setattr(fetch_mod.requests, "Session", FakeSession)
    monkeypatch.setattr(asr_mod, "transcribe", lambda audio, name=None, device=None: asr_mod.Transcript("we launched today", "en", 0.5, True, 12.0))
    m = mm.mm()
    monkeypatch.setitem(m._ST_MODEL_CACHE, (m.QWEN_MODEL_ID, "cpu"), FakeST())
    import xviral.embed.siglip as sig_mod

    monkeypatch.setattr(sig_mod, "SiglipEncoder", FakeSiglip)

    def fake_train(npy, out, num_levels=6, num_centroids=256, iters=25, sample_size=None, seed=42):
        x = np.load(npy)
        cb = np.stack([x[np.random.default_rng(l).choice(len(x), num_centroids, replace=False)] for l in range(num_levels)])
        mm.reference_module("sid_io").save_codebook_npz(out, cb, iters, seed, len(x))
        return out

    monkeypatch.setattr(sid, "train", fake_train)

    pipeline.media_select(cfg)
    pipeline.fetch_images(cfg)
    pipeline.fetch_videos(cfg)
    pipeline.asr_stage(cfg)
    pipeline.embed_mm(cfg)
    pipeline.sid_train(cfg)
    pipeline.sid_assign(cfg)
    pipeline.embed_siglip(cfg)
    pipeline.neighbors_stage(cfg)

    w = cfg.paths.work
    imgs = pl.read_parquet(w / "media" / "images")
    assert imgs.height == 4 and (imgs["status"] == 200).all()
    from PIL import Image

    assert max(Image.open(io.BytesIO(imgs["jpeg"][0])).size) == cfg.media.vlm_side
    vids = pl.read_parquet(w / "media" / "videos")
    assert vids.height == 2 and (vids["status"] == "ok").all() and vids["embed_frames"].list.len().to_list() == [4, 4]
    tr = pl.read_parquet(w / "media" / "transcripts.parquet")
    assert tr.height == 2 and tr["transcript"][0] == "we launched today"
    ids_mm, vecs = EmbeddingStore(w / "emb" / "mm").all()
    assert len(ids_mm) == rows.height and np.allclose(np.linalg.norm(np.asarray(vecs, dtype=np.float32), axis=1), 1.0, atol=1e-2)
    s = pl.read_parquet(w / "emb" / "sid.parquet")
    assert s.height == rows.height and s["sid_0"].min() >= 1 and s["sid_0"].max() <= 4
    nb = pl.read_parquet(w / "emb" / "neighbors.parquet")
    assert nb.height == rows.height and "nbr_cos" in nb.columns and "self_sim_48h" in nb.columns
    vf = pl.read_parquet(w / "emb" / "video_frames.parquet")
    assert vf.height == 2 and (vf["frames_n"] > 0).all()
    keys = pl.read_parquet(w / "emb" / "siglip_images_keys.parquet")
    assert keys.height == 4 and keys["key"].n_unique() == 4
