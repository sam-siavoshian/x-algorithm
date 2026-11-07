"""Video probing and frame sampling with ffmpeg (ARCHITECTURE.md S0).

Three frame sets per video:
  embed     `embed_frames` uniformly spaced frames, for the X-faithful post
            embedding (X's v8 embedding plan uses up to 4 frames per video)
  temporal  up to `temporal_frames` frames at ~1 fps (uniform over the whole
            video when it is longer), for SigLIP temporal features
  thumb     the first frame: what the feed shows before playback
Plus the audio track as 16 kHz mono FLAC (<= max seconds) for transcription,
matching grox's ASR input (asr_processor.py: 16 kHz mono, <= 600 s).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class VideoProbe:
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool

    @property
    def resolution_class(self) -> str:
        """X's embedding plan renders 'quality SD/HD/Full HD'."""
        short = min(self.width, self.height)
        return "full_hd" if short >= 1080 else "hd" if short >= 720 else "sd"


@dataclass
class VideoSample:
    probe: VideoProbe
    embed: list[bytes] = field(default_factory=list)
    temporal: list[tuple[float, bytes]] = field(default_factory=list)
    thumb: bytes | None = None
    audio_flac: bytes | None = None


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be on PATH")


def probe(path: str | Path) -> VideoProbe:
    _require_ffmpeg()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    a = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    num, _, den = (v.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 1) else 0.0
    dur = float(info.get("format", {}).get("duration") or v.get("duration") or 0.0)
    return VideoProbe(duration_s=dur, width=int(v.get("width", 0)), height=int(v.get("height", 0)), fps=fps, has_audio=a)


def _frame_at(path: Path, t: float, side: int, q: int = 3) -> bytes | None:
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(path), "-frames:v", "1",
         "-vf", f"scale='if(gt(iw,ih),{side},-2)':'if(gt(iw,ih),-2,{side})'", "-q:v", str(q), "-f", "image2", "-c:v", "mjpeg", "pipe:1"],
        capture_output=True,
    )
    return r.stdout if r.returncode == 0 and r.stdout else None


END_MARGIN_S = 0.5
"""Seeking to within a few frames of the end can return no frame; stay half a second clear."""


def sample_times(duration_s: float, embed_n: int, temporal_n: int) -> tuple[list[float], list[float]]:
    d = max(duration_s, 0.0)
    if d <= 0:
        return [0.0], [0.0]
    last = max(d - END_MARGIN_S, 0.0)
    embed = list(np.linspace(0, last, embed_n)) if embed_n > 1 else [0.0]
    if d <= temporal_n:
        temporal = [float(t) for t in np.arange(0, last + 1e-9, 1.0)][:temporal_n]
    else:
        temporal = list(np.linspace(0, last, temporal_n))
    if temporal and abs(temporal[-1] - last) > 0.5:
        temporal.append(last)
    return [float(t) for t in embed], [float(t) for t in temporal]


def sample(path: str | Path, side: int = 448, embed_n: int = 4, temporal_n: int = 16, max_audio_s: int = 600) -> VideoSample:
    path = Path(path)
    pr = probe(path)
    et, tt = sample_times(pr.duration_s, embed_n, temporal_n)
    out = VideoSample(probe=pr)
    out.embed = [b for b in (_frame_at(path, t, side) for t in et) if b]
    out.temporal = [(t, b) for t, b in ((t, _frame_at(path, t, side)) for t in tt) if b]
    out.thumb = out.temporal[0][1] if out.temporal and out.temporal[0][0] == 0.0 else _frame_at(path, 0.0, side)
    if pr.has_audio:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-t", str(max_audio_s), "-vn", "-ac", "1", "-ar", "16000", "-f", "flac", "pipe:1"],
            capture_output=True,
        )
        out.audio_flac = r.stdout if r.returncode == 0 and r.stdout else None
    return out


def download(url: str, timeout: float = 60.0, max_bytes: int = 200 * 1024 * 1024) -> Path:
    import requests

    tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        n = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                n += len(chunk)
                if n > max_bytes:
                    raise ValueError("video too large")
                f.write(chunk)
    return tmp


def frame_motion(frame_vecs: np.ndarray) -> float:
    """Mean cosine distance between consecutive frame embeddings: slideshow (~0) vs cut-heavy (high)."""
    v = np.asarray(frame_vecs, dtype=np.float32)
    if len(v) < 2:
        return 0.0
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return float(np.mean(1.0 - np.sum(v[1:] * v[:-1], axis=1)))
