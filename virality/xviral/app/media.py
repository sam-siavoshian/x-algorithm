"""Media attached to a draft version: what the server records today, and the seam for media understanding.

Today
    A file uploaded with a version is read once, in memory, for the request. `probe` records its kind and basic
    properties (image: format, width, height, animated or not; video: container, width, height, duration, audio track).
    The scorer is told only "image" or "video", exactly as when the user ticks a box. Nothing is written to permanent
    storage; a video is spooled to a temporary file for ffprobe and deleted before `probe` returns.

The seam
    A `MediaAnalyzer` is any callable `(info: MediaInfo, data: bytes) -> MediaUnderstanding | None`. The media tracks
    (image and video embeddings, typed attributes) plug in by providing one:

        XVIRAL_MEDIA_ANALYZER=package.module:function        # or Service(..., media_analyzer=fn)

    The server calls it once per uploaded file, after `probe`, while the bytes are still in memory. What it returns is
      - attached to the version in the report (`versions[i].media.attributes`, and `embedding_dim`; never the vector);
      - passed to the scorer in the draft dict as `media_attributes` and `media_embedding`, next to `media`.
    `DraftScorer` ignores those two keys today. A model trained with media inputs reads them in
    `serve.DraftScorer._draft_row` / `featurize`; no change is needed in the service, the HTTP layer or the page.
    Attributes use the vocabulary below so that evidence measured on typed media ("demo video against talking head")
    can be keyed on the same names; an analyzer may add other keys, which are carried through untouched.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np

IMAGE_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}
VIDEO_CONTAINERS = ("mp4", "mov", "webm", "matroska", "m4v", "quicktime")

# Typed attributes an analyzer may return. `content` is one label; the booleans are independent.
ATTRIBUTE_VOCABULARY = {
    "content": ["product_screenshot", "product_demo", "talking_head", "screen_recording", "animation_or_motion_graphic", "chart_or_benchmark",
                "text_card", "logo_or_brand", "photo_of_people", "meme", "other"],
    "has_text_overlay": "bool", "has_captions": "bool", "has_speech": "bool", "has_music": "bool", "shows_product_ui": "bool", "shows_face": "bool",
    "first_frame": "one of the `content` labels, for the frame shown before playback (video only)",
}


class MediaError(ValueError):
    """The file cannot be used; the message is shown to the user."""


@dataclass
class MediaInfo:
    kind: str                           # "image" or "video": what the scorer is told
    mime: str
    bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    duration_s: Optional[float] = None  # video and animated images
    animated: bool = False              # an animated GIF or WebP; X plays it like a video, so kind is "video"
    has_audio: Optional[bool] = None    # video only
    filename: Optional[str] = None
    attributes: dict = field(default_factory=dict)        # from a MediaAnalyzer; empty today
    embedding_dim: Optional[int] = None

    def public(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None and v != {} and k != "filename"}


@dataclass
class MediaUnderstanding:
    attributes: dict = field(default_factory=dict)        # keys from ATTRIBUTE_VOCABULARY (others are carried through)
    embedding: Optional[np.ndarray] = None                # one vector for the file, from the media encoder
    extra: dict = field(default_factory=dict)             # anything else for the scorer: transcript, per-frame features


MediaAnalyzer = Callable[[MediaInfo, bytes], Optional[MediaUnderstanding]]


def _probe_image(data: bytes) -> MediaInfo | None:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            if fmt not in IMAGE_FORMATS:
                return None
            frames = getattr(im, "n_frames", 1)
            dur = None
            if frames > 1:
                total = 0
                for i in range(min(frames, 2000)):
                    im.seek(i)
                    total += im.info.get("duration", 0) or 0
                dur = round(total / 1000.0, 3) or None
            return MediaInfo(kind="video" if frames > 1 else "image", mime=IMAGE_FORMATS[fmt], bytes=len(data), width=im.width, height=im.height,
                             animated=frames > 1, duration_s=dur)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return None


def _probe_video(data: bytes, ffprobe: str = "ffprobe", timeout: float = 20.0) -> MediaInfo | None:
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:      # deleted on exit; the only time the bytes touch a disk
        f.write(data)
        f.flush()
        try:
            out = subprocess.run([ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", f.name], capture_output=True, timeout=timeout, check=False)
        except FileNotFoundError as e:
            raise MediaError("video files cannot be read on this server (ffprobe is not installed); tick \"video\" instead of uploading") from e
        except subprocess.TimeoutExpired as e:
            raise MediaError("the video took too long to read") from e
    if out.returncode != 0:
        return None
    try:
        meta = json.loads(out.stdout or b"{}")
    except json.JSONDecodeError:
        return None
    fmt = (meta.get("format") or {}).get("format_name", "")
    streams = meta.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video" and (s.get("disposition") or {}).get("attached_pic", 0) == 0), None)
    if video is None or not any(c in fmt for c in VIDEO_CONTAINERS):
        return None
    dur = (meta.get("format") or {}).get("duration") or video.get("duration")
    mime = "video/webm" if "webm" in fmt and "mp4" not in fmt else "video/quicktime" if fmt.startswith("mov") and "mp4" not in fmt else "video/mp4"
    return MediaInfo(kind="video", mime=mime, bytes=len(data), width=video.get("width"), height=video.get("height"),
                     duration_s=round(float(dur), 3) if dur not in (None, "N/A") else None, has_audio=any(s.get("codec_type") == "audio" for s in streams))


def probe(data: bytes, filename: str | None = None, max_bytes: int = 64 * 1024 * 1024) -> MediaInfo:
    """Kind and basic properties of an uploaded file. The content decides, not the file name or the declared type."""
    if not data:
        raise MediaError("the attached file is empty")
    if len(data) > max_bytes:
        raise MediaError(f"the attached file is {len(data) / 1e6:.0f} MB; the limit is {max_bytes / 1e6:.0f} MB")
    info = _probe_image(data) or _probe_video(data)
    if info is None:
        raise MediaError("the attached file is not an image (JPEG, PNG, WebP, GIF) or a video (MP4, MOV, WebM) we can read")
    info.filename = os.path.basename(filename) if filename else None
    return info


def understand(info: MediaInfo, data: bytes, analyzer: MediaAnalyzer | None) -> MediaUnderstanding | None:
    """Run the configured analyzer, if any, and record what it found on `info`. An analyzer that fails does not fail
    the request: the version is scored with the kind alone, as it is without an analyzer."""
    if analyzer is None:
        return None
    try:
        u = analyzer(info, data)
    except Exception:       # noqa: BLE001  a plug-in must not take the request down
        return None
    if u is None:
        return None
    info.attributes = dict(u.attributes)
    info.embedding_dim = int(np.asarray(u.embedding).shape[-1]) if u.embedding is not None else None
    return u


def load_analyzer(spec: str | None) -> MediaAnalyzer | None:
    """`package.module:function` -> the callable. Empty or None: no analyzer."""
    if not spec:
        return None
    mod, _, name = spec.partition(":")
    if not name:
        raise ValueError("XVIRAL_MEDIA_ANALYZER must look like package.module:function")
    return getattr(importlib.import_module(mod), name)
