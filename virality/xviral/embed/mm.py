"""The X-faithful post embedding (ARCHITECTURE.md S1, DATA_PIPELINE.md 7.1).

Uses phoenix/reference/mm_encoder.py from this repository directly: its
Post/Author/Image/Video classes, render_post (author line, t.co stripping,
image pads in position, quoted post, video-frame subsampling), the v5 system
prompt, and truncate_l2 (4096 -> 1024, L2-normalised). The only addition is
batching: `conversations` builds exactly the inputs _encode_transformers
builds for one post, so a batch of them can go through the model in one call.
tests/test_mm.py checks that equivalence against mm_encoder itself.
"""

from __future__ import annotations

import importlib.util
import io
import logging
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_DIR = REPO_ROOT / "phoenix" / "reference"


@lru_cache(maxsize=None)
def reference_module(name: str):
    """Import a module from phoenix/reference by file path (it is not an installed package)."""
    path = REFERENCE_DIR / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; xviral expects to live inside the x-algorithm repo")
    if str(REFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(REFERENCE_DIR))  # sid_codebook/sid_assign import sid_io by name
    spec = importlib.util.spec_from_file_location(f"xreference_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations through sys.modules
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # leave no __pycache__ inside phoenix/reference
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = prev
    return mod


def mm():
    return reference_module("mm_encoder")


def build_post(row: dict, images: list[bytes] | None = None, video_frames: list[list[bytes]] | None = None,
               quoted: dict | None = None):
    """Unified row -> mm_encoder.Post. Media order follows the post: images, then videos."""
    m = mm()
    media = [m.Image(content=b) for b in (images or []) if b]
    media += [m.Video(frames=[f for f in frames if f]) for frames in (video_frames or []) if frames]
    author = m.Author(name=row.get("author_display_name"), handle=row.get("author_handle"))
    q = None
    if quoted and quoted.get("text"):
        q = m.Post(id=quoted.get("post_id"), text=quoted.get("text"),
                   author=m.Author(name=quoted.get("author_display_name"), handle=quoted.get("author_handle")))
    text = row.get("text_note") or row.get("text")
    return m.Post(id=row.get("post_id"), text=text, author=author, media=media or None, quoted_post=q)


def render(post, transcript: str | None = None, max_images: int | None = 8) -> tuple[str, list[bytes]]:
    """render_post plus the transcript suffix exactly as mm_encoder.embed_post appends it."""
    text, imgs = mm().render_post(post, max_images=max_images)
    if transcript:
        text += f"\nTranscript: {transcript}"
    return text, imgs


def conversation(text: str, images: list[bytes], system_prompt: str | None = None):
    """The model input _encode_transformers builds for one post.

    Returns ('text', clean_text) for text-only posts (encoded with prompt=system_prompt)
    or ('conv', conversation) when images are present.
    """
    m = mm()
    system_prompt = system_prompt or m.DEFAULT_SYSTEM_PROMPT
    clean = " ".join(text.replace(m.IMAGE_PAD, " ").split()).strip()
    if not images:
        return "text", clean
    from PIL import Image as PILImage

    content = [{"type": "image", "image": PILImage.open(io.BytesIO(b)).convert("RGB")} for b in images]
    if clean:
        content.append({"type": "text", "text": clean})
    return "conv", [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]


class MMEncoder:
    """Batched Qwen3-VL-Embedding-8B encoder with mm_encoder's exact inputs and output transform."""

    def __init__(self, device: str = "cuda", batch_size: int = 16, model=None):
        m = mm()
        self.system_prompt = m.DEFAULT_SYSTEM_PROMPT
        self.dim = m.TRUNCATE_DIM
        self.batch_size = batch_size
        self.model = model if model is not None else m._load_st_model(m.QWEN_MODEL_ID, device)

    def encode(self, rendered: list[tuple[str, list[bytes]]]) -> np.ndarray:
        m = mm()
        out = np.zeros((len(rendered), self.dim), dtype=np.float32)
        kinds = [conversation(t, imgs, self.system_prompt) for t, imgs in rendered]
        text_idx = [i for i, (k, _) in enumerate(kinds) if k == "text"]
        conv_idx = [i for i, (k, _) in enumerate(kinds) if k == "conv"]
        if text_idx:
            raw = self.model.encode([kinds[i][1] for i in text_idx], prompt=self.system_prompt, batch_size=self.batch_size)
            for i, v in zip(text_idx, raw):
                out[i] = m.truncate_l2(np.asarray(v), self.dim)
        if conv_idx:
            raw = self.model.encode([kinds[i][1] for i in conv_idx], batch_size=self.batch_size)
            for i, v in zip(conv_idx, raw):
                out[i] = m.truncate_l2(np.asarray(v), self.dim)
        return out
