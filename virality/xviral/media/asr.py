"""Speech transcription for video audio (ARCHITECTURE.md S0).

X's grox pipeline transcribes 16 kHz mono audio up to 600 s and appends the
text to the post embedding input as 'Transcript:' capped at MAX_ASR = 2000
characters. We do the same with Whisper large-v3-turbo (MIT) via faster-whisper.
Speech ratio and speech in the first 3 seconds are kept as scalars: autoplay
shows the opening seconds muted or unmuted depending on the viewer.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from functools import lru_cache

log = logging.getLogger(__name__)

MAX_ASR_CHARS = 2000


@dataclass
class Transcript:
    text: str
    language: str | None
    speech_ratio: float
    speech_in_first_3s: bool
    duration_s: float


@lru_cache(maxsize=1)
def _model(name: str, device: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError("install the 'encoders' extra (faster-whisper) on the GPU box") from e
    return WhisperModel(name, device=device, compute_type="float16" if device == "cuda" else "int8")


def summarize(segments: list[tuple[float, float, str, float]], duration_s: float) -> tuple[str, float, bool]:
    """segments: (start, end, text, no_speech_prob). Returns (text, speech_ratio, speech_in_first_3s)."""
    speech = [(s, e, t) for s, e, t, nsp in segments if nsp < 0.6 and t.strip()]
    text = " ".join(t.strip() for _, _, t in speech)[:MAX_ASR_CHARS]
    covered = sum(max(e - s, 0) for s, e, _ in speech)
    ratio = min(covered / duration_s, 1.0) if duration_s > 0 else 0.0
    first3 = any(s < 3.0 for s, _, _ in speech)
    return text, ratio, first3


def transcribe(audio_flac: bytes, name: str = "large-v3-turbo", device: str = "cuda") -> Transcript:
    model = _model(name, device)
    segs, info = model.transcribe(io.BytesIO(audio_flac), vad_filter=True, beam_size=1)
    seg_list = [(s.start, s.end, s.text, s.no_speech_prob) for s in segs]
    text, ratio, first3 = summarize(seg_list, info.duration)
    return Transcript(text=text, language=info.language, speech_ratio=ratio, speech_in_first_3s=first3, duration_s=info.duration)
