"""Audio features of a post's video, on CPU: is there sound, what kind, when speech starts, how fast, how loud.

    f = features("clip.mp4")            # dict of scalars plus the transcript; None values where a thing does not apply

What is measured and how (every threshold is a constant below, so the evidence and the product read the same thing):
  has_audio_stream   the file has an audio track (ffprobe)
  audible            the track is not empty: integrated loudness above AUDIBLE_LUFS and under 95% of frames silent
  loudness           EBU R128 integrated loudness (LUFS), loudness range (LU) and true peak (dBTP) from ffmpeg's ebur128
  silent_share       share of 50 ms frames whose RMS is under SILENCE_DBFS
  speech             Silero voice-activity detection (the copy shipped with faster-whisper) on 16 kHz mono
  music              an AudioSet tagger (distilled AST, 527 classes) on 10 s windows: a window counts as music when
                     P(Music) >= MUSIC_P (0.3), as speech when P(Speech) >= SPEECH_P
  sound_class        silent | speech_only | music_only | speech_and_music | other, from the shares (cut-offs below)
  time_to_first_speech_s   start of the first transcribed word (Whisper word timestamps), else the first voiced segment
  speech_rate_wpm    transcribed words per minute of voiced time, over the transcribed part
  tempo_bpm, onset_rate    librosa beat tracker and onset detector on the music windows (only when music is present)
  opening line       the first sentence of the transcript (at most OPENING_WORDS words), with the same trait definitions
                     as the post's first line in xviral/text/post_type.py: names the product, problem first, number

Analysis is capped at MAX_S seconds per video; launch videos are mostly shorter. Nothing here needs a GPU.
"""

from __future__ import annotations

import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from xviral.text import post_type as PT

SR = 16000
MAX_S = 120.0                # seconds analysed per video (tagger, transcript, tempo)
SILENCE_DBFS = -50.0         # a 50 ms frame under this RMS is silent
AUDIBLE_LUFS = -50.0         # a track quieter than this overall counts as having no sound
MUSIC_P, SPEECH_P = 0.3, 0.5   # on controlled mixes pure speech scores P(Music) <= 0.05 and speech over a quiet bed >= 0.36
WIN_S = 10.0                 # tagger window
SPEECH_MIN, MUSIC_MIN = 0.15, 0.30   # shares of the analysed time needed to call a video "has speech" / "has music"
OPENING_WORDS = 30
TAGGER = "bookbot/distil-ast-audioset"
ASR_MODEL = "small"
ASR_MAX_S = 75.0             # seconds transcribed per video: the opening line, the first speech and the rate need no more

_RESULT = re.compile(r"\b(sota|state[- ]of[- ]the[- ]art|\d+(\.\d+)?x (faster|cheaper|smaller|better|more)|faster|cheaper|beats|outperforms|#1|top[- ]ranked|record|best[- ]in[- ]class|\d+(\.\d+)?%)")
_ANNOUNCE = re.compile(r"\b(introducing|announcing|unveiling|launching|launched|launch|meet|say hello|releasing|released|open[- ]?sourcing|excited to|thrilled to|proud to|today,? (we|i))\b")
_NUMWORD = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|fifty|hundred|thousand|million|billion)\b")


def probe(path: str | Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, text=True)
    info = json.loads(out.stdout or "{}")
    st = info.get("streams", [])
    a = next((s for s in st if s.get("codec_type") == "audio"), None)
    v = next((s for s in st if s.get("codec_type") == "video"), {})
    return {"duration_s": float(info.get("format", {}).get("duration") or 0.0), "has_audio_stream": a is not None, "audio_codec": a and a.get("codec_name"),
            "audio_bitrate": a and int(a.get("bit_rate") or 0), "audio_channels": a and int(a.get("channels") or 0), "width": int(v.get("width") or 0), "height": int(v.get("height") or 0)}


def decode(path: str | Path, sr: int = SR, max_s: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)] + (["-t", str(max_s)] if max_s else []) + ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.float32).copy()


def ebur128(path: str | Path) -> dict:
    """Integrated loudness, loudness range and true peak of the whole track."""
    err = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    tail = err[err.rfind("Summary:"):] if "Summary:" in err else ""
    def grab(pat):
        m = re.search(pat, tail)
        try:
            return float(m.group(1)) if m else None
        except ValueError:
            return None
    return {"lufs": grab(r"I:\s+(-?[\d.]+|-inf) LUFS"), "lra_lu": grab(r"LRA:\s+(-?[\d.]+) LU"), "true_peak_dbtp": grab(r"Peak:\s+(-?[\d.]+|-inf) dBFS")}


def level_stats(x: np.ndarray, sr: int = SR) -> dict:
    """Frame levels of a mono signal: share of silent 50 ms frames, RMS level, and the 95th minus 10th percentile of the
    levels of the non-silent frames (a dynamic range that does not need 3 s of programme, unlike LRA)."""
    n = int(0.05 * sr)
    if len(x) < n:
        return {"silent_share": 1.0, "rms_dbfs": None, "level_range_db": None, "silent_frames": np.ones(0, bool)}
    fr = x[: len(x) // n * n].reshape(-1, n)
    db = 10 * np.log10(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-12)
    sil = db < SILENCE_DBFS
    loud = db[~sil]
    return {"silent_share": float(sil.mean()), "rms_dbfs": float(10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12)),
            "level_range_db": float(np.quantile(loud, 0.95) - np.quantile(loud, 0.10)) if len(loud) >= 20 else None, "silent_frames": sil}


def voiced_segments(x: np.ndarray, sr: int = SR) -> list[tuple[float, float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    ts = get_speech_timestamps(x, VadOptions(min_silence_duration_ms=300, speech_pad_ms=50))
    return [(t["start"] / sr, t["end"] / sr) for t in ts]


@lru_cache(maxsize=1)
def _tagger():
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    fe = ASTFeatureExtractor.from_pretrained(TAGGER)
    m = ASTForAudioClassification.from_pretrained(TAGGER).eval()
    lab = {v: k for k, v in m.config.id2label.items()}
    return fe, m, lab, torch


TAGS = ("Speech", "Music", "Singing", "Silence", "Narration, monologue", "Conversation", "Speech synthesizer", "Male speech, man speaking", "Female speech, woman speaking",
        "Electronic music", "Background music", "Applause", "Typing", "Computer keyboard", "Sound effect", "Drum", "Laughter")


def tag_windows(x: np.ndarray, sr: int = SR, win_s: float = WIN_S) -> list[dict]:
    """AudioSet tag probabilities per window (the last window may be shorter; windows under 1 s are dropped)."""
    fe, m, lab, torch = _tagger()
    w = int(win_s * sr)
    chunks = [(i / sr, x[i:i + w]) for i in range(0, len(x), w) if len(x[i:i + w]) >= sr]
    out = []
    for b in range(0, len(chunks), 8):
        part = chunks[b:b + 8]
        inp = fe([c for _, c in part], sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            p = torch.sigmoid(m(**inp).logits).numpy()
        for (t0, c), row in zip(part, p):
            out.append({"t0": t0, "len_s": len(c) / sr} | {k: float(row[lab[k]]) for k in TAGS if k in lab})
    return out


@lru_cache(maxsize=1)
def _asr(name: str):
    from faster_whisper import WhisperModel
    return WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=4)


def transcribe(x: np.ndarray, name: str = ASR_MODEL) -> dict:
    segs, info = _asr(name).transcribe(x, vad_filter=True, beam_size=1, word_timestamps=True, condition_on_previous_text=False)
    words, text = [], []
    for s in segs:
        if s.no_speech_prob >= 0.6 or not s.text.strip():
            continue
        text.append(s.text.strip())
        words += [(float(w.start), float(w.end), w.word.strip()) for w in (s.words or []) if w.word.strip()]
    return {"text": " ".join(text), "language": info.language, "language_p": float(info.language_probability), "words": words}


def opening_line(text: str, max_words: int = OPENING_WORDS) -> str:
    """The first sentence of a transcript: up to the first . ! or ? that follows at least two words, at most max_words."""
    toks = (text or "").split()
    out = []
    for t in toks[:max_words]:
        out.append(t)
        if len(out) >= 2 and re.search(r"[.!?][\"')\]]?$", t) and not re.fullmatch(r"(?:[A-Z]\.|Mr\.|Dr\.|vs\.|\d+\.)", t):
            break
    return " ".join(out)


def opening_traits(line: str, product_names: list[str] | None = None) -> dict:
    """Traits of the spoken opening line. The first three use the regular expressions of xviral/text/post_type.py unchanged.
    `says_name` is the spoken counterpart of naming the product: one of `product_names` (the names the post's own first
    line carries, the account's handle or display name) is heard in the line, compared without case, spaces or punctuation."""
    lc = line.lower().replace("’", "'")
    squash = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    sq = squash(line)
    names = [squash(n) for n in (product_names or [])]
    return {"open_names_product": bool(PT._NAMES.search(line)), "open_problem": bool(PT._PROBLEM.search(lc)), "open_number": bool(PT._NUMBER.search(lc) or _NUMWORD.search(lc)),
            "open_result": bool(_RESULT.search(lc)), "open_announce": bool(_ANNOUNCE.search(lc)), "open_question": "?" in line,
            "says_name": (any(len(n) >= 3 and n in sq for n in names) if names else None)}


def name_candidates(post_text: str | None, handle: str | None = None, display_name: str | None = None) -> list[str]:
    """Names a voiceover could say: what the post's first line names, plus the account's handle and display name."""
    raw = PT._LINK.sub("", post_text or "").strip()
    first = next((ln for ln in raw.split("\n") if ln.strip()), "")
    out = []
    for m in re.finditer(r"(?:[Ii]ntroducing|[Aa]nnouncing|[Mm]eet|[Ll]aunching|[Ll]aunch(?:ed)?|[Rr]eleas(?:e|ed|ing)|[Uu]nveiling|[Ss]ay hello to|[Oo]pen[- ]?sourc(?:ed|ing))[:\s]+(?:the |our |my |a |an )?(?:new |all-new )?([@A-Z0-9][\w.-]+(?: [A-Z0-9][\w.-]*)?)", first):
        out += [m.group(1), m.group(1).split()[0]]
    out += re.findall(r"\b[A-Z][a-z]+[A-Z]\w*|\b[A-Z][A-Za-z]*[- ]?\d+(?:\.\d+)*\b", first) + [h[1:] for h in re.findall(r"@\w+", first)]
    out += [handle or "", (display_name or "").split("|")[0].split("(")[0].strip()]
    return sorted({o.strip(" .:,-") for o in out if len(re.sub(r"[^A-Za-z0-9]", "", o)) >= 3})


def music_rhythm(y: np.ndarray, sr: int) -> dict:
    import librosa
    if len(y) < 4 * sr:
        return {"tempo_bpm": None, "onset_rate": None, "percussive_share": None, "beat_strength": None}
    env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr)
    h, p = librosa.effects.hpss(y[: 30 * sr])
    eh, ep = float(np.sum(h ** 2)), float(np.sum(p ** 2))
    # how much stronger the onset envelope is on the beats than on average: a steady pulse scores high, a pad scores near 1
    bs = float(env[beats].mean() / (env.mean() + 1e-9)) if len(beats) >= 4 else None
    return {"tempo_bpm": float(np.atleast_1d(tempo)[0]), "onset_rate": float(len(onsets) / (len(y) / sr)), "percussive_share": ep / (eh + ep + 1e-12), "beat_strength": bs}


def _overlap(segs: list[tuple[float, float]], a: float, b: float) -> float:
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in segs)


def sound_class(audible: bool, speech_share: float, music_share: float) -> str:
    if not audible:
        return "silent"
    sp, mu = speech_share >= SPEECH_MIN, music_share >= MUSIC_MIN
    return "speech_and_music" if sp and mu else "speech_only" if sp else "music_only" if mu else "other"


def features(path: str | Path, product_names: list[str] | None = None, asr: bool = True, max_s: float = MAX_S) -> dict:
    pr = probe(path)
    f: dict = dict(pr)
    if not pr["has_audio_stream"]:
        return f | {"audible": False, "sound_class": "silent", "analysed_s": 0.0}
    f |= ebur128(path)
    x = decode(path, SR, max_s)
    dur = len(x) / SR
    lv = level_stats(x)
    f |= {k: lv[k] for k in ("silent_share", "rms_dbfs", "level_range_db")} | {"analysed_s": dur}
    f["audible"] = bool(dur >= 0.5 and f["lufs"] is not None and f["lufs"] > AUDIBLE_LUFS and lv["silent_share"] < 0.95)
    if not f["audible"]:
        return f | {"sound_class": "silent"}
    vs = voiced_segments(x)
    tw = tag_windows(x)
    tot = sum(w["len_s"] for w in tw) or 1.0
    f["vad_speech_share"] = _overlap(vs, 0, dur) / dur
    f["tag_speech_share"] = sum(w["len_s"] for w in tw if w["Speech"] >= SPEECH_P) / tot
    f["music_share"] = sum(w["len_s"] for w in tw if w["Music"] >= MUSIC_P) / tot
    f["music_p_mean"] = sum(w["len_s"] * w["Music"] for w in tw) / tot
    f["speech_p_mean"] = sum(w["len_s"] * w["Speech"] for w in tw) / tot
    f["singing_p_mean"] = sum(w["len_s"] * w["Singing"] for w in tw) / tot
    f["synth_tag_p_max"] = max((w.get("Speech synthesizer", 0.0) for w in tw), default=0.0)
    # speech = voiced by the detector inside windows the tagger also hears speech in (sung vocals trip the detector alone)
    sp_w = [(w["t0"], w["t0"] + w["len_s"]) for w in tw if w["Speech"] >= 0.2]
    speech = [(max(s, a), min(e, b)) for s, e in vs for a, b in sp_w if min(e, b) > max(s, a)]
    f["speech_share"] = _overlap(speech, 0, dur) / dur
    f["music_first_window"] = bool(tw and tw[0]["Music"] >= MUSIC_P)
    f["sound_class"] = sound_class(True, f["speech_share"], f["music_share"])
    f["first_voiced_s"] = speech[0][0] if speech else None
    tr = {"text": "", "language": None, "language_p": None, "words": []}
    if asr and f["speech_share"] >= 0.03:
        tr = transcribe(x[: int(ASR_MAX_S * SR)])
    words = tr["words"]
    f |= {"transcript": tr["text"][:4000], "language": tr["language"], "language_p": tr["language_p"], "n_words": len(words)}
    f["time_to_first_speech_s"] = words[0][0] if words else f["first_voiced_s"]
    f["speech_first_2s"] = bool(f["time_to_first_speech_s"] is not None and f["time_to_first_speech_s"] < 2.0)
    f["musical_intro"] = bool(f["music_first_window"] and not f["speech_first_2s"] and f["music_share"] >= MUSIC_MIN)
    voiced = _overlap(speech, 0, min(dur, ASR_MAX_S))
    f["speech_rate_wpm"] = float(60 * len(words) / voiced) if len(words) >= 15 and voiced >= 5 else None
    if words:
        t10 = [w for w in words if w[0] < words[0][0] + 10]
        f["opening_rate_wpm"] = float(60 * len(t10) / 10) if words[-1][0] >= words[0][0] + 10 else None
    line = opening_line(tr["text"])
    f["opening_line"] = line
    f |= opening_traits(line, product_names) if len(line.split()) >= 2 else {k: None for k in ("open_names_product", "open_problem", "open_number", "open_result", "open_announce", "open_question", "says_name")}
    # level of the speech against the rest (the music bed): positive = voice above the bed
    sil = lv["silent_frames"]; n = int(0.05 * SR); t = (np.arange(len(sil)) + 0.5) * 0.05
    insp = np.zeros(len(sil), bool)
    for s, e in speech:
        insp |= (t >= s) & (t < e)
    fr = x[: len(sil) * n].reshape(-1, n).astype(np.float64)
    e = np.mean(fr ** 2, axis=1)
    a, b = e[insp & ~sil], e[~insp & ~sil]
    f["speech_over_bed_db"] = float(10 * np.log10(a.mean() / b.mean())) if len(a) >= 40 and len(b) >= 40 else None
    if f["music_share"] >= MUSIC_MIN:
        import librosa
        mus = np.concatenate([x[int(w["t0"] * SR): int((w["t0"] + w["len_s"]) * SR)] for w in tw if w["Music"] >= MUSIC_P][:6])
        f |= music_rhythm(librosa.resample(mus, orig_sr=SR, target_sr=22050), 22050)
    else:
        f |= {"tempo_bpm": None, "onset_rate": None, "percussive_share": None, "beat_strength": None}
    f["windows"] = [{k: round(v, 3) for k, v in w.items() if k in ("t0", "len_s", "Speech", "Music", "Singing", "Speech synthesizer")} for w in tw]
    return f
