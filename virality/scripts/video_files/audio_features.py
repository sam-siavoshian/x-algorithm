#!/usr/bin/env python3
"""Audio features of each video's reduced audio track (mono 16 kHz Opus, the first 90 seconds), CPU only.

    python scripts/video_files/audio_features.py <shard> <of> [limit]     # run <of> copies, one per core
        reads  data/vf/audio_x/audio/<post_id>.opus, data/vf/order.parquet, data/launch/table.parquet (names a voice could say)
        writes data/vf/audio/part_<shard>_<k>.parquet

Uses the functions and thresholds of xviral/media/audio.py (loudness, level statistics, voice detection, AudioSet
tagger, sound class, opening line and its traits). Three reductions against audio.features(), all for CPU time on
5,291 files (about 0.25 s of CPU per second of audio there, 18 hours for this set):
  - the tagger hears at most 4 ten-second windows a file: the first, and 3 spread evenly over the rest;
  - Whisper `base` (not `small`) transcribes the first 30 seconds only: enough for the opening line, the time to the
    first word and a speech rate over the opening 30 seconds;
  - no harmonic/percussive separation (percussive share dropped); tempo and onset rate from the music windows.
A video without an audio track (meta.jsonl has_audio false, GIFs included) has no file here and is silent by definition.
"""
import os, sys
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
import numpy as np, polars as pl
sys.path.insert(0, ".")
from xviral.media import audio as A
SHARD, OF = int(sys.argv[1]), int(sys.argv[2]); LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None
SRC = "data/vf/audio_x/audio"; OUT = "data/vf/audio"; CHUNK = 100; ASR_S = 30.0; ASR_NAME = "base"; SR = A.SR


def pick_windows(x):
    w = int(A.WIN_S * SR); starts = [i for i in range(0, len(x), w) if len(x) - i >= SR]
    if len(starts) > 4: starts = [starts[0]] + [starts[int(round(k))] for k in np.linspace(1, len(starts) - 1, 3)]
    return starts, w


def tag(x):
    fe, m, lab, torch = A._tagger()
    starts, w = pick_windows(x); out = []
    if not starts: return out
    with torch.no_grad():
        p = torch.sigmoid(m(**fe([x[s:s + w] for s in starts], sampling_rate=SR, return_tensors="pt")).logits).numpy()
    for s, row in zip(starts, p):
        out.append({"t0": s / SR, "len_s": min(w, len(x) - s) / SR} | {k: float(row[lab[k]]) for k in A.TAGS if k in lab})
    return out


def feats(path, names, asr):
    f = A.ebur128(path)
    x = A.decode(path, SR); dur = len(x) / SR
    lv = A.level_stats(x)
    f |= {k: lv[k] for k in ("silent_share", "rms_dbfs", "level_range_db")} | {"analysed_s": dur}
    f["audible"] = bool(dur >= 0.5 and f["lufs"] is not None and f["lufs"] > A.AUDIBLE_LUFS and lv["silent_share"] < 0.95)
    if not f["audible"]: return f | {"sound_class": "silent"}
    vs = A.voiced_segments(x); tw = tag(x); tot = sum(w["len_s"] for w in tw) or 1.0
    f["vad_speech_share"] = A._overlap(vs, 0, dur) / dur
    f["music_share"] = sum(w["len_s"] for w in tw if w["Music"] >= A.MUSIC_P) / tot
    f["tag_speech_share"] = sum(w["len_s"] for w in tw if w["Speech"] >= 0.2) / tot
    for k, n in (("Music", "music_p_mean"), ("Speech", "speech_p_mean"), ("Singing", "singing_p_mean")): f[n] = sum(w["len_s"] * w[k] for w in tw) / tot
    f["synth_tag_p_max"] = max((w.get("Speech synthesizer", 0.0) for w in tw), default=0.0)
    # speech share: voiced time (detector) scaled by the share of tagged windows that also hear speech; with all windows
    # tagged this equals audio.features(); with 4 of up to 9 tagged it assumes the untagged windows are like the tagged
    f["speech_share"] = f["vad_speech_share"] * f["tag_speech_share"]
    f["music_first_window"] = bool(tw and tw[0]["Music"] >= A.MUSIC_P)
    f["sound_class"] = A.sound_class(True, f["speech_share"], f["music_share"])
    sp0 = [(s, e) for s, e in vs if s < ASR_S]
    f["first_voiced_s"] = vs[0][0] if vs and f["tag_speech_share"] > 0 else None
    words, text, lang, langp = [], "", None, None
    if f["speech_share"] >= 0.03 and sp0:
        segs, info = asr.transcribe(x[: int(ASR_S * SR)], vad_filter=True, beam_size=1, word_timestamps=True, condition_on_previous_text=False)
        tx = []
        for s in segs:
            if s.no_speech_prob >= 0.6 or not s.text.strip(): continue
            tx.append(s.text.strip()); words += [(float(w.start), float(w.end), w.word.strip()) for w in (s.words or []) if w.word.strip()]
        text, lang, langp = " ".join(tx), info.language, float(info.language_probability)
    f |= {"transcript": text[:2000], "language": lang, "language_p": langp, "n_words": len(words)}
    f["time_to_first_speech_s"] = words[0][0] if words else (f["first_voiced_s"] if f["speech_share"] >= A.SPEECH_MIN else None)
    f["speech_first_2s"] = bool(f["time_to_first_speech_s"] is not None and f["time_to_first_speech_s"] < 2.0)
    f["musical_intro"] = bool(f["music_first_window"] and not f["speech_first_2s"] and f["music_share"] >= A.MUSIC_MIN)
    voiced = A._overlap(vs, 0, min(dur, ASR_S))
    f["speech_rate_wpm"] = float(60 * len(words) / voiced) if len(words) >= 15 and voiced >= 5 else None
    line = A.opening_line(text); f["opening_line"] = line
    f |= A.opening_traits(line, names) if len(line.split()) >= 2 else {k: None for k in ("open_names_product", "open_problem", "open_number", "open_result", "open_announce", "open_question", "says_name")}
    if f["music_share"] >= A.MUSIC_MIN:
        import librosa
        mus = np.concatenate([x[int(w["t0"] * SR): int((w["t0"] + w["len_s"]) * SR)] for w in tw if w["Music"] >= A.MUSIC_P][:3])
        if len(mus) >= 4 * SR:
            env = librosa.onset.onset_strength(y=mus, sr=SR); tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR); on = librosa.onset.onset_detect(onset_envelope=env, sr=SR)
            f |= {"tempo_bpm": float(np.atleast_1d(tempo)[0]), "onset_rate": float(len(on) / (len(mus) / SR)), "beat_strength": float(env[beats].mean() / (env.mean() + 1e-9)) if len(beats) >= 4 else None}
    f["windows"] = str([{k: round(v, 3) for k, v in w.items() if k in ("t0", "Speech", "Music", "Singing")} for w in tw])
    return f


def main():
    import torch
    from faster_whisper import WhisperModel
    torch.set_num_threads(1); os.makedirs(OUT, exist_ok=True)
    ids = pl.read_parquet("data/vf/order.parquet")["post_id"].to_list()
    ids = [p for p in ids if os.path.exists(f"{SRC}/{p}.opus")]
    ids = [p for i, p in enumerate(ids) if i % OF == SHARD]
    done = set()
    for fn in os.listdir(OUT):
        if fn.startswith(f"part_{SHARD}_"): done |= set(pl.read_parquet(os.path.join(OUT, fn), columns=["post_id"])["post_id"].to_list())
    ids = [p for p in ids if p not in done][:LIMIT]; k0 = len([fn for fn in os.listdir(OUT) if fn.startswith(f"part_{SHARD}_")])
    T = pl.read_parquet("data/launch/table.parquet", columns=["post_id", "text", "author_handle", "author_display_name"]).filter(pl.col("post_id").is_in(ids))
    meta = {r[0]: r[1:] for r in T.iter_rows()}
    asr = WhisperModel(ASR_NAME, device="cpu", compute_type="int8", cpu_threads=1)
    rows = []
    print("to do", len(ids), flush=True)
    for n, pid in enumerate(ids):
        t, h, dn = meta.get(pid, (None, None, None))
        try: r = feats(f"{SRC}/{pid}.opus", A.name_candidates(t, h, dn), asr)
        except Exception as e: r = {"error": repr(e)[:200]}  # noqa: BLE001
        rows.append({"post_id": pid} | r)
        if len(rows) >= CHUNK or n == len(ids) - 1:
            pl.DataFrame(rows, infer_schema_length=None).write_parquet(f"{OUT}/part_{SHARD}_{k0}.parquet"); k0 += 1; rows = []; print(n + 1, "/", len(ids), flush=True)
    print("AUDIO_DONE", SHARD, flush=True)


main()
