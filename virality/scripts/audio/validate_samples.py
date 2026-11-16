#!/usr/bin/env python3
"""Check xviral/media/audio.py on public audio with a known answer: read speech, four pieces of music, silence, and
controlled mixes of the two (speech 15, 5 and 0 dB above the music; a 6 s musical intro; 3 s of silence before speech).

    python scripts/audio/validate_samples.py     # downloads librosa's example recordings; writes data/audio/validation.json

Speech: two LibriVox readings (librosa examples libri1, libri2). Music: a drum and bass loop (choice), an orchestral
piece (nutcracker), a guitar song without vocals in the excerpt (fishin), a jazz piece (vibeace).
"""
import json, os, time, numpy as np, soundfile as sf
os.makedirs("data/audio/samples", exist_ok=True)
os.environ.setdefault("LIBROSA_DATA_DIR", "data/audio/samples")
import librosa
from xviral.media import audio as A

SR = A.SR
def ld(k, dur=None):
    y, _ = librosa.load(librosa.ex(k), sr=SR, mono=True, duration=dur)
    return y / (np.abs(y).max() + 1e-9) * 0.5
rms = lambda x: np.sqrt(np.mean(x ** 2))
def mix(s, m, db):       # speech `db` above the music
    n = min(len(s), len(m)); return s[:n] + m[:n] * rms(s[:n]) / (rms(m[:n]) * 10 ** (db / 20))

sp = np.concatenate([ld("libri1"), ld("libri2")])[: SR * 25]
mus = {k: ld(k, 40) for k in ("choice", "nutcracker", "fishin", "vibeace")}
cases = {"speech_only": (sp, "speech_only"), "silence": (np.zeros(SR * 10, np.float32), "silent"), "noise_at_-80dBFS": ((np.random.default_rng(0).standard_normal(SR * 10) * 1e-4).astype(np.float32), "silent")}
for k, m in mus.items(): cases["music_" + k] = (m, "music_only")
for db in (15, 5, 0):
    for k in ("choice", "nutcracker"): cases[f"speech_{db}dB_over_{k}"] = (mix(sp, mus[k], db), "speech_and_music")
cases["intro_6s_then_speech_over_vibeace"] = (np.concatenate([mus["vibeace"][: 6 * SR], mix(sp[: SR * 20], mus["vibeace"][6 * SR:], 8)]), "speech_and_music")
cases["silence_3s_then_speech"] = (np.concatenate([np.zeros(3 * SR, np.float32), sp[: SR * 15]]), "speech_only")
out, ok = [], 0
for k, (x, want) in cases.items():
    p = f"data/audio/samples/{k}.flac"; sf.write(p, x.astype(np.float32), SR)
    t = time.time(); f = A.features(p); dt = time.time() - t
    r = {"case": k, "expected": want, "got": f["sound_class"], "correct": f["sound_class"] == want, "seconds_audio": round(len(x) / SR, 1), "seconds_cpu": round(dt, 1)} | {
        m: (None if f.get(m) is None else round(f[m], 2) if isinstance(f[m], float) else f[m]) for m in ("lufs", "lra_lu", "silent_share", "speech_share", "music_share", "time_to_first_speech_s", "speech_first_2s", "musical_intro",
                                                                                                    "speech_rate_wpm", "tempo_bpm", "onset_rate", "percussive_share", "speech_over_bed_db", "opening_line")}
    out.append(r); ok += r["correct"]
    print(f"{k:<36}{want:<17}{f['sound_class']:<17} first speech {r['time_to_first_speech_s']}  wpm {r['speech_rate_wpm']}  bpm {r['tempo_bpm']}  lufs {r['lufs']}  [{dt:.1f}s]")
print(f"{ok} of {len(out)} classes correct")
json.dump({"thresholds": {"MUSIC_P": A.MUSIC_P, "SPEECH_P": A.SPEECH_P, "SPEECH_MIN": A.SPEECH_MIN, "MUSIC_MIN": A.MUSIC_MIN, "SILENCE_DBFS": A.SILENCE_DBFS, "AUDIBLE_LUFS": A.AUDIBLE_LUFS}, "correct": ok, "cases": out},
          open("data/audio/validation.json", "w"), indent=1)
