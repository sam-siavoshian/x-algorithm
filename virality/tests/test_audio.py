"""Audio features (xviral/media/audio.py): the parts that need no model download."""
import numpy as np

from xviral.media import audio as A


def test_level_stats_silence_and_tone():
    sr = A.SR
    t = np.arange(sr * 4) / sr
    x = np.concatenate([np.zeros(sr * 4, np.float32), (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)])
    s = A.level_stats(x)
    assert abs(s["silent_share"] - 0.5) < 0.02
    assert -27 < s["rms_dbfs"] < -25          # a 0.1 sine is -23 dBFS RMS; half the signal is silent
    assert A.level_stats(np.zeros(sr, np.float32))["silent_share"] == 1.0


def test_sound_class():
    assert A.sound_class(False, 0.9, 0.9) == "silent"
    assert A.sound_class(True, 0.5, 0.0) == "speech_only"
    assert A.sound_class(True, 0.0, 1.0) == "music_only"
    assert A.sound_class(True, 0.4, 0.8) == "speech_and_music"
    assert A.sound_class(True, 0.05, 0.1) == "other"


def test_opening_line_stops_at_first_sentence():
    assert A.opening_line("Introducing Atlas. It reads your codebase and writes the tests.") == "Introducing Atlas."
    assert A.opening_line("Hi. I'm Sam, and this is Atlas. More words follow here.") == "Hi. I'm Sam, and this is Atlas."
    assert len(A.opening_line(" ".join(["word"] * 100)).split()) == A.OPENING_WORDS
    assert A.opening_line("") == ""


def test_opening_traits_use_post_definitions():
    t = A.opening_traits("Introducing Atlas, the fastest way to ship.", ["Atlas"])
    assert t["open_names_product"] and t["open_announce"] and t["says_name"] and not t["open_problem"] and not t["open_number"]
    t = A.opening_traits("Writing tests is slow and nobody likes it.", ["Atlas"])
    assert t["open_problem"] and not t["open_names_product"] and t["says_name"] is False
    t = A.opening_traits("We cut build times by 40% for three hundred teams.")
    assert t["open_number"] and t["open_result"] and t["says_name"] is None
    assert A.opening_traits("Meet Deep Agent, from the team at lang chain.", ["DeepAgent", "LangChain"])["says_name"]


def test_name_candidates():
    n = A.name_candidates("Introducing Eleven Music: studio-grade songs.\n\nTry it https://x.co/a", "ElevenLabs", "ElevenLabs | AI audio")
    assert "Eleven Music" in n and "ElevenLabs" in n and "Eleven" in n
