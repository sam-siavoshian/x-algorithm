#!/usr/bin/env python3
"""Plumbing check of scripts/audio/effects.py on placeholder features dealt at random (no audio, no finding).

    python scripts/audio/dry_run.py

Every trait must come out at "no effect" (intervals around 50% and 1.0x); the script only proves the analysis runs on the
real post table, with the real joins, scopes and restrictions, before any audio is in hand.
"""
import os, sys, numpy as np, polars as pl
sys.path.insert(0, "scripts/audio")
import effects as E

r = np.random.default_rng(3)
I = pl.read_parquet("data/media/videos/index.parquet")
L = pl.read_parquet("data/tracks/launch/launch_posts.parquet").filter(pl.col("has_video")).select("post_id").join(I.select("post_id", "mtype"), on="post_id", how="left")
n = L.height
cls = r.choice(["silent", "speech_only", "music_only", "speech_and_music", "other"], n, p=[.1, .2, .3, .35, .05])
sp, mu = np.isin(cls, ["speech_only", "speech_and_music"]), np.isin(cls, ["music_only", "speech_and_music"])
num = lambda name, ok, v: pl.Series(name, np.where(ok, v, np.nan)).fill_nan(None)
F = L.with_columns(
    (pl.col("mtype") == "gif").fill_null(False).alias("is_gif"), pl.Series("sound_class", cls), pl.Series("audible", cls != "silent"),
    pl.Series("speech_first_2s", sp & (r.random(n) < .6)), pl.Series("musical_intro", sp & mu & (r.random(n) < .3)),
    num("speech_rate_wpm", sp, r.normal(155, 25, n)), num("tempo_bpm", mu, r.normal(118, 20, n)), num("onset_rate", mu, r.normal(3, 1, n)), num("percussive_share", mu, r.random(n) * .5),
    pl.Series("lufs", r.normal(-19, 5, n)), pl.Series("lra_lu", r.gamma(4, 2, n)), pl.Series("language", ["en" if s else None for s in sp], dtype=pl.String), pl.Series("opening_line", ["Introducing Atlas." if s else "" for s in sp]),
    *[pl.Series(k, r.random(n) < .3) for k in ("open_names_product", "says_name", "open_problem", "open_number", "open_result", "open_announce")]).drop("mtype")
os.makedirs("data/audio", exist_ok=True)
F.write_parquet("data/audio/_dry_features.parquet")
F.select("post_id").with_columns(pl.Series("first_frame_class", r.choice(["screen", "person", "graphic"], n))).write_parquet("data/audio/_dry_visual.parquet")
T = E.load_table("data/audio/_dry_features.parquet", "data/audio/_dry_visual.parquet")
print(T.height, "video posts joined")
import json
out = E.run(T, B=200)
E.show(out)
json.dump(out, open("data/audio/_dry_effects.json", "w"))    # only for checking build_audio_evidence.py; never an evidence file
os.remove("data/audio/_dry_features.parquet"); os.remove("data/audio/_dry_visual.parquet")
