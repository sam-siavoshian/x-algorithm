#!/usr/bin/env python3
"""Audio features of every video file in a folder, one row per post.

    python scripts/audio/extract.py <folder with <post_id>.mp4 (or .m4a/.flac/.wav/.opus)> [out.parquet]

Reads data/tracks/launch/launch_posts.parquet for the names a voiceover could say (the post's first line, the account's
handle and display name) and data/media/videos/index.parquet (the video track's index) to mark GIFs. Resumable: rows are
appended to <out>.jsonl and files already done are skipped. About 0.25 s of CPU time per second of audio with speech
(Whisper small, int8, 4 threads), 0.1 s without.
"""
import glob, json, os, sys, time, polars as pl
sys.path.insert(0, ".")
from xviral.media import audio as A

src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "data/audio/features.parquet"
L = pl.read_parquet("data/tracks/launch/launch_posts.parquet").select("post_id", "text", "author_handle", "author_display_name")
names = {r["post_id"]: A.name_candidates(r["text"], r["author_handle"], r["author_display_name"]) for r in L.iter_rows(named=True)}
gif = set()
if os.path.exists("data/media/videos/index.parquet"):
    gif = set(pl.read_parquet("data/media/videos/index.parquet").filter(pl.col("mtype") == "gif")["post_id"].to_list())
log = out + ".jsonl"
done = {json.loads(l)["post_id"] for l in open(log)} if os.path.exists(log) else set()
files = sorted(f for f in glob.glob(os.path.join(src, "*")) if os.path.basename(f).split(".")[0].isdigit() and int(os.path.basename(f).split(".")[0]) not in done)
print(len(files), "files to do,", len(done), "done", flush=True)
with open(log, "a") as fh:
    for n, f in enumerate(files):
        pid = int(os.path.basename(f).split(".")[0]); t = time.time()
        try:
            row = A.features(f, names.get(pid)) | {"error": None}
        except Exception as e:           # one broken file must not stop the run
            row = {"error": f"{type(e).__name__}: {e}"[:200]}
        row |= {"post_id": pid, "is_gif": pid in gif, "name_candidates": names.get(pid, []), "seconds_cpu": round(time.time() - t, 2)}
        row["windows"] = json.dumps(row.get("windows", []))
        fh.write(json.dumps(row) + "\n"); fh.flush()
        if n % 25 == 0: print(n, pid, row.get("sound_class"), row["seconds_cpu"], flush=True)
rows = [json.loads(l) for l in open(log)]
keys = sorted({k for r in rows for k in r})
pl.DataFrame([{k: r.get(k) for k in keys} for r in rows], infer_schema_length=None).write_parquet(out)
print("wrote", out, len(rows))
