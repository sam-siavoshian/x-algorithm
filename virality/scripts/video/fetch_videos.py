#!/usr/bin/env python3
"""Fetch the video of each target post through X's public embed endpoint and keep small derived files.

    python scripts/video/fetch_videos.py --index media/videos/index.parquet --out <dir> [--max-prio 3]

For a public post, `cdn.syndication.twimg.com/tweet-result?id=<post id>` returns the mp4 variants with no login,
cookie or key. The lowest-bitrate variant is downloaded, then reduced to what the analysis needs and deleted:
  audio/<post_id>.opus     mono 16 kHz, first 90 seconds (absent when the video has no audio track)
  preview/<post_id>.mp4    2 frames a second, 224 px high, no audio, first 180 seconds (cuts, pace, frame content)
  meta.jsonl               one line per post: status, duration, size, bitrate, has_audio, like count now
One request a second to the endpoint; the run stops after 40 refusals (403/429) in a row. Resumable.
The owner of this project approved this fetch on 2026-11-16.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import polars as pl
import requests

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-prio", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    for d in ("audio", "preview", "tmp"):
        (out / d).mkdir(parents=True, exist_ok=True)
    meta = out / "meta.jsonl"
    done = {json.loads(x)["post_id"] for x in open(meta)} if meta.exists() else set()
    todo = pl.read_parquet(args.index).filter(pl.col("prio") <= args.max_prio).sort("prio", "post_id")["post_id"].to_list()
    todo = [p for p in todo if p not in done]
    if args.limit:
        todo = todo[:args.limit]
    s, refused, sink = requests.Session(), 0, open(meta, "a")
    for n, pid in enumerate(todo):
        rec, t0 = {"post_id": pid, "status": "ok"}, time.time()
        try:
            r = s.get("https://cdn.syndication.twimg.com/tweet-result", params={"id": str(pid), "token": "a"}, headers=UA, timeout=20)
            if r.status_code in (403, 429):
                refused += 1
                if refused >= 40:
                    print("stopping: 40 refusals in a row", flush=True)
                    break
                time.sleep(min(60, 5 * refused))
                continue                                   # not recorded: retried on the next run
            refused = 0
            if r.status_code != 200 or not r.text.strip() or r.text.strip() == "{}":
                rec["status"] = f"http_{r.status_code}"
            else:
                j = r.json()
                rec["likes_now"] = j.get("favorite_count")
                vids = [m for m in (j.get("mediaDetails") or []) if m.get("type") in ("video", "animated_gif")]
                if not vids:
                    rec["status"] = "no_video" if j.get("__typename") == "Tweet" else str(j.get("__typename") or "gone").lower()
                else:
                    v = vids[0]
                    info = v.get("video_info") or {}
                    mp4 = sorted([x for x in info.get("variants", []) if x.get("content_type") == "video/mp4"], key=lambda x: x.get("bitrate") or 0)
                    rec.update(media_type=v.get("type"), duration=(info.get("duration_millis") or 0) / 1000 or None, variants=len(mp4),
                               width=(v.get("original_info") or {}).get("width"), height=(v.get("original_info") or {}).get("height"),
                               max_bitrate=max([x.get("bitrate") or 0 for x in mp4], default=None))
                    if not mp4:
                        rec["status"] = "no_mp4"
                    else:
                        tmp = out / "tmp" / f"{pid}.mp4"
                        with s.get(mp4[0]["url"], headers=UA, timeout=60, stream=True) as g:
                            g.raise_for_status()
                            with open(tmp, "wb") as f:
                                for chunk in g.iter_content(1 << 16):
                                    f.write(chunk)
                        rec["bytes"] = tmp.stat().st_size
                        p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration", "-of", "json", str(tmp)], capture_output=True, text=True)
                        pj = json.loads(p.stdout or "{}")
                        rec["has_audio"] = any(x.get("codec_type") == "audio" for x in pj.get("streams", []))
                        rec["duration"] = rec.get("duration") or float((pj.get("format") or {}).get("duration") or 0) or None
                        if rec["has_audio"]:
                            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-t", "90", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "24k",
                                            str(out / "audio" / f"{pid}.opus")], capture_output=True)
                        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-t", "180", "-an", "-vf", "fps=2,scale=-2:224", "-c:v", "libx264", "-preset", "veryfast",
                                        "-crf", "30", str(out / "preview" / f"{pid}.mp4")], capture_output=True)
                        tmp.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"error:{type(e).__name__}"
        sink.write(json.dumps(rec) + "\n")
        sink.flush()
        if (n + 1) % 100 == 0:
            print(f"{n + 1}/{len(todo)} last={rec['status']}", flush=True)
        time.sleep(max(0.0, 1.0 - (time.time() - t0)))
    print("FETCH_DONE", flush=True)


if __name__ == "__main__":
    main()
