#!/usr/bin/env python3
"""Fetch the poster frame (thumbnail) of every target video post from pbs.twimg.com (media[].url of the corpus row).
5 requests a second, stops on repeated 403/429, resumable. Keeps a 448 px JPEG per post in data/video/thumbs/ and a
status line in data/video/thumbs.jsonl.   python scripts/video/fetch_thumbs.py [targets.parquet]"""
import json, os, sys, time, requests, polars as pl
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, ".")
from xviral.media.fetch import RateLimiter, resize_jpeg
T = pl.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "data/video/targets.parquet")
OUT = "data/video/thumbs.jsonl"; os.makedirs("data/video/thumbs", exist_ok=True)
done = {json.loads(x)["post_id"] for x in open(OUT)} if os.path.exists(OUT) else set()
todo = [(i, u) for i, u in zip(T["post_id"].to_list(), T["thumb_url"].to_list()) if i not in done and u]
print("todo", len(todo), flush=True)
lim = RateLimiter(5.0); s = requests.Session(); s.headers["User-Agent"] = "Mozilla/5.0 (compatible; research-fetch)"
refused = [0]
def one(x):
    i, u = x
    if refused[0] >= 8: return None
    lim.wait()
    try:
        r = s.get(u, timeout=20)
        if r.status_code in (403, 429): refused[0] += 1
        if r.status_code != 200: return {"post_id": i, "status": r.status_code}
        refused[0] = 0
        jpg, w, h = resize_jpeg(r.content, 448, 88)
        open(f"data/video/thumbs/{i}.jpg", "wb").write(jpg)
        return {"post_id": i, "status": 200, "thumb_w": w, "thumb_h": h, "bytes": len(r.content)}
    except Exception as e:  # noqa: BLE001
        return {"post_id": i, "status": -1, "error": str(e)[:80]}
sink = open(OUT, "a")
with ThreadPoolExecutor(4) as ex:
    for n, rec in enumerate(ex.map(one, todo)):
        if rec is None: print("STOP: repeated refusals", flush=True); break
        sink.write(json.dumps(rec) + "\n"); sink.flush()
        if (n + 1) % 500 == 0: print(n + 1, "/", len(todo), flush=True)
print("THUMBS_DONE", flush=True)
