#!/usr/bin/env python3
"""Fetch the first image of selected corpus posts from pbs.twimg.com (the 680 px `small` rendition), at most 5 requests
a second, resumable, stopping on repeated 403/429.

    python scripts/images/fetch.py --selection data/images/selection.parquet --out data/images/raw --minutes 90

Selection columns: post_id, media_idx, url, priority (lower first). Writes shards <out>/img-NNNNN.parquet
(post_id, media_idx, url, status, bytes, jpeg) and nothing else; build the index with scripts/images/index.py.
"""
import argparse, glob, io, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import polars as pl, requests
from xviral.media.fetch import RateLimiter, image_url
ap = argparse.ArgumentParser()
ap.add_argument("--selection", required=True); ap.add_argument("--out", required=True); ap.add_argument("--minutes", type=float, default=90)
ap.add_argument("--rps", type=float, default=5.0); ap.add_argument("--shard", type=int, default=1000); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
S = pl.read_parquet(a.selection).sort("priority", "order")
done = set()
for f in glob.glob(f"{a.out}/img-*.parquet"):
    d = pl.read_parquet(f, columns=["post_id", "media_idx"]); done |= set(zip(d["post_id"].to_list(), d["media_idx"].to_list()))
todo = [r for r in S.select("post_id", "media_idx", "url").iter_rows() if (r[0], r[1]) not in done]
if a.limit: todo = todo[:a.limit]
print(f"{len(todo)} to fetch, {len(done)} done", flush=True)
lim = RateLimiter(a.rps); tl = threading.local(); bad = [0]; stop = threading.Event()
def sess():
    if not hasattr(tl, "s"):
        tl.s = requests.Session(); tl.s.headers["User-Agent"] = "Mozilla/5.0 (compatible; research-fetch)"
    return tl.s
def one(row):
    pid, idx, url = row
    if stop.is_set(): return None
    lim.wait()
    try:
        r = sess().get(image_url(url), timeout=20)
        if r.status_code in (403, 429):
            bad[0] += 1
            if bad[0] >= 25: stop.set()
        elif r.status_code == 200: bad[0] = 0
        return {"post_id": pid, "media_idx": idx, "url": url, "status": r.status_code, "bytes": len(r.content) if r.status_code == 200 else None, "jpeg": r.content if r.status_code == 200 else None}
    except Exception:  # noqa: BLE001
        return {"post_id": pid, "media_idx": idx, "url": url, "status": -1, "bytes": None, "jpeg": None}
schema = {"post_id": pl.Int64, "media_idx": pl.Int64, "url": pl.String, "status": pl.Int32, "bytes": pl.Int64, "jpeg": pl.Binary}
t0 = time.time(); shard = len(glob.glob(f"{a.out}/img-*.parquet")); n = 0
with ThreadPoolExecutor(8) as ex:
    for s in range(0, len(todo), a.shard):
        if stop.is_set() or time.time() - t0 > a.minutes * 60: break
        rows = [r for r in ex.map(one, todo[s:s + a.shard]) if r is not None]
        if rows: pl.DataFrame(rows, schema=schema).write_parquet(f"{a.out}/img-{shard:05d}.parquet"); shard += 1; n += len(rows)
        ok = sum(r["status"] == 200 for r in rows)
        print(f"{n}/{len(todo)} fetched, last shard ok {ok}/{len(rows)}, {time.time()-t0:.0f}s", flush=True)
print("STOPPED on repeated 403/429" if stop.is_set() else "FETCH_DONE", flush=True)
