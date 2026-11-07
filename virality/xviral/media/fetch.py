"""Bounded, rate-limited media fetching (DATA_PIPELINE.md section 5).

Media bytes are fetched once, resized, embedded, and then only embeddings and
sampled frames are kept. Images are stored in Parquet shards (post_id,
media_idx, jpeg bytes) rather than millions of small files. The fetch is
resumable: completed (post_id, media_idx) pairs are skipped.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import requests

from xviral.config import MediaCfg
from xviral.util.core import scan_dir, write_parquet

log = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, per_second: float):
        self.interval = 1.0 / max(per_second, 1e-6)
        self.lock = threading.Lock()
        self.next = time.monotonic()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if self.next > now:
                time.sleep(self.next - now)
            self.next = max(self.next, now) + self.interval


def resize_jpeg(data: bytes, side: int, quality: int) -> tuple[bytes, int, int]:
    from PIL import Image

    im = Image.open(io.BytesIO(data))
    im = im.convert("RGB")
    w, h = im.size
    scale = side / max(w, h)
    if scale < 1:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), w, h


def image_url(url: str, size: str = "small") -> str:
    """pbs.twimg.com serves sized variants; 'small' is 680 px on the long side, enough for 448."""
    if "pbs.twimg.com/media/" in url and "?" not in url:
        base, _, ext = url.rpartition(".")
        return f"{base}?format={ext}&name={size}"
    return url


def select_media(posts: pl.DataFrame, cfg: MediaCfg) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Choose which images and videos to fetch: newest first, spread across follower tiers."""
    m = posts.select("post_id", "created_at_ms", "follower_tier", "media").explode("media").filter(pl.col("media").is_not_null())
    m = m.with_columns(pl.int_range(pl.len()).over("post_id").alias("media_idx"), pl.col("media").struct.unnest()).drop("media")

    def take(df: pl.DataFrame, n: int) -> pl.DataFrame:
        tiers = df["follower_tier"].n_unique() or 1
        per = max(n // tiers, 1)
        out = df.sort("created_at_ms", descending=True).group_by("follower_tier").head(per)
        return out.head(n)

    imgs = take(m.filter(pl.col("type").is_in(["photo", "gif"]) & pl.col("url").is_not_null()), cfg.max_images)
    vids = take(m.filter((pl.col("type") == "video") & pl.col("video_url").is_not_null()), cfg.max_videos)
    return imgs, vids


def fetch_images(items: pl.DataFrame, out_dir: str | Path, cfg: MediaCfg, shard_rows: int = 2000,
                 session: requests.Session | None = None) -> None:
    out_dir = Path(out_dir)
    done = set()
    if out_dir.exists() and any(out_dir.glob("*.parquet")):
        d = scan_dir(out_dir).select("post_id", "media_idx").collect()
        done = set(zip(d["post_id"].to_list(), d["media_idx"].to_list()))
    todo = [r for r in items.select("post_id", "media_idx", "url").iter_rows() if (r[0], r[1]) not in done]
    log.info("images: %d to fetch (%d already done)", len(todo), len(done))
    limiter = RateLimiter(cfg.requests_per_second)
    s = session or requests.Session()
    s.headers["User-Agent"] = cfg.user_agent
    shard = len(list(out_dir.glob("*.parquet"))) if out_dir.exists() else 0

    def one(row):
        pid, idx, url = row
        limiter.wait()
        try:
            r = s.get(image_url(url), timeout=cfg.timeout_s)
            if r.status_code != 200:
                return {"post_id": pid, "media_idx": idx, "status": r.status_code, "jpeg": None, "orig_w": None, "orig_h": None}
            jpg, w, h = resize_jpeg(r.content, cfg.vlm_side, cfg.jpeg_quality)
            return {"post_id": pid, "media_idx": idx, "status": 200, "jpeg": jpg, "orig_w": w, "orig_h": h}
        except Exception as e:  # noqa: BLE001 - network and decode errors are recorded, not raised
            log.debug("image %s failed: %s", url, e)
            return {"post_id": pid, "media_idx": idx, "status": -1, "jpeg": None, "orig_w": None, "orig_h": None}

    schema = {"post_id": pl.Int64, "media_idx": pl.Int64, "status": pl.Int32, "jpeg": pl.Binary, "orig_w": pl.Int32, "orig_h": pl.Int32}
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        for start in range(0, len(todo), shard_rows):
            rows = list(ex.map(one, todo[start:start + shard_rows]))
            write_parquet(pl.DataFrame(rows, schema=schema), out_dir / f"images-{shard:05d}.parquet")
            shard += 1
            log.info("images: %d/%d", min(start + shard_rows, len(todo)), len(todo))
