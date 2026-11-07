"""The 2025 tech scrape (JSONL), whose layout is not fixed.

Fields are located by trying candidate key paths on a sample of lines and
keeping the first path present in at least half of them. The chosen mapping
is logged and written next to the output so it can be checked and pinned with
an override. The three 768-d embeddings are written to a separate table: they
feed the phase-0 baseline and an ablation, never the surrogate.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from xviral.schema import MEDIA_STRUCT, URL_STRUCT, conform
from xviral.text.urls import is_media_link, registrable_domain
from xviral.util.core import iso_to_ms, snowflake_ms, synthetic_author_id, synthetic_post_id, to_int, write_parquet

try:
    import orjson as _json

    _loads = _json.loads
except ImportError:  # pragma: no cover
    _loads = json.loads

log = logging.getLogger(__name__)

CANDIDATES: dict[str, list[str]] = {
    "post_id": ["id_str", "id", "tweet_id", "rest_id", "tweet.id_str", "tweet.id", "tweet.rest_id", "legacy.id_str"],
    "text": ["text", "full_text", "rawContent", "tweet.text", "tweet.full_text", "legacy.full_text", "note_tweet.text"],
    "created_at": ["created_at", "tweet.created_at", "legacy.created_at", "date", "timestamp"],
    "likes": ["likes", "favorite_count", "like_count", "likeCount", "tweet.likes", "tweet.favorite_count", "legacy.favorite_count", "public_metrics.like_count"],
    "views": ["impressions", "views", "view_count", "viewCount", "impression_count", "tweet.impressions", "tweet.views", "views.count", "public_metrics.impression_count"],
    "replies": ["replies", "reply_count", "replyCount", "tweet.reply_count", "legacy.reply_count", "public_metrics.reply_count"],
    "reposts": ["retweets", "reposts", "retweet_count", "retweetCount", "tweet.retweet_count", "legacy.retweet_count", "public_metrics.retweet_count"],
    "quotes": ["quotes", "quote_count", "quoteCount", "tweet.quote_count", "legacy.quote_count", "public_metrics.quote_count"],
    "bookmarks": ["bookmarks", "bookmark_count", "bookmarkCount", "tweet.bookmark_count", "legacy.bookmark_count"],
    "media": ["media_url_https", "media", "tweet.media_url_https", "tweet.media", "entities.media", "extended_entities.media", "legacy.extended_entities.media"],
    "urls": ["urls", "entities.urls", "tweet.entities.urls", "legacy.entities.urls"],
    "handle": ["screen_name", "author.screen_name", "user.screen_name", "author.username", "user.username", "username", "author_handle"],
    "author_id": ["author.id_str", "user.id_str", "author.id", "user.id", "author_id", "user_id"],
    "followers": ["followers_count", "author.followers_count", "user.followers_count", "author.followers"],
    "following": ["following_count", "friends_count", "author.following_count", "author.friends_count", "user.friends_count"],
    "posts_count": ["posts_count", "statuses_count", "author.posts_count", "author.statuses_count", "user.statuses_count"],
    "media_count": ["media_count", "author.media_count", "user.media_count"],
    "verified": ["is_blue_verified", "author.is_blue_verified", "user.is_blue_verified", "verified", "author.verified"],
    "author_created_at": ["author.created_at", "user.created_at", "account_created_at"],
    "in_reply_to": ["in_reply_to_status_id_str", "in_reply_to_status_id", "tweet.in_reply_to_status_id_str", "legacy.in_reply_to_status_id_str"],
    "quoted_id": ["quoted_status_id_str", "quoted_status_id", "tweet.quoted_status_id_str", "legacy.quoted_status_id_str"],
    "is_quote": ["is_quote_status", "tweet.is_quote_status", "legacy.is_quote_status"],
    "is_retweet": ["retweeted_status", "is_retweet", "tweet.retweeted_status"],
    "lang": ["lang", "tweet.lang", "legacy.lang"],
    "scraped_at": ["scraped_at", "collected_at", "fetched_at", "scrape_time", "crawled_at"],
    "is_viral": ["is_viral", "label.is_viral", "label"],
}
EMBEDDING_KEYS = ["text_embedding", "img_embedding", "img_description_embedding"]
_MISSING = object()


def get_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def detect_fields(lines: list[dict], overrides: dict[str, str] | None = None) -> dict[str, str]:
    overrides = overrides or {}
    mapping: dict[str, str] = {}
    n = max(len(lines), 1)
    for field, cands in CANDIDATES.items():
        if field in overrides:
            mapping[field] = overrides[field]
            continue
        for path in cands:
            hits = sum(1 for o in lines if get_path(o, path) not in (_MISSING, None))
            if hits / n >= 0.5:
                mapping[field] = path
                break
    return mapping


def _media_from(value: Any) -> list[dict]:
    if value in (_MISSING, None, "", []):
        return []
    items = value if isinstance(value, list) else [value]
    out = []
    for it in items:
        if isinstance(it, str):
            url, mtype, dur, vurl, w, h = it, None, None, None, None, None
        elif isinstance(it, dict):
            url = it.get("media_url_https") or it.get("url") or it.get("media_url")
            mtype = it.get("type")
            vi = it.get("video_info") or {}
            dur = vi.get("duration_millis") or it.get("duration_ms")
            mp4s = sorted((v for v in vi.get("variants", []) if v.get("content_type") == "video/mp4"),
                          key=lambda v: abs((v.get("bitrate") or 0) - 832_000))
            vurl = mp4s[0]["url"] if mp4s else None
            info = it.get("original_info") or {}
            w, h = info.get("width"), info.get("height")
        else:
            continue
        if not url:
            continue
        if mtype is None:
            mtype = "gif" if "tweet_video_thumb" in url else ("video" if "video_thumb" in url else "photo")
        mtype = {"animated_gif": "gif"}.get(mtype, mtype)
        out.append({"type": mtype, "url": url, "video_url": vurl, "width": w, "height": h, "duration_ms": to_int(dur), "alt": None})
    return out


def _urls_from(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    out = []
    for u in value:
        if isinstance(u, dict):
            exp = u.get("expanded_url") or u.get("expanded") or u.get("url")
            out.append({"tco": u.get("url"), "expanded": exp, "final_domain": registrable_domain(exp), "is_media": is_media_link(exp)})
        elif isinstance(u, str):
            out.append({"tco": None, "expanded": u, "final_domain": registrable_domain(u), "is_media": is_media_link(u)})
    return out


def _iter_lines(path: Path) -> Iterator[dict]:
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield _loads(line)
            except ValueError:
                continue


def _row(obj: dict, mp: dict[str, str], file_ms: int, source_file: str) -> tuple[dict, dict | None]:
    g = lambda f: (get_path(obj, mp[f]) if f in mp else _MISSING)  # noqa: E731
    v = lambda f: (None if (x := g(f)) is _MISSING else x)  # noqa: E731
    created_src = v("created_at")
    created_src_ms = iso_to_ms(created_src) if isinstance(created_src, str) else to_int(created_src)
    if created_src_ms is not None and created_src_ms < 10**11:  # seconds, not ms
        created_src_ms *= 1000
    pid = to_int(v("post_id"))
    handle = v("handle")
    text = v("text")
    synthetic = pid is None
    if synthetic:
        pid = synthetic_post_id(handle, created_src_ms, text)
    created_ms = snowflake_ms(pid) if not synthetic else created_src_ms
    aid = to_int(v("author_id"))
    if aid is None and handle:
        aid = synthetic_author_id(handle)
    scraped = v("scraped_at")
    snap_ms = (iso_to_ms(scraped) if isinstance(scraped, str) else to_int(scraped)) if scraped is not None else None
    in_reply = to_int(v("in_reply_to"))
    quoted = to_int(v("quoted_id"))
    is_rt = v("is_retweet") not in (None, False, 0) or (isinstance(text, str) and text.startswith("RT @"))
    kind = "repost" if is_rt else "reply" if in_reply else "quote" if (quoted or v("is_quote") is True) else "original"
    av = v("author_created_at")
    row = {
        "post_id": pid, "author_id": aid, "author_handle": handle,
        "created_at_ms": created_ms, "created_at_src_ms": created_src_ms,
        "snapshot_at_ms": snap_ms or file_ms, "snapshot_kind": "exact" if snap_ms else "file_date",
        "text": text, "lang_src": v("lang"), "kind": kind, "in_reply_to_post_id": in_reply,
        "is_reply": kind == "reply", "quoted_post_id": quoted,
        "media": _media_from(g("media")), "urls": _urls_from(v("urls")),
        "views": to_int(v("views")), "likes": to_int(v("likes")), "replies": to_int(v("replies")),
        "reposts": to_int(v("reposts")), "quotes": to_int(v("quotes")), "bookmarks": to_int(v("bookmarks")),
        "author_followers": to_int(v("followers")), "author_following": to_int(v("following")),
        "author_posts_count": to_int(v("posts_count")), "author_media_count": to_int(v("media_count")),
        "author_verified": bool(v("verified")) if v("verified") is not None else None,
        "author_created_at_ms": iso_to_ms(av) if isinstance(av, str) else to_int(av),
        "source": "user_scrape", "source_file": source_file, "synthetic_id": synthetic,
        "views_state": "count" if v("views") is not None else "absent",
        "is_viral_src": v("is_viral"),
    }
    emb = None
    present = {k: obj.get(k) for k in EMBEDDING_KEYS if isinstance(obj.get(k), list)}
    if present:
        emb = {"post_id": pid, **present}
    return row, emb


_ROW_SCHEMA = {
    "media": pl.List(MEDIA_STRUCT),
    "urls": pl.List(URL_STRUCT),
}


def ingest(paths: list[str | Path], out_dir: str | Path, overrides: dict[str, str] | None = None,
           batch_rows: int = 50_000, sample_lines: int = 2000) -> dict[str, str]:
    out_dir = Path(out_dir)
    sample: list[dict] = []
    for p in paths:
        for obj in _iter_lines(Path(p)):
            sample.append(obj)
            if len(sample) >= sample_lines:
                break
        if len(sample) >= sample_lines:
            break
    mapping = detect_fields(sample, overrides)
    (out_dir / "field_mapping.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "field_mapping.json").write_text(json.dumps(mapping, indent=2))
    log.info("user_scrape field mapping: %s", mapping)
    for required in ("text", "created_at"):
        if required not in mapping:
            raise ValueError(f"user_scrape: could not locate '{required}'; pass an override")
    part = 0
    for p in paths:
        p = Path(p)
        file_ms = int(os.path.getmtime(p) * 1000)
        rows, embs = [], []

        def flush():
            nonlocal part
            if rows:
                df = pl.DataFrame(rows, schema_overrides=_ROW_SCHEMA, infer_schema_length=None)
                viral = df.select("post_id", pl.col("is_viral_src").cast(pl.String).alias("is_viral_src"))
                write_parquet(conform(df), out_dir / "posts" / f"part-{part:05d}.parquet")
                write_parquet(viral, out_dir / "is_viral" / f"part-{part:05d}.parquet")
            if embs:
                e = pl.DataFrame(embs)
                e = e.with_columns([pl.col(k).cast(pl.List(pl.Float32)) for k in EMBEDDING_KEYS if k in e.columns])
                write_parquet(e, out_dir / "old_embeddings" / f"part-{part:05d}.parquet")
            part += 1
            rows.clear()
            embs.clear()

        for obj in _iter_lines(p):
            r, e = _row(obj, mapping, file_ms, p.name)
            rows.append(r)
            if e:
                embs.append(e)
            if len(rows) >= batch_rows:
                flush()
        flush()
    return mapping
