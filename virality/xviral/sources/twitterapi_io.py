"""Saved twitterapi.io result pages (the `user/last_tweets` and search endpoints), as published in Zenodo 21783628.

Zenodo 21783628 ("Beyond Luxury SMMA", published 2026-08-04, no licence stated): 193 official X accounts of 50 Chinese
global brands (Xiaomi, OnePlus, Anker, Temu, TikTok, BYD, Insta360, ...), every post from November 2020 to November
2025, collected with the twitterapi.io API by the authors. One JSON file per account, `<handle>_<user id>.json`: a list
of response pages, each `{"data": {"tweets": [...]}}`.

It is the richest public format found: view counts (posts from 2022-12 on), bookmarks, quotes, reply and conversation
ids, the quoted post, media with dimensions, video duration and mp4 variants, and a full author object with follower
count and bio (`author.profile_bio.description`; `author.description` is empty). The same reader serves any other
twitterapi.io dump, including an owner's own scrape made with that API.

Checked on the full archive (scripts/data_track/brands_run.py):
- `author.id` equals the id in the file name and is the real X user id; `id` is the post id, and `createdAt` agrees
  with the time in the id to the second.
- The author object is the account at scrape time, repeated on every post: follower counts are one snapshot, not
  the count when the post was made.
- `viewCount` is 0 or missing for posts before view counts existed, and partial for posts made shortly before
  (likes exceed views there); posts created before 2023-01-01 get views null and state "absent".
  Guard: where views exist, likes <= views must hold in at least 99% of rows of a file, else the file is refused.
- There is no scrape time in the files. The archive's newest post is the snapshot time (`snapshot`), which makes the
  last days immature for the maturity filter and everything older mature.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from xviral.schema import POSTS_SCHEMA, conform
from xviral.sources.user_scrape import _media_from, _urls_from
from xviral.util.core import date_to_ms, snowflake_ms, to_int

log = logging.getLogger(__name__)

SOURCE = "twitterapi_io"
VIEWS_START_MS = date_to_ms("2023-01-01")   # view counts became public in mid-December 2022; posts from the weeks before carry partial counts


def _ms(s: str | None) -> int | None:
    if not s:
        return None
    try:
        return int(datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").timestamp() * 1000)
    except ValueError:
        return None


def _id(x) -> int | None:
    """Ids arrive as decimal strings. They must not pass through a float (`to_int` does for strings), which would
    change the low bits of a 19-digit id."""
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    if isinstance(x, str) and x.strip().lstrip("-").isdigit():
        return int(x)
    return None


def iter_tweets(path: Path):
    obj = json.loads(Path(path).read_text())
    pages = obj if isinstance(obj, list) else [obj]
    for p in pages:
        d = p.get("data") if isinstance(p, dict) else None
        tweets = (d.get("tweets") if isinstance(d, dict) else d) or (p.get("tweets") if isinstance(p, dict) else None) or []
        yield from tweets


def _row(t: dict, snapshot_ms: int, source_file: str) -> tuple[dict, dict]:
    a = t.get("author") or {}
    pid = _id(t.get("id"))
    created = snowflake_ms(pid)
    q, r = t.get("quoted_tweet") or None, t.get("retweeted_tweet") or None
    kind = "repost" if r else "reply" if t.get("isReply") else "quote" if q else "original"
    views = to_int(t.get("viewCount"))
    has_views = views is not None and (created or 0) >= VIEWS_START_MS
    ent = t.get("entities") or {}
    bio = ((a.get("profile_bio") or {}).get("description") or a.get("description") or "").strip() or None
    row = {
        "post_id": pid, "author_id": _id(a.get("id")), "author_handle": a.get("userName"), "created_at_ms": created, "created_at_src_ms": _ms(t.get("createdAt")),
        "snapshot_at_ms": snapshot_ms, "snapshot_kind": "file_date", "text": t.get("text"), "lang_src": t.get("lang"), "kind": kind,
        "in_reply_to_post_id": _id(t.get("inReplyToId")), "is_reply": bool(t.get("isReply")), "quoted_post_id": _id(q.get("id")) if q else None,
        "quoted_text": q.get("text") if q else None, "quoted_author_id": _id((q.get("author") or {}).get("id")) if q else None,
        "reposted_post_id": _id(r.get("id")) if r else None, "conversation_id": _id(t.get("conversationId")),
        "media": _media_from((t.get("extendedEntities") or {}).get("media")), "urls": _urls_from(ent.get("urls")),
        "mentions_n": len(ent.get("user_mentions") or []), "hashtags_n": len(ent.get("hashtags") or []),
        "views": views if has_views else None, "views_state": "count" if has_views else "absent",
        "likes": to_int(t.get("likeCount")), "replies": to_int(t.get("replyCount")), "reposts": to_int(t.get("retweetCount")), "quotes": to_int(t.get("quoteCount")),
        "bookmarks": to_int(t.get("bookmarkCount")),
        "author_followers": to_int(a.get("followers")), "author_following": to_int(a.get("following")), "author_posts_count": to_int(a.get("statusesCount")),
        "author_media_count": to_int(a.get("mediaCount")), "author_created_at_ms": _ms(a.get("createdAt")), "author_verified": bool(a.get("isBlueVerified") or a.get("isVerified")),
        "author_display_name": a.get("name"), "author_bio": bio, "source": SOURCE, "source_file": source_file, "synthetic_id": False,
    }
    # What the post schema has no column for, kept beside it: who was replied to, so self-replies (threads, "link in the reply") can be told apart.
    side = {"post_id": pid, "author_id": row["author_id"], "conversation_id": row["conversation_id"], "in_reply_to_post_id": row["in_reply_to_post_id"],
            "in_reply_to_user_id": _id(t.get("inReplyToUserId")), "in_reply_to_handle": t.get("inReplyToUsername"),
            "is_self_reply": bool(t.get("isReply")) and _id(t.get("inReplyToUserId")) == row["author_id"], "has_card": bool(t.get("card")), "has_article": bool(t.get("article")),
            "verified_type": a.get("verifiedType"), "source_app": t.get("source")}
    return row, side


SIDE_SCHEMA = {"post_id": pl.Int64, "author_id": pl.Int64, "conversation_id": pl.Int64, "in_reply_to_post_id": pl.Int64, "in_reply_to_user_id": pl.Int64, "in_reply_to_handle": pl.String,
               "is_self_reply": pl.Boolean, "has_card": pl.Boolean, "has_article": pl.Boolean, "verified_type": pl.String, "source_app": pl.String}


def read_file(path: str | Path, snapshot: str = "2025-11-08", expect_author_id: int | None = None) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """One account file -> (posts in the unified schema, thread side table, check report). Raises when a guard fails."""
    path = Path(path)
    snap = date_to_ms(snapshot)
    rows, sides, seen = [], [], set()
    n_raw = 0
    for t in iter_tweets(path):
        n_raw += 1
        if not isinstance(t, dict) or t.get("type") not in (None, "tweet"):
            continue
        row, side = _row(t, snap, path.name)
        if row["post_id"] is None or row["post_id"] in seen:
            continue
        seen.add(row["post_id"]); rows.append(row); sides.append(side)
    df = pl.DataFrame(rows, schema_overrides={k: v for k, v in POSTS_SCHEMA.items() if k in (rows[0] if rows else {})}, infer_schema_length=None) if rows else pl.DataFrame()
    rep = {"file": path.name, "raw": n_raw, "unique": len(rows)}
    if not rows:
        return conform(pl.DataFrame({"post_id": []})), pl.DataFrame(schema=SIDE_SCHEMA), rep
    bad_time = df.filter(pl.col("created_at_ms").is_null() | ((pl.col("created_at_ms") - pl.col("created_at_src_ms")).abs() > 1000)).height
    v = df.filter(pl.col("views").is_not_null())
    rep.update(time_mismatch=bad_time, authors=df["author_id"].n_unique(), with_views=v.height, likes_le_views=float((v["likes"] <= v["views"]).mean()) if v.height else None,
               after_snapshot=df.filter(pl.col("created_at_ms") > snap).height)
    if v.height >= 200 and rep["likes_le_views"] < 0.99:
        raise ValueError(f"{path.name}: likes <= views in only {rep['likes_le_views']:.1%} of rows with views; the count fields are not what this reader expects")
    if expect_author_id is not None and (df["author_id"] != expect_author_id).mean() > 0.01:
        raise ValueError(f"{path.name}: author id in the rows differs from the file name's ({expect_author_id})")
    df = df.filter(pl.col("created_at_ms").is_not_null() & ((pl.col("created_at_ms") - pl.col("created_at_src_ms")).abs() <= 1000))
    side = pl.DataFrame(sides, schema=SIDE_SCHEMA).join(df.select("post_id"), on="post_id", how="semi")
    return conform(df), side, rep


def ingest(in_dir: str | Path, out_dir: str | Path, snapshot: str = "2025-11-08") -> list[dict]:
    out = Path(out_dir); (out / "posts").mkdir(parents=True, exist_ok=True); (out / "threads").mkdir(parents=True, exist_ok=True)
    reports = []
    for f in sorted(Path(in_dir).glob("*.json")):
        tail = f.stem.rsplit("_", 1)[-1]
        posts, side, rep = read_file(f, snapshot, int(tail) if tail.isdigit() else None)
        posts.write_parquet(out / "posts" / f"{f.stem}.parquet", compression="zstd"); side.write_parquet(out / "threads" / f"{f.stem}.parquet", compression="zstd")
        reports.append(rep)
    return reports
