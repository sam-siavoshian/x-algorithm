"""The owner's private home-timeline scrape (October to December 2025), converted to the post schema.

How it was collected (Scraper.py, shipped next to the data): the `HomeTimeline` GraphQL endpoint ("For You") polled
by three logged-in accounts, promoted content off. Only top-level entries of type `Tweet` are kept, so conversation
modules (a post with its replies) never appear: the source has no replies. Each post id is saved once, at the moment
it is first served (`saved_at`), with the counts it had then. This is therefore a sample of posts X chose to
distribute, read while they were still being distributed: success-biased in who appears, immature in the counts.

What the file carries per post: `legacy.full_text` (long posts are cut at about 280 characters and end in a t.co
link), likes, view count, the author's follower, following, post and media counts and the blue-check flag at save
time, media URLs, and `post_image_description`, which the scraper reads from X's own response (the Grok image
annotation field), not from a captioning model of the owner's. No author id, no reply, repost, quote or bookmark
counts, no language, no bio, no quoted-post id. `is_viral`, `engagement_rate` and the hand-made text flags are
the owner's derived labels and are not read.

Two inputs are accepted: the converted `posts.parquet` (one row per post, `media_url_https` as a JSON string,
`creator_*` columns) and the raw JSONL (`creator` nested). Guards, checked on every read and written to the
returned report; a failed guard raises:

- ids: `id_str` must be a string of digits and is parsed with `int()`. It never goes through a float (a 19-digit
  id loses its low digits there; `tweet_id` in the file is exactly such a float column and is ignored).
- time: `created_at` must agree with the time encoded in the id to the second in at least 99% of rows.
- counts: likes <= views in at least 99% of rows.
- one row per post id (the newest reading is kept and the number of dropped rows reported).

The image description goes to `media[0].alt`, the schema's slot for a text description of the media, so every
consumer of the post schema gets it; `image_text()` returns it per post for rendering into model text.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import polars as pl

from xviral.schema import MEDIA_STRUCT, conform
from xviral.util.core import synthetic_author_id

log = logging.getLogger(__name__)

SOURCE = "home_scrape"
TWITTER_EPOCH_MS = 1288834974657
_NULL_TEXT = ("", "nan", "none", "null")


def media_type(url: str) -> str:
    """photo | video | gif from the URL path, which is all the scrape kept of the media entity."""
    if "tweet_video_thumb" in url:
        return "gif"
    if "video_thumb" in url or "/amplify_video" in url or "ext_tw_video" in url:
        return "video"
    return "photo"


def _media(urls: list[str] | None, desc: str | None) -> list[dict]:
    out = []
    for i, u in enumerate(urls or []):
        out.append({"type": media_type(u), "url": u, "video_url": None, "width": None, "height": None, "duration_ms": None,
                    "alt": desc if i == 0 else None})
    return out


def _raw_rows(path: Path) -> Iterator[dict]:
    """Raw JSONL -> flat records with the converted file's column names. Embedding vectors are skipped."""
    try:
        import orjson
        loads = orjson.loads
    except ImportError:  # pragma: no cover
        loads = json.loads
    keep = ("id_str", "text", "user_name", "screen_name", "likes", "impressions", "created_at", "saved_at", "post_image_description")
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = loads(line)
            except ValueError:
                continue
            r = {k: o.get(k) for k in keep}
            r["id_str"] = None if r["id_str"] is None else str(r["id_str"])
            r["media_url_https"] = json.dumps(o.get("media_url_https") or [])
            c = o.get("creator") or {}
            for k in ("is_blue_verified", "followers_count", "following_count", "posts_count", "media_count"):
                r[f"creator_{k}"] = c.get(k)
            yield r


def _load(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    schema = {"id_str": pl.String, "text": pl.String, "user_name": pl.String, "screen_name": pl.String, "likes": pl.Int64, "impressions": pl.String,
              "created_at": pl.String, "saved_at": pl.String, "post_image_description": pl.String, "media_url_https": pl.String,
              "creator_is_blue_verified": pl.Boolean, "creator_followers_count": pl.Int64, "creator_following_count": pl.Int64,
              "creator_posts_count": pl.Int64, "creator_media_count": pl.Int64}
    return pl.DataFrame(list(_raw_rows(path)), schema=schema)


def read(path: str | Path, strict: bool = True) -> tuple[pl.DataFrame, dict]:
    """The scrape in the post schema, plus the guard report. One row per post id."""
    raw = _load(path)
    n0 = raw.height
    rep: dict = {"rows_read": n0, "file": Path(path).name}
    if raw["id_str"].dtype != pl.String:
        raise ValueError(f"home_scrape: id_str is {raw['id_str'].dtype}, not a string; refusing to read ids that may have passed through a float")
    ok_id = raw["id_str"].str.contains(r"^\d{15,20}$").fill_null(False)
    rep["bad_ids"] = int((~ok_id).sum())
    raw = raw.filter(ok_id)
    d = raw.with_columns(
        pl.col("id_str").cast(pl.Int64).alias("post_id"),       # String -> Int64: exact, no float on the way
        pl.col("created_at").str.to_datetime("%a %b %d %H:%M:%S %z %Y", strict=False).dt.epoch("ms").alias("created_at_src_ms"),
        pl.col("saved_at").str.to_datetime(strict=False, time_zone="UTC").dt.epoch("ms").alias("snapshot_at_ms"),
        pl.col("impressions").cast(pl.Int64, strict=False).alias("views"),
        pl.col("likes").cast(pl.Int64, strict=False),
        pl.col("media_url_https").fill_null("[]").str.json_decode(pl.List(pl.String)).alias("_urls"),
        pl.when(pl.col("post_image_description").str.strip_chars().str.to_lowercase().is_in(_NULL_TEXT)).then(None)
          .otherwise(pl.col("post_image_description").str.strip_chars()).alias("_desc"),
    ).with_columns(((pl.col("post_id") // (1 << 22)) + TWITTER_EPOCH_MS).alias("created_at_ms"))
    assert (d["post_id"].cast(pl.String) == d["id_str"]).all(), "post ids changed on the way to Int64"

    agree = ((d["created_at_ms"] - d["created_at_src_ms"]).abs() <= 1000).fill_null(False)
    rep["timestamp_agrees_with_id"] = float(agree.mean())
    le = (d["likes"] <= d["views"]).fill_null(False)
    rep["likes_le_views"] = float(le.mean())
    rep["views_missing"] = int(d["views"].is_null().sum())
    rep["snapshot_missing"] = int(d["snapshot_at_ms"].is_null().sum())
    rep["snapshot_before_post"] = int((d["snapshot_at_ms"] < d["created_at_ms"]).sum())
    if strict and (rep["timestamp_agrees_with_id"] < 0.99 or rep["likes_le_views"] < 0.99):
        raise ValueError(f"home_scrape: guard failed: {rep}")

    d = d.sort("snapshot_at_ms").unique("post_id", keep="last", maintain_order=True)
    rep["duplicate_rows_dropped"] = n0 - rep["bad_ids"] - d.height
    media = [_media(u, t) for u, t in zip(d["_urls"].to_list(), d["_desc"].to_list())]
    is_rt = d["text"].fill_null("").str.starts_with("RT @")
    d = d.with_columns(
        pl.Series("media", media, dtype=pl.List(MEDIA_STRUCT)),
        pl.col("screen_name").alias("author_handle"),
        pl.col("screen_name").str.to_lowercase().map_elements(synthetic_author_id, return_dtype=pl.Int64).alias("author_id"),
        pl.col("user_name").alias("author_display_name"),
        pl.lit("exact").alias("snapshot_kind"),
        pl.when(is_rt).then(pl.lit("repost")).otherwise(pl.lit("original")).alias("kind"),   # replies are never collected; quotes cannot be told apart
        pl.lit(False).alias("is_reply"),
        pl.when(pl.col("views").is_null()).then(pl.lit("absent")).otherwise(pl.lit("count")).alias("views_state"),
        pl.col("creator_followers_count").alias("author_followers"), pl.col("creator_following_count").alias("author_following"),
        pl.col("creator_posts_count").alias("author_posts_count"), pl.col("creator_media_count").alias("author_media_count"),
        pl.col("creator_is_blue_verified").alias("author_verified"),
        pl.col("text").str.count_matches(r"(?:^|\s)@\w+").cast(pl.Int32).alias("mentions_n"),
        pl.col("text").str.count_matches(r"(?:^|\s)#\w+").cast(pl.Int32).alias("hashtags_n"),
        pl.lit(SOURCE).alias("source"), pl.lit(Path(path).name).alias("source_file"), pl.lit(False).alias("synthetic_id"),
    )
    rep["rows_out"] = d.height
    rep["accounts"] = int(d["author_handle"].str.to_lowercase().n_unique())
    rep["with_media"] = int((d["media"].list.len() > 0).sum())
    rep["with_image_description"] = int(d["_desc"].is_not_null().sum())
    rep["reposts"] = int(is_rt.sum())
    log.info("home_scrape: %s", rep)
    return conform(d), rep


def image_text(posts: pl.DataFrame) -> pl.DataFrame:
    """post_id, image_desc: the description of the first media item, for posts that have one."""
    return posts.select("post_id", pl.col("media").list.eval(pl.element().struct.field("alt")).list.drop_nulls().list.first().alias("image_desc")) \
        .filter(pl.col("image_desc").is_not_null())


def text_with_image(text: str | None, desc: str | None) -> str:
    """The post's text with the picture written out, as the text encoder should see a media post."""
    return (text or "") if not desc else f"{text or ''}\n[image: {desc}]"
