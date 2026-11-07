"""Community Archive (github.com/TheExGenesis/community-archive): donated X archives.

Two access paths, both producing one Parquet directory per table:
1. REST (PostgREST over Supabase), paginated by keyset. The anon key is public
   by design and documented in the archive's docs/api-doc.md; it is read from
   an environment variable rather than stored in this repository.
2. The bulk consent-safe export (tweets.parquet + profiles.parquet) when the
   archive's latest.json pointer is live.

Tables (REST, verified 2026-11-07):
  tweets        tweet_id, account_id, created_at, full_text, retweet_count, favorite_count,
                reply_to_tweet_id, reply_to_user_id, reply_to_username, archive_upload_id
  all_account   account_id, username, created_at, account_display_name, num_tweets,
                num_following, num_followers, num_likes, updated_at, is_tombstone
  profile       account_id, bio, website, location, avatar_media_url, header_media_url
  tweet_media   media_id, tweet_id, media_url, media_type, width, height
  tweet_urls    id, url, expanded_url, display_url, tweet_id
  quote_tweets  tweet_id, quoted_tweet_id
  followers     account_id, follower_account_id
  following     account_id, following_account_id
  likes         account_id, liked_tweet_id   (the donor's own likes: per-viewer positives)
  liked_tweets  tweet_id, full_text, author_account_id, is_tombstone
  archive_upload id, ...  (474 donor uploads)

Counts are likes and reposts at archive time. There are no views. Donors are
the viewers: their likes and follow lists are the only per-viewer engagement
data available outside X, and they feed the viewer-level research track.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import polars as pl
import requests

from xviral.config import CommunityArchiveCfg
from xviral.schema import MEDIA_STRUCT, URL_STRUCT, conform
from xviral.text.urls import is_media_link, registrable_domain
from xviral.util.core import scan_dir, snowflake_ms_expr, write_parquet

log = logging.getLogger(__name__)

# Keyset column per table. Text keys order lexicographically; PostgREST compares the same way,
# so `gt.<last>` pagination is consistent.
KEYSET = {
    "tweets": "tweet_id",
    "all_account": "account_id",
    "profile": "account_id",
    "tweet_media": "media_id",
    "tweet_urls": "id",
    "quote_tweets": "tweet_id",
    "followers": "id",
    "following": "id",
    "likes": "id",
    "liked_tweets": "tweet_id",
    "archive_upload": "id",
}

SELECT = {
    "tweets": "tweet_id,account_id,created_at,full_text,retweet_count,favorite_count,reply_to_tweet_id,reply_to_user_id,archive_upload_id",
    "all_account": "account_id,username,created_at,account_display_name,num_tweets,num_following,num_followers,num_likes,updated_at,is_tombstone",
    "profile": "account_id,bio,website,location",
    "tweet_media": "media_id,tweet_id,media_url,media_type,width,height",
    "tweet_urls": "id,url,expanded_url,tweet_id",
    "quote_tweets": "tweet_id,quoted_tweet_id",
    "followers": "id,account_id,follower_account_id",
    "following": "id,account_id,following_account_id",
    "likes": "id,account_id,liked_tweet_id",
    "liked_tweets": "tweet_id,full_text,author_account_id,is_tombstone",
    "archive_upload": "*",
}


class RestDownloader:
    """Resumable keyset paginator. State per table lives in <out>/<table>/_state.json."""

    def __init__(self, cfg: CommunityArchiveCfg, out_dir: str | Path, session: requests.Session | None = None):
        key = os.environ.get(cfg.anon_key_env)
        if not key:
            raise RuntimeError(
                f"Set ${cfg.anon_key_env} to the Community Archive public anon key "
                "(published in its docs/api-doc.md)."
            )
        self.cfg = cfg
        self.out = Path(out_dir)
        self.s = session or requests.Session()
        self.s.headers.update({"apikey": key, "Authorization": f"Bearer {key}"})
        self.min_interval = 1.0 / cfg.requests_per_second
        self._last = 0.0

    def _get(self, table: str, params: dict) -> list[dict]:
        for attempt in range(6):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.s.get(f"{self.cfg.rest_url}/{table}", params=params, timeout=60)
            except requests.RequestException as e:
                log.warning("CA %s request error %s; retry %d", table, e, attempt)
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"CA {table}: giving up after retries")

    def download(self, table: str, flush_rows: int = 100_000, max_rows: int | None = None) -> int:
        """Keyset pagination. `tweets` pages by created_at (indexed; combining a date filter with
        ordering by tweet_id times out on the live API, verified 2026-11-07), using >= on the last
        timestamp and dropping tweets already seen at that timestamp. Other tables page by their key."""
        by_time = table == "tweets"
        key = "created_at" if by_time else KEYSET[table]
        tdir = self.out / table
        tdir.mkdir(parents=True, exist_ok=True)
        state_path = tdir / "_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"last": None, "part": 0, "rows": 0, "done": False, "seen_at_last": []}
        if state["done"]:
            return state["rows"]
        buf: list[dict] = []

        def flush():
            if not buf:
                return
            write_parquet(pl.DataFrame(buf, infer_schema_length=None), tdir / f"part-{state['part']:05d}.parquet")
            state["part"] += 1
            buf.clear()
            state_path.write_text(json.dumps(state))

        while True:
            params = {"select": SELECT[table], "order": f"{key}.asc", "limit": str(self.cfg.page_size)}
            if by_time:
                start = state["last"] or f"{self.cfg.min_created}T00:00:00+00:00"
                params[key] = f"gte.{start}"
            elif state["last"] is not None:
                params[key] = f"gt.{state['last']}"
            rows = self._get(table, params)
            n_page = len(rows)
            if by_time and state["last"] is not None:
                seen = set(state["seen_at_last"])
                rows = [r for r in rows if not (r["created_at"] == state["last"] and r["tweet_id"] in seen)]
            if not rows:
                if n_page == self.cfg.page_size and by_time:
                    raise RuntimeError(f"CA {table}: a full page shares one timestamp; increase page_size")
                break
            buf.extend(rows)
            new_last = rows[-1][key]
            if by_time:
                at_last = [r["tweet_id"] for r in rows if r["created_at"] == new_last]
                state["seen_at_last"] = (state["seen_at_last"] + at_last) if new_last == state["last"] else at_last
            state["last"] = new_last
            state["rows"] += len(rows)
            if len(buf) >= flush_rows:
                flush()
            if max_rows and state["rows"] >= max_rows:
                break
            if n_page < self.cfg.page_size:
                break
        flush()
        state["done"] = not (max_rows and state["rows"] >= max_rows)
        state_path.write_text(json.dumps(state))
        log.info("CA %s: %d rows", table, state["rows"])
        return state["rows"]


def _read(ca_dir: Path, table: str) -> pl.DataFrame | None:
    d = ca_dir / table
    if not d.exists() or not any(d.glob("*.parquet")):
        return None
    return scan_dir(d).collect()


def _i64(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.String).cast(pl.Int64, strict=False)


def _ts(col: str) -> pl.Expr:
    return pl.col(col).str.to_datetime(time_zone="UTC", strict=False).dt.epoch("ms")


def to_unified(ca_dir: str | Path) -> dict[str, pl.DataFrame]:
    """Join the CA tables into unified posts, accounts and follow edges."""
    ca_dir = Path(ca_dir)
    tweets = _read(ca_dir, "tweets")
    if tweets is None:
        raise FileNotFoundError(f"no tweets table under {ca_dir}")
    acct = _read(ca_dir, "all_account")
    prof = _read(ca_dir, "profile")
    media = _read(ca_dir, "tweet_media")
    urls = _read(ca_dir, "tweet_urls")
    quotes = _read(ca_dir, "quote_tweets")

    t = tweets.select(
        _i64("tweet_id").alias("post_id"),
        _i64("account_id").alias("author_id"),
        _ts("created_at").alias("created_at_src_ms"),
        pl.col("full_text").alias("text"),
        pl.col("favorite_count").cast(pl.Int64, strict=False).alias("likes"),
        pl.col("retweet_count").cast(pl.Int64, strict=False).alias("reposts"),
        _i64("reply_to_tweet_id").alias("in_reply_to_post_id"),
    ).with_columns(snowflake_ms_expr("post_id").alias("created_at_ms"))

    if quotes is not None:
        q = quotes.select(_i64("tweet_id").alias("post_id"), _i64("quoted_tweet_id").alias("quoted_post_id")).unique("post_id")
        t = t.join(q, on="post_id", how="left")
    else:
        t = t.with_columns(pl.lit(None, dtype=pl.Int64).alias("quoted_post_id"))

    t = t.with_columns(
        pl.when(pl.col("text").fill_null("").str.starts_with("RT @")).then(pl.lit("repost"))
        .when(pl.col("in_reply_to_post_id").is_not_null()).then(pl.lit("reply"))
        .when(pl.col("quoted_post_id").is_not_null()).then(pl.lit("quote"))
        .otherwise(pl.lit("original")).alias("kind"),
        pl.col("in_reply_to_post_id").is_not_null().alias("is_reply"),
        pl.col("text").fill_null("").str.count_matches(r"(?:^|[^\w])@\w{1,15}").cast(pl.Int32).alias("mentions_n"),
        pl.col("text").fill_null("").str.count_matches(r"(?:^|[^\w&])#\w+").cast(pl.Int32).alias("hashtags_n"),
    )

    if media is not None:
        m = media.select(
            _i64("tweet_id").alias("post_id"),
            pl.struct(
                pl.col("media_type").replace({"animated_gif": "gif"}).alias("type"),
                pl.col("media_url").alias("url"),
                pl.lit(None, dtype=pl.String).alias("video_url"),
                pl.col("width").cast(pl.Int32, strict=False).alias("width"),
                pl.col("height").cast(pl.Int32, strict=False).alias("height"),
                pl.lit(None, dtype=pl.Int64).alias("duration_ms"),
                pl.lit(None, dtype=pl.String).alias("alt"),
            ).alias("m"),
        ).group_by("post_id").agg(pl.col("m").alias("media"))
        t = t.join(m, on="post_id", how="left")
    if urls is not None:
        rows = urls.select(_i64("tweet_id").alias("post_id"), "url", "expanded_url").to_dicts()
        recs = [
            {
                "post_id": r["post_id"],
                "u": {"tco": r["url"], "expanded": r["expanded_url"], "final_domain": registrable_domain(r["expanded_url"]),
                      "is_media": is_media_link(r["expanded_url"])},
            }
            for r in rows
        ]
        u = pl.DataFrame(recs, schema={"post_id": pl.Int64, "u": URL_STRUCT}).group_by("post_id").agg(pl.col("u").alias("urls"))
        t = t.join(u, on="post_id", how="left")

    if acct is not None:
        a = acct.select(
            _i64("account_id").alias("author_id"),
            pl.col("username").alias("author_handle"),
            pl.col("account_display_name").alias("author_display_name"),
            _ts("created_at").alias("author_created_at_ms"),
            pl.col("num_followers").cast(pl.Int64, strict=False).alias("author_followers"),
            pl.col("num_following").cast(pl.Int64, strict=False).alias("author_following"),
            pl.col("num_tweets").cast(pl.Int64, strict=False).alias("author_posts_count"),
            _ts("updated_at").alias("snapshot_at_ms"),
        ).unique("author_id")
        if prof is not None:
            a = a.join(prof.select(_i64("account_id").alias("author_id"), pl.col("bio").alias("author_bio")).unique("author_id"), on="author_id", how="left")
        t = t.join(a, on="author_id", how="left")

    t = t.with_columns(
        pl.lit("file_date").alias("snapshot_kind"),
        pl.lit("absent").alias("views_state"),
        pl.lit("community_archive").alias("source"),
        pl.lit(str(ca_dir.name)).alias("source_file"),
        pl.lit(False).alias("synthetic_id"),
    )
    posts = conform(t)

    accounts = (
        posts.select(
            "author_id", "author_handle", "snapshot_at_ms",
            pl.col("author_followers").alias("followers"), pl.col("author_following").alias("following"),
            pl.col("author_posts_count").alias("posts_count"), pl.col("author_created_at_ms").alias("created_at_ms"),
            pl.lit(None, dtype=pl.Boolean).alias("verified"), pl.lit("community_archive").alias("source"),
        ).unique("author_id")
    )

    edges = []
    fol = _read(ca_dir, "followers")
    if fol is not None:
        edges.append(fol.select(_i64("follower_account_id").alias("src_author_id"), _i64("account_id").alias("dst_author_id")))
    fing = _read(ca_dir, "following")
    if fing is not None:
        edges.append(fing.select(_i64("account_id").alias("src_author_id"), _i64("following_account_id").alias("dst_author_id")))
    edge_df = (
        pl.concat(edges).unique().with_columns(pl.lit("community_archive").alias("source"))
        if edges else pl.DataFrame(schema={"src_author_id": pl.Int64, "dst_author_id": pl.Int64, "source": pl.String})
    )

    # Donors: accounts whose own follower/following lists are in the archive. Only for these is
    # the mutual-follow share exact.
    donor_ids = []
    if fol is not None:
        donor_ids.append(fol.select(_i64("account_id").alias("author_id")))
    if fing is not None:
        donor_ids.append(fing.select(_i64("account_id").alias("author_id")))
    donors = pl.concat(donor_ids).unique() if donor_ids else pl.DataFrame(schema={"author_id": pl.Int64})

    out = {"posts": posts, "accounts": accounts, "edges": edge_df, "donors": donors}
    likes = _read(ca_dir, "likes")
    if likes is not None:
        out["viewer_likes"] = likes.select(_i64("account_id").alias("viewer_id"), _i64("liked_tweet_id").alias("post_id")).unique()
    return out
