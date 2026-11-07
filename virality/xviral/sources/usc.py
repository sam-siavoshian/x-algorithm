"""USC X 2024 election corpus (github.com/sinking8/usc-x-24-us-election).

Layout: part_<n>/<name>_chunk_<k>.csv.gz, 50,000 rows per chunk, one row per post.
License: CC BY-NC-SA 4.0, research use only. Rows from this source are tagged
source='usc' so the commercial model can exclude them.

Format facts established on real chunks (see DATA_PIPELINE.md section 1):
- Nested fields (`user`, `media`, `links`, `viewCount`, `mentionedUsers`) are
  Python reprs, and `user` contains datetime.datetime(...) calls.
- `user.username` holds the display name; the screen name is only in `url`.
- `in_reply_to_status_id_str`, `conversationId` and `user_id` were stored as
  floats and lost precision. Reply status is taken from field presence only.
- `quotedTweet` is a bare 'True'/'False'; quoted text is not present.
- Counts are mature: within a 2.7 h chunk, views/followers show no age trend.
  `maturity_audit` re-checks that on every chunk.
- The corpus mixes at least five header layouts (checked across parts 1-46):
  May-July (32 cols), August (33, `id` and `id_str` both float, i.e. destroyed),
  September and November (`username` added; November `id` float), and
  "octobergap" (35 cols: `inReplyToTweetId`, `card`, no retweet IDs, and the
  placeholder 'PW' in views, reply and card fields). So: the post ID is always
  taken from the status URL, which is exact in every layout; 'PW' is null; and
  when reply fields are absent or 'PW', a leading @mention marks a reply, which
  is how X renders replies in text (99% agreement with the reply field in May-July).
"""

from __future__ import annotations

import gzip
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

from xviral.schema import EDGES_SCHEMA, MEDIA_STRUCT, URL_STRUCT, conform  # noqa: F401
from xviral.text.urls import domain_class, is_media_link, registrable_domain  # noqa: F401
from xviral.util.core import parse_pyrepr, snowflake_ms_expr, write_parquet

log = logging.getLogger(__name__)

_USER_CHAIN = (
    r"'created': datetime\.datetime\(([^)]*)\), 'followersCount': (\d+), 'friendsCount': (\d+), "
    r"'statusesCount': (\d+), 'favouritesCount': (\d+), 'listedCount': (\d+), 'mediaCount': (\d+)"
)


_NULLISH = ["", "PW", "nan", "NaN", "None", "none", "null"]
_EXPECTED = ['id', 'text', 'url', 'epoch', 'media', 'retweetedTweet', 'retweetedTweetID', 'id_str', 'lang', 'rawContent',
             'replyCount', 'retweetCount', 'likeCount', 'quoteCount', 'hashtags', 'mentionedUsers', 'links', 'viewCount',
             'quotedTweet', 'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'inReplyToTweetId', 'user']


def _normalize_layout(raw: pl.DataFrame) -> tuple[pl.DataFrame, bool]:
    """Null out placeholders, add missing columns. Returns (frame, reply_fields_known)."""
    raw = raw.with_columns(pl.when(pl.col(c).str.strip_chars().is_in(_NULLISH)).then(None).otherwise(pl.col(c)).alias(c) for c in raw.columns)
    has_reply_cols = "in_reply_to_screen_name" in raw.columns or "inReplyToTweetId" in raw.columns
    raw = raw.with_columns(pl.lit(None, dtype=pl.String).alias(c) for c in _EXPECTED if c not in raw.columns)
    reply_known = has_reply_cols and (
        raw.select(pl.col("in_reply_to_screen_name").is_not_null().any() | pl.col("inReplyToTweetId").is_not_null().any()).item()
    )
    return raw, bool(reply_known)


def _false_or_empty(col: str) -> pl.Expr:
    c = pl.col(col).fill_null("").str.strip_chars()
    return c.is_in(["", "False", "nan", "NaN", "None"])


def _int(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)


def _media_list(raw: str | None) -> list[dict] | None:
    items = parse_pyrepr(raw)
    if not items or not isinstance(items, list):
        return []
    out = []
    for m in items:
        if not isinstance(m, dict):
            continue
        t = m.get("type")
        t = {"animated_gif": "gif"}.get(t, t)
        info = m.get("original_info") or {}
        vi = m.get("video_info") or {}
        mp4s = [v for v in vi.get("variants", []) if v.get("content_type") == "video/mp4"]
        # Pick the variant nearest 832 kbps: enough for 384-448 px frames, small to fetch.
        mp4s.sort(key=lambda v: abs((v.get("bitrate") or 0) - 832_000))
        out.append(
            {
                "type": t,
                "url": m.get("media_url_https"),
                "video_url": mp4s[0]["url"] if mp4s else None,
                "width": info.get("width"),
                "height": info.get("height"),
                "duration_ms": vi.get("duration_millis"),
                "alt": m.get("ext_alt_text"),
            }
        )
    return out


def _url_list(links_raw: str | None, media_raw: str | None) -> list[dict]:
    out = []
    for l in parse_pyrepr(links_raw) or []:
        if not isinstance(l, dict):
            continue
        exp = l.get("expanded_url")
        dom = registrable_domain(exp)
        out.append({"tco": l.get("url"), "expanded": exp, "final_domain": dom, "is_media": is_media_link(exp)})
    # The media t.co link is part of the text; record it so weighted length can subtract it.
    seen = set()
    for m in parse_pyrepr(media_raw) or []:
        if isinstance(m, dict) and m.get("url") and m["url"] not in seen:
            seen.add(m["url"])
            out.append({"tco": m["url"], "expanded": m.get("expanded_url"), "final_domain": "x.com", "is_media": True})
    return out


def _count_list(raw: str | None) -> int:
    v = parse_pyrepr(raw)
    return len(v) if isinstance(v, list) else 0


def read_chunk(path: str | Path, prefilter: bool = True, keep_langs: tuple[str, ...] = ("en",),
               undetermined: tuple[str, ...] = ("und", "zxx", "qme", "qht", "qam", "qct", "qst", "art")) -> dict[str, pl.DataFrame]:
    """Parse one chunk. Returns posts (unified schema), accounts, author activity and an audit row."""
    path = Path(path)
    with gzip.open(path, "rb") as f:
        raw = pl.read_csv(f.read(), infer_schema=False, quote_char='"')
    n_raw = raw.height
    raw, reply_known = _normalize_layout(raw)
    df = raw.filter(pl.col("url").is_not_null() | pl.col("id").is_not_null())  # ad rows carry only text
    n_ads = n_raw - df.height

    # Exact ID from the status URL; id_str only if it is a plain integer (some layouts store floats).
    url_id = pl.col("url").str.extract(r"/status/(\d+)", 1).cast(pl.Int64, strict=False)
    str_id = pl.when(pl.col("id_str").str.contains(r"^\d+$")).then(pl.col("id_str").cast(pl.Int64, strict=False))
    if reply_known:
        is_reply = ~_false_or_empty("in_reply_to_status_id_str") | ~_false_or_empty("in_reply_to_screen_name") | ~_false_or_empty("inReplyToTweetId")
    else:
        is_reply = pl.col("rawContent").fill_null(pl.col("text")).fill_null("").str.contains(r"^@\w")
    df = df.with_columns(
        pl.coalesce(url_id, str_id).alias("post_id"),
        pl.when(~_false_or_empty("retweetedTweet") | ~_false_or_empty("retweetedTweetID")).then(pl.lit("repost"))
        .when(is_reply).then(pl.lit("reply"))
        .when(~_false_or_empty("quotedTweet")).then(pl.lit("quote"))
        .otherwise(pl.lit("original")).alias("kind"),
        pl.col("url").str.extract(r"(?:twitter|x)\.com/([^/]+)/status/", 1).alias("author_handle"),
        pl.col("user").str.extract(r"^\{'id': (\d+)", 1).cast(pl.Int64, strict=False).alias("author_id"),
        pl.col("user").str.extract_groups(_USER_CHAIN).alias("_u"),
        pl.col("user").str.extract(r"'blue': (True|False)", 1).alias("_blue"),
        pl.col("user").str.extract(r"'verified': (True|False)", 1).alias("_verified"),
    ).filter(pl.col("post_id").is_not_null())

    # Author activity side table: every row, including replies, feeds author-level features.
    activity = (
        df.with_columns(snowflake_ms_expr("post_id").alias("created_at_ms"))
        .group_by("author_id")
        .agg(
            pl.len().alias("rows"),
            (pl.col("kind") == "reply").sum().alias("replies"),
            (pl.col("kind") == "repost").sum().alias("reposts"),
            pl.col("kind").is_in(["original", "quote"]).sum().alias("originals"),
            pl.col("created_at_ms").min().alias("first_ms"),
            pl.col("created_at_ms").max().alias("last_ms"),
        )
        .with_columns(pl.lit("usc").alias("source"))
    )

    if prefilter:
        lang = pl.col("lang").fill_null("und")
        df = df.filter(pl.col("kind").is_in(["original", "quote"]) & (lang.is_in(list(keep_langs)) | lang.is_in(list(undetermined))))

    u = pl.col("_u").struct
    df = df.with_columns(
        snowflake_ms_expr("post_id").alias("created_at_ms"),
        (pl.col("epoch").cast(pl.Float64, strict=False) * 1000).cast(pl.Int64).alias("created_at_src_ms"),
        pl.col("rawContent").fill_null(pl.col("text")).alias("text"),
        pl.col("lang").alias("lang_src"),
        (pl.col("kind") == "reply").alias("is_reply"),
        pl.col("retweetedTweetID").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False).alias("reposted_post_id"),
        pl.col("viewCount").str.extract(r"'count': '(\d+)'", 1).cast(pl.Int64, strict=False).alias("views"),
        pl.when(pl.col("viewCount").str.contains("'count':")).then(pl.lit("count"))
        .when(pl.col("viewCount").str.contains("'state':")).then(pl.lit("hidden"))
        .otherwise(pl.lit("absent")).alias("views_state"),
        _int("likeCount").alias("likes"),
        _int("replyCount").alias("replies"),
        _int("retweetCount").alias("reposts"),
        _int("quoteCount").alias("quotes"),
        u.field("2").cast(pl.Int64, strict=False).alias("author_followers"),
        u.field("3").cast(pl.Int64, strict=False).alias("author_following"),
        u.field("4").cast(pl.Int64, strict=False).alias("author_posts_count"),
        u.field("7").cast(pl.Int64, strict=False).alias("author_media_count"),
        u.field("1").alias("_created_parts"),
        ((pl.col("_blue") == "True") | (pl.col("_verified") == "True")).alias("author_verified"),
        pl.col("user").str.extract(r"'username': '((?:[^'\\]|\\.)*)'", 1).alias("author_display_name"),
        pl.coalesce(
            pl.col("user").str.extract(r"'rawDescription': '((?:[^'\\]|\\.)*)', 'created'", 1),
            pl.col("user").str.extract(r"'rawDescription': \"((?:[^\"\\]|\\.)*)\", 'created'", 1),
        ).alias("author_bio"),
        pl.col("mentionedUsers").fill_null("").str.count_matches("'screen_name':").cast(pl.Int32).alias("mentions_n"),
        pl.lit("assumed_mature").alias("snapshot_kind"),
        pl.lit("usc").alias("source"),
        pl.lit(path.name).alias("source_file"),
        pl.lit(False).alias("synthetic_id"),
    )
    # author_created_at from 'Y, M, D, h, m, s' parts.
    parts = pl.col("_created_parts").str.extract_all(r"\d+")
    df = df.with_columns(
        pl.datetime(
            parts.list.get(0, null_on_oob=True).cast(pl.Int32), parts.list.get(1, null_on_oob=True).cast(pl.Int32),
            parts.list.get(2, null_on_oob=True).cast(pl.Int32), parts.list.get(3, null_on_oob=True).cast(pl.Int32),
            parts.list.get(4, null_on_oob=True).cast(pl.Int32), parts.list.get(5, null_on_oob=True).cast(pl.Int32),
            time_zone="UTC",
        ).dt.epoch("ms").alias("author_created_at_ms")
    )

    # Nested lists: parsed in Python, only for surviving rows that have them.
    media_raw = df["media"].fill_null("[]").to_list()
    links_raw = df["links"].fill_null("[]").to_list()
    tags_raw = df["hashtags"].fill_null("[]").to_list()
    media = [_media_list(m) if m and m != "[]" else [] for m in media_raw]
    urls = [_url_list(l if l != "[]" else None, m if m != "[]" else None) for l, m in zip(links_raw, media_raw)]
    tags = [_count_list(t) if t and t != "[]" else 0 for t in tags_raw]
    df = df.with_columns(
        pl.Series("media", media, dtype=pl.List(MEDIA_STRUCT)),
        pl.Series("urls", urls, dtype=pl.List(URL_STRUCT)),
        pl.Series("hashtags_n", tags, dtype=pl.Int32),
    )

    posts = conform(df)
    accounts = (
        df.select(
            "author_id", "author_handle",
            pl.col("created_at_ms").alias("snapshot_at_ms"),
            pl.col("author_followers").alias("followers"),
            pl.col("author_following").alias("following"),
            pl.col("author_posts_count").alias("posts_count"),
            pl.col("author_created_at_ms").alias("created_at_ms"),
            pl.col("author_verified").alias("verified"),
            pl.lit("usc").alias("source"),
        )
        .filter(pl.col("author_id").is_not_null())
        .unique(subset=["author_id", "followers"], keep="first")
    )
    audit = maturity_audit(df).with_columns(
        pl.lit(path.name).alias("source_file"),
        pl.lit(reply_known).alias("reply_fields_known"),
        pl.lit(n_raw).alias("rows_raw"),
        pl.lit(n_ads).alias("rows_ads"),
        pl.lit(posts.height).alias("rows_kept"),
    )
    return {"posts": posts, "accounts": accounts, "activity": activity, "audit": audit}


def maturity_audit(df: pl.DataFrame) -> pl.DataFrame:
    """Rank correlation between posting time and log(views/followers) within a chunk.

    If counts had been read soon after posting, later posts in the chunk would have
    systematically fewer views per follower (strongly negative rho). Mature counts give rho near 0.
    """
    d = df.filter(pl.col("views").is_not_null() & pl.col("author_followers").is_not_null() & pl.col("kind").is_in(["original", "quote"]))
    n = d.height
    if n < 200:
        return pl.DataFrame({"audit_n": [n], "audit_rho": [None], "audit_span_h": [None]}, schema={"audit_n": pl.Int64, "audit_rho": pl.Float64, "audit_span_h": pl.Float64})
    t = d["created_at_src_ms"].to_numpy().astype(np.float64)
    r = np.log1p(d["views"].to_numpy()) - np.log1p(d["author_followers"].to_numpy())
    rt = t.argsort().argsort()
    rr = r.argsort().argsort()
    rho = float(np.corrcoef(rt, rr)[0, 1])
    span = float((t.max() - t.min()) / 3.6e6)
    return pl.DataFrame({"audit_n": [n], "audit_rho": [rho], "audit_span_h": [span]})


def _process(path: str, out: str, prefilter: bool, keep_langs: tuple[str, ...]) -> str:
    out_dir = Path(out)
    stem = Path(path).parent.name + "__" + Path(path).name.replace(".csv.gz", "")
    target = out_dir / "posts" / f"{stem}.parquet"
    if target.exists():
        return f"skip {stem}"
    res = read_chunk(path, prefilter=prefilter, keep_langs=keep_langs)
    for name, frame in res.items():
        write_parquet(frame, out_dir / name / f"{stem}.parquet")
    return f"ok {stem} kept={res['posts'].height}"


def ingest(usc_root: str | Path, out_dir: str | Path, workers: int = 4, prefilter: bool = True,
           keep_langs: tuple[str, ...] = ("en",), limit: int | None = None) -> None:
    files = sorted(Path(usc_root).glob("part_*/*.csv.gz"))
    if limit:
        files = files[:limit]
    log.info("USC: %d chunks", len(files))
    # 'spawn', not fork: forking after Polars has started its thread pool can deadlock the children.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_process, str(f), str(out_dir), prefilter, keep_langs) for f in files]
        for i, fut in enumerate(as_completed(futs), 1):
            log.info("USC %d/%d %s", i, len(files), fut.result())
