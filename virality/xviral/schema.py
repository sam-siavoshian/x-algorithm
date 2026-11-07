"""The unified post schema every source is parsed into.

Fields a source does not have are null. Nothing is imputed at this layer:
imputation is a feature-engineering decision and lives in xviral.features.
"""

from __future__ import annotations

import polars as pl

MEDIA_STRUCT = pl.Struct(
    {
        "type": pl.String,          # photo | video | gif
        "url": pl.String,           # image URL, or the thumbnail for video and gif
        "video_url": pl.String,     # an mp4 variant suitable for frame extraction
        "width": pl.Int32,
        "height": pl.Int32,
        "duration_ms": pl.Int64,
        "alt": pl.String,
    }
)

URL_STRUCT = pl.Struct(
    {
        "tco": pl.String,
        "expanded": pl.String,
        "final_domain": pl.String,
        "is_media": pl.Boolean,     # t.co links that point at the post's own media
    }
)

KINDS = ("original", "quote", "reply", "repost")
SOURCES = ("user_scrape", "usc", "community_archive", "analytics")

POSTS_SCHEMA: dict[str, pl.DataType] = {
    "post_id": pl.Int64,
    "author_id": pl.Int64,
    "author_handle": pl.String,
    "created_at_ms": pl.Int64,
    "created_at_src_ms": pl.Int64,
    "snapshot_at_ms": pl.Int64,
    "snapshot_kind": pl.String,     # exact | file_date | assumed_mature
    "text": pl.String,
    "text_note": pl.String,
    "lang_src": pl.String,
    "kind": pl.String,
    "in_reply_to_post_id": pl.Int64,
    "is_reply": pl.Boolean,
    "quoted_post_id": pl.Int64,
    "quoted_text": pl.String,
    "quoted_author_id": pl.Int64,
    "reposted_post_id": pl.Int64,
    "conversation_id": pl.Int64,
    "media": pl.List(MEDIA_STRUCT),
    "urls": pl.List(URL_STRUCT),
    "mentions_n": pl.Int32,
    "hashtags_n": pl.Int32,
    "views": pl.Int64,
    "views_state": pl.String,       # count | hidden | absent
    "likes": pl.Int64,
    "replies": pl.Int64,
    "reposts": pl.Int64,
    "quotes": pl.Int64,
    "bookmarks": pl.Int64,
    # Analytics-only labels (null elsewhere).
    "detail_expands": pl.Int64,
    "profile_clicks": pl.Int64,
    "link_clicks": pl.Int64,
    "new_follows": pl.Int64,
    "shares": pl.Int64,
    "engagements": pl.Int64,
    # Author snapshot carried on the row.
    "author_followers": pl.Int64,
    "author_following": pl.Int64,
    "author_posts_count": pl.Int64,
    "author_media_count": pl.Int64,
    "author_created_at_ms": pl.Int64,
    "author_verified": pl.Boolean,
    "author_display_name": pl.String,
    "author_bio": pl.String,
    # Provenance.
    "source": pl.String,
    "source_file": pl.String,
    "synthetic_id": pl.Boolean,     # post_id was synthesized because the source had none
}


def empty_posts() -> pl.DataFrame:
    return pl.DataFrame(schema=POSTS_SCHEMA)


def conform(df: pl.DataFrame) -> pl.DataFrame:
    """Add missing columns as typed nulls, cast present ones, drop extras, fix order."""
    exprs = []
    for name, dtype in POSTS_SCHEMA.items():
        e = pl.col(name).cast(dtype, strict=False) if name in df.columns else pl.lit(None, dtype=dtype)
        if isinstance(dtype, pl.List):
            # List columns (media, urls) are never null: "no media" is an empty list.
            e = e.fill_null(pl.lit([], dtype=dtype))
        exprs.append(e.alias(name))
    return df.select(exprs)


ACCOUNTS_SCHEMA: dict[str, pl.DataType] = {
    "author_id": pl.Int64,
    "author_handle": pl.String,
    "snapshot_at_ms": pl.Int64,
    "followers": pl.Int64,
    "following": pl.Int64,
    "posts_count": pl.Int64,
    "created_at_ms": pl.Int64,
    "verified": pl.Boolean,
    "source": pl.String,
}

EDGES_SCHEMA: dict[str, pl.DataType] = {
    "src_author_id": pl.Int64,   # the follower
    "dst_author_id": pl.Int64,   # the followed account
    "source": pl.String,
}
