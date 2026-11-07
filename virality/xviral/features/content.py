"""Content scalars that home-mixer computes but the ranker ignores (DATA_PIPELINE.md 7.2).

Phoenix can ignore them because its post-ID table memorises each post's
outcome. A surrogate scoring unseen posts cannot, so they are inputs. The set
is exactly what home-mixer hydrates (models/content_features.rs,
candidate_hydrators/media_info_hydrator.rs, quote_hydrator.rs,
tweet_type_metrics_hydrator.rs), plus the final-domain class needed by the
URL-verdict rules.
"""

from __future__ import annotations

import polars as pl

from xviral.constants import MIN_VIDEO_DURATION_MS, THUNDER_VIDEO_MIN_MS, VIDEO_LONG_MS, VIDEO_SHORT_MS
from xviral.text.urls import DOMAIN_CLASS_NAMES, domain_class
from xviral.text.weighted_length import text_metrics

ASPECT_CLASSES = ("none", "vertical", "square", "horizontal")


def _metrics(texts: list[str | None], attach: list[int]) -> tuple[list[int], list[int]]:
    lens, ext = [], []
    for t, a in zip(texts, attach):
        m = text_metrics(t, a)
        lens.append(m.weighted_len)
        ext.append(m.external_urls)
    return lens, ext


def content_features(df: pl.DataFrame) -> pl.DataFrame:
    """Returns post_id plus content scalars. Input: unified posts."""
    mtype = pl.element().struct.field("type")
    has_media = pl.col("media").list.len() > 0
    # The t.co link X appends for attached media is not an external URL (content_features.rs).
    attach = df.select(has_media.cast(pl.Int32).alias("a"))["a"].to_list()
    lens, ext = _metrics(df["text"].to_list(), attach)

    # External URLs as the source reports them (more reliable than the text scan for expanded links).
    ext_urls = pl.col("urls").list.eval(
        pl.element().filter(~pl.element().struct.field("is_media").fill_null(False)
                            & (pl.element().struct.field("final_domain").fill_null("") != "x.com"))
        .struct.field("final_domain")
    )
    out = df.select(
        "post_id",
        pl.Series("weighted_text_len", lens, dtype=pl.Int32),
        pl.Series("external_urls_text", ext, dtype=pl.Int32),
        pl.col("text").fill_null("").str.count_matches("\n").cast(pl.Int32).alias("newline_count"),
        ext_urls.alias("_domains"),
        has_media.alias("has_media"),
        pl.col("media").list.eval(mtype == "photo").list.any().fill_null(False).alias("has_photo"),
        pl.col("media").list.eval(mtype == "video").list.any().fill_null(False).alias("has_video"),
        pl.col("media").list.eval(mtype == "gif").list.any().fill_null(False).alias("has_gif"),
        pl.col("media").list.len().cast(pl.Int32).alias("media_count"),
        pl.col("media").list.eval(pl.element().filter(mtype == "video").struct.field("duration_ms")).alias("_durs"),
        pl.col("media").list.first().struct.field("width").alias("_w"),
        pl.col("media").list.first().struct.field("height").alias("_h"),
        (pl.col("kind") == "quote").alias("is_quote"),
        pl.col("mentions_n").fill_null(0),
        pl.col("hashtags_n").fill_null(0),
        pl.col("text_note").is_not_null().alias("_has_note"),
    ).with_columns(
        pl.col("_durs").list.max().alias("video_duration_ms"),
        pl.col("_durs").list.min().alias("min_video_duration_ms"),
        pl.col("_domains").list.len().cast(pl.Int32).alias("external_url_count"),
        pl.col("_domains").list.first().alias("first_external_domain"),
    ).with_columns(
        (pl.col("external_url_count") > 0).alias("has_url"),
        # VQV eligibility and the retrieval video index both require > 10 s
        # (param.rs:707-712; eventProcessing.strato:403-418: every media item must be a video > 10 s).
        (pl.col("min_video_duration_ms") > MIN_VIDEO_DURATION_MS).fill_null(False).alias("video_gt_10s"),
        pl.when(pl.col("has_video") & (pl.col("media_count") == pl.col("_durs").list.len()) & (pl.col("min_video_duration_ms") > MIN_VIDEO_DURATION_MS))
        .then(True).otherwise(False).alias("video_index_eligible"),
        (pl.col("has_video") & (pl.col("media_count") == 1) & (pl.col("video_duration_ms") >= THUNDER_VIDEO_MIN_MS)).fill_null(False).alias("thunder_has_video"),
        pl.when(~pl.col("has_video")).then(pl.lit("none"))
        .when(pl.col("min_video_duration_ms") <= VIDEO_SHORT_MS).then(pl.lit("le_10s"))
        .when(pl.col("min_video_duration_ms") <= VIDEO_LONG_MS).then(pl.lit("10_60s"))
        .otherwise(pl.lit("gt_60s")).alias("video_duration_bucket"),
        pl.when(pl.col("_w").is_null() | pl.col("_h").is_null()).then(pl.lit("none"))
        .when(pl.col("_w") * 10 < pl.col("_h") * 9).then(pl.lit("vertical"))
        .when(pl.col("_w") * 10 > pl.col("_h") * 11).then(pl.lit("horizontal"))
        .otherwise(pl.lit("square")).alias("media_aspect"),
        # "Show more" (head 45) applies to posts beyond 280 weighted characters.
        ((pl.col("weighted_text_len") > 280) | pl.col("_has_note")).alias("is_long_post"),
    )
    classes = [domain_class(d) for d in out["first_external_domain"].to_list()]
    out = out.with_columns(pl.Series("domain_class", classes, dtype=pl.Enum(list(DOMAIN_CLASS_NAMES))))
    return out.drop("_domains", "_durs", "_w", "_h", "_has_note")
