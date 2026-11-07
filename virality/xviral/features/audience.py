"""Audience time zone and audience-local time (DATA_PIPELINE.md 7.4).

Phoenix embeds the viewer's local hour. We score for an audience, so we need
the audience's time zone. Without follower location data the best available
estimate is the author's own posting-hour distribution: accounts post when
their audience is awake. The estimate picks, among a small set of zones that
cover the tech audience, the one that best aligns the author's posting hours
with a typical waking-hours profile.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from xviral.features.phoenix import local_hour_index

# UTC offsets in hours (standard time; DST shifts are within the model's hour dither tolerance).
ZONES = {"us_pacific": -8.0, "us_central": -6.0, "us_eastern": -5.0, "uk": 0.0, "central_europe": 1.0, "india": 5.5, "east_asia": 8.0}

# Relative posting intensity by local hour for active social accounts (low overnight, broad day peak).
_WAKE = np.array([0.25, 0.15, 0.1, 0.08, 0.08, 0.12, 0.3, 0.55, 0.8, 0.95, 1.0, 1.0,
                  1.0, 1.0, 0.98, 0.95, 0.92, 0.9, 0.9, 0.9, 0.85, 0.75, 0.6, 0.4])


def estimate_audience_zone(posts: pl.DataFrame, min_posts: int = 10) -> pl.DataFrame:
    """author_id -> audience_zone, audience_utc_offset. Authors with too few posts get null."""
    h = posts.select("author_id", ((pl.col("created_at_ms") // 3_600_000) % 24).cast(pl.Int32).alias("h"))
    hist = h.group_by("author_id", "h").len().pivot(on="h", index="author_id", values="len").fill_null(0)
    hours = [str(i) for i in range(24)]
    for c in hours:
        if c not in hist.columns:
            hist = hist.with_columns(pl.lit(0, dtype=pl.UInt32).alias(c))
    counts = hist.select(hours).to_numpy().astype(np.float64)
    n = counts.sum(axis=1)
    p = counts / np.maximum(n[:, None], 1)
    names = list(ZONES)
    scores = []
    for z in names:
        off = ZONES[z]
        # Local hour for each UTC hour bin; interpolate fractional offsets between neighbours.
        local = (np.arange(24) + off) % 24
        lo = np.floor(local).astype(int) % 24
        hi = (lo + 1) % 24
        frac = local - np.floor(local)
        profile = (1 - frac) * _WAKE[lo] + frac * _WAKE[hi]
        scores.append(p @ profile)
    best = np.array(scores).argmax(axis=0)
    zone = [names[i] if n_i >= min_posts else None for i, n_i in zip(best, n)]
    return pl.DataFrame({
        "author_id": hist["author_id"],
        "audience_zone": pl.Series(zone, dtype=pl.String),
        "audience_utc_offset": pl.Series([ZONES[z] if z else None for z in zone], dtype=pl.Float64),
    })


def audience_time_features(posts: pl.DataFrame, zones: pl.DataFrame, default_offset: float = -5.0) -> pl.DataFrame:
    """Post-level local hour (Phoenix's 25-class index) and weekday in the audience's zone.

    Unknown authors default to US Eastern, the median zone of the US tech audience.
    """
    d = posts.select("post_id", "author_id", "created_at_ms").join(zones, on="author_id", how="left")
    off = d["audience_utc_offset"].fill_null(default_offset).to_numpy()
    ts = (d["created_at_ms"].to_numpy() // 1000).astype(np.int64)
    idx = local_hour_index(ts, off)
    dow = (((ts + (off * 3600).astype(np.int64)) // 86400 + 3) % 7).astype(np.int32)  # 1970-01-01 was a Thursday
    return pl.DataFrame({
        "post_id": d["post_id"],
        "local_hour_idx": idx,
        "local_dow": dow,
        "audience_zone_known": d["audience_zone"].is_not_null(),
    })
