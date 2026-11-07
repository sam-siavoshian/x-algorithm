"""Phoenix candidate-feature transforms, reproduced exactly (DATA_PIPELINE.md 7.1).

These are evaluated at model time on an age grid (a draft has age 0 and zero
counts; the momentum loop re-evaluates at later ages with simulated counts),
so they are functions, not stored columns. NumPy versions are used by the
data layer and tests; the model layer reuses them in its own framework.
"""

from __future__ import annotations

import numpy as np

from xviral.constants import (
    ENGAGEMENT_COUNT_LOG2_SCALE,
    HOUR_OF_DAY_CARDINALITY,
    POST_AGE_GRANULARITY_MINUTES,
    POST_AGE_MAX_MINUTES,
    STALE_POST_14D_TTL_SEC,
)

AGE_GRID_HOURS = (0.25, 1.0, 2.0, 6.0, 24.0)
COUNT_ORDER = ("likes", "replies", "reposts", "quotes", "views")  # recsys_feature_prep.py:32-37


def post_age_bucket(impr_ts_sec: np.ndarray, creation_ts_sec: np.ndarray, granularity_mins: int = POST_AGE_GRANULARITY_MINUTES) -> np.ndarray:
    """Port of _compute_post_age_bucket_linear (recsys_feature_prep.py:55-72)."""
    impr = np.asarray(impr_ts_sec, dtype=np.int64)
    cre = np.asarray(creation_ts_sec, dtype=np.int64)
    num_normal = POST_AGE_MAX_MINUTES // granularity_mins
    overflow = num_normal + 1
    age_min = (impr - cre) // 60
    bucket = np.clip(age_min // granularity_mins + 1, 0, overflow)
    bucket = np.where((age_min < 0) | (impr == 0) | (cre == 0), 0, bucket)
    return bucket.astype(np.int32)


def post_age_bucket_from_hours(age_hours: float | np.ndarray) -> np.ndarray:
    a = np.asarray(age_hours, dtype=np.float64)
    base = np.full(a.shape, 1_700_000_000, dtype=np.int64)
    return post_age_bucket(base + (a * 3600).astype(np.int64), base)


def count_log_features(counts: np.ndarray) -> np.ndarray:
    """log2(max(c, 0) + 1) / 32 per count in COUNT_ORDER, plus the constant 1 Phoenix appends.

    recsys_feature_prep.py:730-754 feeds [fav, reply, repost, quote, view, 1] into a 6->64->D MLP.
    """
    c = np.maximum(np.asarray(counts, dtype=np.float32), 0.0)
    v = np.log2(c + 1.0) / ENGAGEMENT_COUNT_LOG2_SCALE
    ones = np.ones(v.shape[:-1] + (1,), dtype=np.float32)
    return np.concatenate([v, ones], axis=-1)


def zero_stale_counts(counts: np.ndarray, age_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Counts are zeroed and a stale flag set for posts older than 14 days (feature_config.py:87)."""
    stale = np.asarray(age_sec) > STALE_POST_14D_TTL_SEC
    out = np.where(stale[..., None], 0, counts)
    return out, stale


def local_hour_index(utc_ts_sec: np.ndarray, utc_offset_hours: np.ndarray) -> np.ndarray:
    """Hour in the viewer's (here: audience's) time zone, 1..24; 0 = unknown (cardinality 25)."""
    ts = np.asarray(utc_ts_sec, dtype=np.int64)
    off = np.asarray(utc_offset_hours, dtype=np.float64)
    known = (ts > 0) & ~np.isnan(off)
    hour = ((ts + np.nan_to_num(off) * 3600).astype(np.int64) // 3600) % 24
    idx = np.where(known, hour + 1, 0)
    assert idx.max(initial=0) < HOUR_OF_DAY_CARDINALITY
    return idx.astype(np.int32)
