"""Constants ported from the X For You repository.

These are not tunable: they describe X's system. Each one cites the file it
comes from, relative to the repository root. If X changes one upstream, change
it here and in nothing else.
"""

from __future__ import annotations

import math

# Snowflake epoch used to decode post creation time from the post ID.
# phoenix/xrex/data/recsys/recsys_batch.py:996-997 (TWITTER_EPOCH_MS)
TWITTER_EPOCH_MS = 1288834974657

# --- Phoenix candidate feature prep -----------------------------------------
# phoenix/xrex/models/recsys_feature_prep.py:38
ENGAGEMENT_COUNT_LOG2_SCALE = 32.0
# phoenix/xrex/models/recsys_feature_prep.py:51, 55-72 (hourly buckets, 0 = unknown, 81 = overflow)
POST_AGE_MAX_MINUTES = 4800
POST_AGE_GRANULARITY_MINUTES = 60
POST_AGE_NUM_BUCKETS = POST_AGE_MAX_MINUTES // POST_AGE_GRANULARITY_MINUTES + 2
# phoenix/xrex/models/recsys_feature_prep.py:227-228 (24 hours + unknown)
HOUR_OF_DAY_CARDINALITY = 25
# phoenix/xrex/data/recsys/feature_config.py:87 (counts zeroed and flag set for posts older than this)
STALE_POST_14D_TTL_SEC = 1_213_200

# --- Exploration ("post unexplored") label -----------------------------------
# phoenix/xrex/data/recsys/recsys_batch.py:105-135
EXPLORATION_REACH_TARGET_FRACTION = 0.03
EXPLORATION_WINDOW_HOURS = 24.0
EXPLORATION_HALF_LIFE_HOURS = 8.0

# --- Popular posts projection ------------------------------------------------
# home-mixer/util/popular_posts.rs:36-53
POPULAR_HALF_LIFE_HOURS = 8.0
POPULAR_FLOOR_HOURS = 0.5
POPULAR_WINDOW_HOURS = 24.0
# home-mixer/util/popular_authors.rs:11
POPULAR_AUTHORS_FRACTION = 0.00005

# --- Cold start --------------------------------------------------------------
# home-mixer/params/param.rs:608-669 and scorers/author_cold_start.rs
COLD_START_FOLLOWER_CAP = 50_000
COLD_START_MAX_AGE_SEC = 7200
COLD_START_IMPRESSION_THRESHOLD = 200
COLD_START_PRIOR_ALPHA = 0.75
COLD_START_PRIOR_BETA = 49.25
# phoenix/xrex/data/cold_pool_filter.py:16, 26-34 (retrieval cold pool)
COLD_POOL_MAX_FAVS = 8
COLD_POOL_MAX_VIEWS = 500

# --- Video gates -------------------------------------------------------------
# home-mixer/params/param.rs:707-712 (MinVideoDurationMs) and
# phoenix-rankall-strato/lib/eventProcessing.strato:403-418 (video index requires all media > 10 s)
MIN_VIDEO_DURATION_MS = 10_000
# home-mixer/candidate_hydrators/tweet_type_metrics_hydrator.rs:80-84
VIDEO_SHORT_MS = 10_000
VIDEO_LONG_MS = 60_000
# thunder/kafka/tweet_events_listener.rs:55-74 (Thunder's has_video requires >= 5 s)
THUNDER_VIDEO_MIN_MS = 5_000

# --- Serving window ----------------------------------------------------------
# home-mixer/params/config.rs:39 (MAX_POST_AGE)
MAX_POST_AGE_HOURS = 48

# --- Follower tiers ----------------------------------------------------------
# home-mixer/candidate_hydrators/tweet_type_metrics_hydrator.rs (AUTHOR_FOLLOWERS_* buckets)
FOLLOWER_TIER_EDGES = (100, 1_000, 10_000, 100_000, 1_000_000)
FOLLOWER_TIER_NAMES = ("0_100", "100_1k", "1k_10k", "10k_100k", "100k_1m", "1m_plus")

# --- Text weighting ----------------------------------------------------------
# home-mixer/models/content_features.rs:4-6
URL_WEIGHTED_LEN = 23
SINGLE_WEIGHT_CODEPOINT_RANGES = ((0, 4351), (8192, 8205), (8208, 8223), (8242, 8247))

# --- Credibility -------------------------------------------------------------
# botmaker-rules/scarecrow/derived-feature/IsHighPageRankUser.df:3-18 (user-cred score >= 54, fallback followers > 25k)
HIGH_PAGERANK_FOLLOWER_FALLBACK = 25_000


def exploration_view_target(followers: float, age_hours: float) -> float:
    """Views below which an original post counts as underexposed at `age_hours`.

    Exact port of compute_post_unexplored_labels (recsys_batch.py:105-135).
    """
    decay = math.log(2.0) / EXPLORATION_HALF_LIFE_HOURS
    saturation = 1.0 - math.exp(-decay * EXPLORATION_WINDOW_HOURS)
    clamped = min(max(age_hours, 0.0), EXPLORATION_WINDOW_HOURS)
    return EXPLORATION_REACH_TARGET_FRACTION * followers * (1.0 - math.exp(-decay * clamped)) / saturation


def popular_quality(views: float, age_hours: float) -> float:
    """Projected 24 h views used to rank popular posts. Port of popular_posts.rs::quality."""
    lam = math.log(2.0) / POPULAR_HALF_LIFE_HOURS
    saturation = 1.0 - math.exp(-lam * POPULAR_WINDOW_HOURS)
    elapsed = 1.0 - math.exp(-lam * max(age_hours, POPULAR_FLOOR_HOURS))
    return views * saturation / elapsed
