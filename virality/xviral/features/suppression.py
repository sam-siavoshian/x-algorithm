"""Suppression-risk and negative-feedback proxies (DATA_PIPELINE.md 7.6).

The value model's largest weights are negative (report -234, mute -58.8,
not interested -53.07, block -31.2; home-mixer/params/param.rs:509-527) and
public counts cannot observe them. These features are the best proxies, and
they also drive the ported visibility rules in the pre-flight stage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from xviral.constants import HIGH_PAGERANK_FOLLOWER_FALLBACK

log = logging.getLogger(__name__)

# grox/flows/ptos/state.py SafetyPolicyType, Spam category. The first three are labelled
# SpamHighRecall regardless of author credibility; the rest only for non-high-PageRank authors
# (grox/flows/ptos/task_write_safety_post_annotations_result_sink.py:233-248).
SPAM_ALWAYS = ("SpamEngagementBaiting", "SpamEngagementFarming", "SpamCardManipulation")
SPAM_UNLESS_CREDIBLE = (
    "SpamEngagementTrading", "SpamFinancialScam", "SpamHashTagAbuse", "SpamScamLinks",
    "SpamBotProduced", "SpamPlatformManipulation", "SpamMentionAbuse", "SpamManipulatedMedia",
)
SPAM_TYPES = ("NoViolation", *SPAM_ALWAYS, *SPAM_UNLESS_CREDIBLE)

# Thresholds on media classifiers that X's rules use (report 05; botmaker-rules derived features).
NSFW_HIGH_PRECISION = 0.95
NSFW_NEAR_PERFECT = 0.999
GORE_IMAGE_PRECISION = 0.6
GORE_VIDEO_RAW = 0.95
# safety-label-user-agg postToUserLabelRules.strato:395-426: 3 of the last 5 posts NSFW -> account label.
AUTHOR_NSFW_WINDOW, AUTHOR_NSFW_HITS = 5, 3


def load_blocklists(directory: str | Path | None) -> set[str]:
    """Domain blocklists, one registrable domain per line; '#' comments allowed."""
    if not directory:
        return set()
    out: set[str] = set()
    for p in Path(directory).glob("*.txt"):
        for line in p.read_text().splitlines():
            line = line.split("#", 1)[0].strip().lower()
            if line:
                out.add(line.removeprefix("www."))
    log.info("blocklists: %d domains", len(out))
    return out


def url_features(df: pl.DataFrame, blocked: set[str]) -> pl.DataFrame:
    """Any hop in a post's links on a blocklist -> the post would be dropped everywhere (SPAM)."""
    doms = pl.col("urls").list.eval(pl.element().struct.field("final_domain"))
    return df.select(
        "post_id",
        doms.list.eval(pl.element().is_in(list(blocked))).list.any().fill_null(False).alias("url_blocked"),
    )


def high_pagerank_fallback(followers: pl.Expr) -> pl.Expr:
    """X's credibility fallback when no user-cred score exists: followers > 25k (IsHighPageRankUser.df)."""
    return followers > HIGH_PAGERANK_FOLLOWER_FALLBACK


def spam_label_would_apply(policy_type: pl.Expr, high_pagerank: pl.Expr) -> pl.Expr:
    return policy_type.is_in(SPAM_ALWAYS) | (policy_type.is_in(SPAM_UNLESS_CREDIBLE) & ~high_pagerank)


def author_nsfw_flag(post_scores: pl.DataFrame, score_col: str = "p_nsfw_max") -> pl.DataFrame:
    """Per post: would the author carry an account NSFW label at this time (3 of the last 5 posts)?

    Uses only strictly earlier posts. Input: post_id, author_id, created_at_ms, score_col.
    """
    d = post_scores.sort("author_id", "created_at_ms").with_columns(
        (pl.col(score_col) >= NSFW_HIGH_PRECISION).cast(pl.Int32).alias("_hit")
    )
    d = d.with_columns(
        pl.col("_hit").shift(1).rolling_sum(window_size=AUTHOR_NSFW_WINDOW, min_samples=1).over("author_id", order_by="created_at_ms").fill_null(0).alias("_prior_hits")
    )
    return d.select("post_id", (pl.col("_prior_hits") >= AUTHOR_NSFW_HITS).alias("author_nsfw_label"))


def pagerank(edges: pl.DataFrame, alpha: float = 0.2, tol: float = 1e-3, max_iter: int = 50) -> pl.DataFrame:
    """PageRank over the follow edges we have (src follows dst), as in user-cred-v2.

    user-cred-v2 (UserCredV2App.scala:320-390) iterates new = (1-alpha)*propagated + alpha*teleport,
    alpha = 0.2, until sum|delta| < 0.001 or 50 iterations, and maps mass to
    clamp(165.2 + 7.07 * ln(mass), 0, 100). Our graph is a partial subgraph (donors' lists),
    so absolute mass is not comparable to X's; the rank is a relative credibility feature only.
    Teleport is uniform because premium status is unknown.
    """
    ids = pl.concat([edges["src_author_id"], edges["dst_author_id"]]).unique().sort()
    idx = {v: i for i, v in enumerate(ids.to_list())}
    n = len(idx)
    if n == 0:
        return pl.DataFrame(schema={"author_id": pl.Int64, "graph_pagerank": pl.Float64})
    src = np.fromiter((idx[v] for v in edges["src_author_id"].to_list()), dtype=np.int64)
    dst = np.fromiter((idx[v] for v in edges["dst_author_id"].to_list()), dtype=np.int64)
    out_deg = np.bincount(src, minlength=n).astype(np.float64)
    p = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        share = np.where(out_deg[src] > 0, p[src] / out_deg[src], 0.0)
        prop = np.bincount(dst, weights=share, minlength=n)
        dangling = p[out_deg == 0].sum() / n
        new = (1 - alpha) * (prop + dangling) + alpha / n
        delta = np.abs(new - p).sum()
        p = new
        if delta < tol:
            break
    return pl.DataFrame({"author_id": ids, "graph_pagerank": p * n})
