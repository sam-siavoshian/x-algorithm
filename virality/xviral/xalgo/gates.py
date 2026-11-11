"""Deterministic distribution gates for a draft (ARCHITECTURE.md S3, and the decidable part of S2).

Everything here is read off X's code with the constants ported in xviral.constants and the
weights parsed from the repository; nothing is learned. Each gate reports a status, the numbers
behind it, and the file it comes from. The classifier-based pre-flight checks of S2 (adult
content, violence, spam sub-types) are not here: they need the label models.

Input is one feature row for the draft (serve.DraftScorer.featurize) plus the count of the
author's original posts in the last 48 hours.
"""

from __future__ import annotations

from xviral import constants as C
from xviral.xalgo import value_model as vm

PASS, WARN, FAIL, INFO = "pass", "warn", "fail", "info"


def _gate(name: str, status: str, detail: str, source: str, **numbers) -> dict:
    return {"gate": name, "status": status, "detail": detail, "source": source, **numbers}


def check(row: dict, originals_last_48h: int, weights: vm.Weights | None = None) -> list[dict]:
    w = weights or vm.Weights.from_repo()
    out: list[dict] = []
    followers = float(row.get("followers_at_post") or 0)

    # Thunder serves at most 50 original posts per author from its 48-hour store, newest first.
    cap = 50
    if originals_last_48h >= cap:
        out.append(_gate("in_network_store", FAIL, f"{originals_last_48h} originals in the last 48 hours: only the newest {cap} per author are "
                         "served to followers, so each new post pushes an older one out", "thunder/config.rs MAX_ORIGINAL_POSTS_PER_AUTHOR",
                         originals_last_48h=originals_last_48h, cap=cap))
    else:
        out.append(_gate("in_network_store", PASS, f"{originals_last_48h} originals in the last 48 hours, under the cap of {cap} per author",
                         "thunder/config.rs MAX_ORIGINAL_POSTS_PER_AUTHOR", originals_last_48h=originals_last_48h, cap=cap))

    # Author diversity: within one feed response, the k-th post by the same author is multiplied down.
    gap = row.get("hours_since_prev_original")
    second = vm.diversity_multiplier(w, 1)
    if gap is not None and gap < C.MAX_POST_AGE_HOURS:
        out.append(_gate("author_diversity", WARN if gap < 6 else INFO,
                         f"previous post {gap:.1f} hours ago: both are candidates for the same feed for 48 hours, and the lower-scored "
                         f"of the two is multiplied by {second:.3f}", "home-mixer/params/param.rs AuthorDiversityDecay, AuthorDiversityFloor",
                         hours_since_previous=float(gap), second_post_multiplier=second))

    # Out-of-network candidates are scored at a discount.
    out.append(_gate("out_of_network_factor", INFO, f"shown to non-followers, the post's score is multiplied by {w.oon_weight_factor}",
                     "home-mixer/params/param.rs OonWeightFactor", factor=w.oon_weight_factor))

    # Cold-start slot.
    if row.get("kind", "original") == "original" and 0 < followers <= C.COLD_START_FOLLOWER_CAP:
        out.append(_gate("cold_start_slot", PASS, f"eligible for the cold-start slot for the first {C.COLD_START_MAX_AGE_SEC // 3600} hours, "
                         f"while home impressions are under {C.COLD_START_IMPRESSION_THRESHOLD}", "home-mixer/scorers/author_cold_start.rs",
                         follower_cap=C.COLD_START_FOLLOWER_CAP))
    else:
        why = "a quote post" if row.get("kind") == "quote" else f"more than {C.COLD_START_FOLLOWER_CAP:,} followers" if followers else "follower count unknown"
        out.append(_gate("cold_start_slot", INFO, f"not eligible for the cold-start slot ({why})", "home-mixer/scorers/author_cold_start.rs",
                         follower_cap=C.COLD_START_FOLLOWER_CAP))

    # Video index: every video must be longer than 10 seconds.
    if row.get("has_video"):
        ok = bool(row.get("video_index_eligible"))
        out.append(_gate("video_index", PASS if ok else FAIL,
                         "eligible for the out-of-network video index" if ok else
                         f"not eligible for the out-of-network video index: every video must be longer than {C.MIN_VIDEO_DURATION_MS // 1000} seconds",
                         "phoenix-rankall-strato phoenixRankAllCandidateProcessor.strato", min_ms=C.MIN_VIDEO_DURATION_MS,
                         duration_ms=row.get("video_duration_ms")))

    # Links: any blocklisted hop removes the post everywhere.
    if row.get("url_blocked"):
        out.append(_gate("link_safety", FAIL, "a link in the post resolves to a blocklisted domain: the post would be dropped on every surface",
                         "visibility-filtering/rules/tweet_rules.rs"))

    # Self-repeat: near-duplicate of the author's own recent post.
    if row.get("is_self_repeat"):
        out.append(_gate("self_repeat", WARN, "near-duplicate of one of the account's posts from the last 30 days: duplicate and copypasta labels "
                         "limit distribution outside followers", "visibility-filtering/rules/tweet_rules.rs"))

    # Exploration target: views needed by age t to stop counting as under-exposed.
    if followers > 0:
        target = {f"{h}h": round(C.exploration_view_target(followers, h)) for h in (1, 2, 8, 24)}
        out.append(_gate("exploration_target", INFO, "views by age below which X's training data labels the post under-exposed for its follower count: "
                         + ", ".join(f"{v:,} by {k}" for k, v in target.items()), "phoenix/xrex/data/recsys/recsys_batch.py", views_needed=target))

    out.append(_gate("expiry", INFO, f"candidates older than {C.MAX_POST_AGE_HOURS} hours are no longer served", "home-mixer/params/config.rs",
                     hours=C.MAX_POST_AGE_HOURS))
    return out
