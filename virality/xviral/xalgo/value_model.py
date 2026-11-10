"""Port of X's value model: xai-value-model/{weights,scoring}.rs.

Final score for one viewer and one post:
    weighted = sum_h weight_h * P(action_h)          (weighted_terms)
    score    = offset(weighted)                      (offset_score)
    score   *= author_diversity * out_of_network     (post_fusion_multipliers)
The weights multiply predicted probabilities, not counts (README, "How weights
work"). Weights come from home-mixer/params/param.rs through the mapping in
home-mixer/scorers/value_model.rs::weights_for; diversity and the
out-of-network factor run only in vm-ranker and come from vm-ranker/params.rs.

Heads are keyed by the names scoring.rs uses. `PHOENIX_INDEX` maps the binary
ones to Phoenix ActionName indices (recsys.proto), which is how the surrogate's
64-way output lines up with this code.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from xviral.xalgo import params

NEGATIVE_SCORES_OFFSET = 0.001  # weights.rs:4

# term name -> home-mixer param name (home-mixer/scorers/value_model.rs:8-52)
_PARAM_OF = {
    "favorite": "FavoriteWeight", "reply": "ReplyWeight", "retweet": "RetweetWeight",
    "photo_expand": "PhotoExpandWeight", "video_open": "VideoOpenWeight", "click": "ClickWeight",
    "open_link": "OpenLinkWeight", "profile_click": "ProfileClickWeight", "vqv": "VqvWeight", "share": "ShareWeight",
    "share_via_dm": "ShareViaDmWeight", "share_via_copy_link": "ShareViaCopyLinkWeight", "dwell": "DwellWeight",
    "quote": "QuoteWeight", "quoted_click": "QuotedClickWeight", "quoted_vqv": "QuotedVqvWeight",
    "follow_author": "FollowAuthorWeight", "post_unexplored": "PostUnexploredWeight", "post_video": "PostVideoWeight",
    "grok_analyze": "GrokAnalyzeWeight", "not_interested": "NotInterestedWeight", "block_author": "BlockAuthorWeight",
    "mute_author": "MuteAuthorWeight", "report": "ReportWeight", "not_dwelled": "NotDwelledWeight",
    "cont_dwell_time": "ContDwellTimeWeight", "cont_click_dwell_time": "ContClickDwellTimeWeight",
}
POSITIVE = ("favorite", "reply", "retweet", "photo_expand", "video_open", "click", "open_link", "profile_click", "vqv",
            "share", "share_via_dm", "share_via_copy_link", "dwell", "quote", "quoted_click", "quoted_vqv",
            "follow_author", "post_unexplored", "post_video", "grok_analyze")          # weights.rs positive_sum
NEGATIVE = ("not_interested", "block_author", "mute_author", "report", "not_dwelled")  # weights.rs negative_sum
CONTINUOUS = ("cont_dwell_time", "cont_click_dwell_time")  # seconds, linear, uncapped (scoring.rs:152-159)

# Binary heads -> Phoenix ActionName index (phoenix/crates/serving/xai-recsys-proto/proto/recsys.proto:456-697).
PHOENIX_INDEX = {
    "favorite": 1, "reply": 4, "quote": 5, "retweet": 6, "report": 10, "dwell": 11, "not_dwelled": 12, "vqv": 13,
    "photo_expand": 14, "not_interested": 17, "follow_author": 21, "block_author": 23, "mute_author": 25, "click": 29,
    "profile_click": 30, "share_via_copy_link": 32, "share_via_dm": 33, "share": 37, "open_link": 39,
    "post_video": 44, "grok_analyze": 46, "video_open": 47, "post_unexplored": 59,
}


@dataclass(frozen=True)
class Weights:
    favorite: float = 0.0
    reply: float = 0.0
    retweet: float = 0.0
    photo_expand: float = 0.0
    video_open: float = 0.0
    click: float = 0.0
    open_link: float = 0.0
    profile_click: float = 0.0
    vqv: float = 0.0
    share: float = 0.0
    share_via_dm: float = 0.0
    share_via_copy_link: float = 0.0
    dwell: float = 0.0
    quote: float = 0.0
    quoted_click: float = 0.0
    quoted_vqv: float = 0.0
    follow_author: float = 0.0
    post_unexplored: float = 0.0
    post_video: float = 0.0
    grok_analyze: float = 0.0
    not_interested: float = 0.0
    block_author: float = 0.0
    mute_author: float = 0.0
    report: float = 0.0
    not_dwelled: float = 0.0
    cont_dwell_time: float = 0.0
    cont_click_dwell_time: float = 0.0
    bidirectional_follow_reply_weight_boost: float = 0.0
    bidirectional_follow_dwell_weight_boost: float = 0.0
    post_unexplored_include_out_of_network: bool = False  # hard-coded false in home-mixer value_model.rs:29
    author_diversity_decay: float = 0.5
    author_diversity_floor: float = 0.25
    oon_weight_factor: float = 0.75

    @classmethod
    def from_repo(cls) -> "Weights":
        hm, vm = params.home_mixer(), params.vm_ranker()
        kw = {term: float(hm[p]) for term, p in _PARAM_OF.items()}
        kw["bidirectional_follow_reply_weight_boost"] = float(hm["BidirectionalFollowReplyWeightBoost"])
        kw["bidirectional_follow_dwell_weight_boost"] = float(hm["BidirectionalFollowDwellWeightBoost"])
        kw["author_diversity_decay"] = float(vm["AuthorDiversityDecay"])
        kw["author_diversity_floor"] = float(vm["AuthorDiversityFloor"])
        kw["oon_weight_factor"] = float(vm["OonWeightFactor"])
        return cls(**kw)

    def positive_sum(self) -> float:
        return sum(getattr(self, t) for t in POSITIVE)

    def negative_sum(self) -> float:
        return -sum(getattr(self, t) for t in NEGATIVE)

    def total_sum(self) -> float:
        return self.positive_sum() + self.negative_sum()


def weighted_terms(scores: dict[str, float | None], w: Weights, *, mutual_follow: bool = False, is_reply: bool = False,
                   is_retweet: bool = False, in_network: bool | None = None, vqv_eligible: bool = False) -> dict[str, float]:
    """weighted_terms (scoring.rs:79-218) for the heads this package can supply. A missing head contributes 0."""
    boost_ok = mutual_follow and not is_reply and not is_retweet  # inputs.rs:20-22
    out: dict[str, float] = {}
    for term in (*POSITIVE, *NEGATIVE, *CONTINUOUS):
        wt = getattr(w, term)
        if term == "reply" and boost_ok and w.bidirectional_follow_reply_weight_boost != 0.0:
            wt += w.bidirectional_follow_reply_weight_boost
        if term == "dwell" and boost_ok and w.bidirectional_follow_dwell_weight_boost != 0.0:
            wt += w.bidirectional_follow_dwell_weight_boost
        if term == "vqv" and not vqv_eligible:
            wt = 0.0
        if term == "post_unexplored" and not (w.post_unexplored_include_out_of_network or in_network is True):
            wt = 0.0
        v = scores.get(term)
        out[term] = (0.0 if v is None else float(v)) * wt
    return out


def offset_score(combined: float, w: Weights) -> float:
    """offset_score (scoring.rs:239-248): net-negative scores are compressed into [0, 0.001)."""
    total = w.total_sum()
    if total == 0.0:
        return max(combined, 0.0)
    if combined < 0.0:
        return (combined + w.negative_sum()) / total * NEGATIVE_SCORES_OFFSET
    return combined + NEGATIVE_SCORES_OFFSET


def diversity_multiplier(w: Weights, k: int) -> float:
    """(1 - floor) * decay^k + floor, k = higher-scored posts by the same author in the feed (scoring.rs:261-263)."""
    return (1.0 - w.author_diversity_floor) * w.author_diversity_decay ** k + w.author_diversity_floor


def oon_multiplier(w: Weights, *, in_network: bool | None, is_reply: bool = False, is_retweet: bool = False,
                   rescore_in_network_replies: bool = True) -> float:
    """oon_applies (scoring.rs:284-293): out-of-network posts, and in-network replies/reposts, are discounted."""
    if in_network is False or (in_network is True and rescore_in_network_replies and (is_reply or is_retweet)):
        return w.oon_weight_factor
    return 1.0


def fuse(scores: dict[str, float | None], w: Weights, *, in_network: bool | None = None, same_author_rank: int = 0, **kw) -> dict:
    """Score for one (viewer, post): terms, weighted sum, offset, then multipliers applied post-offset."""
    terms = weighted_terms(scores, w, in_network=in_network, **kw)
    weighted = sum(terms.values())
    base = offset_score(weighted, w)
    mult = diversity_multiplier(w, same_author_rank) * oon_multiplier(
        w, in_network=in_network, is_reply=kw.get("is_reply", False), is_retweet=kw.get("is_retweet", False))
    return {"terms": terms, "weighted": weighted, "offset": base, "multiplier": mult, "score": base * mult}


def head_weight_vector(w: Weights, n_heads: int = 64, *, mutual_share: float = 0.0, in_network: bool = True,
                       vqv_eligible: bool = False) -> np.ndarray:
    """Weights laid out on Phoenix head indices, so `probs[..., 64] @ vector` is the weighted sum.

    `mutual_share` is the fraction of in-network viewers who are mutual follows; the reply weight is the
    audience average of 5 and 5 + 15 (BidirectionalFollowReplyWeightBoost applies per viewer).
    """
    v = np.zeros(n_heads, dtype=np.float64)
    for term, idx in PHOENIX_INDEX.items():
        wt = getattr(w, term)
        if term == "reply" and in_network:
            wt += mutual_share * w.bidirectional_follow_reply_weight_boost
        if term == "dwell" and in_network:
            wt += mutual_share * w.bidirectional_follow_dwell_weight_boost
        if term == "vqv" and not vqv_eligible:
            wt = 0.0
        if term == "post_unexplored" and not (w.post_unexplored_include_out_of_network or in_network):
            wt = 0.0
        v[idx] = wt
    return v


def weights_table(w: Weights) -> list[tuple[str, float]]:
    return [(f.name, getattr(w, f.name)) for f in fields(w)]
