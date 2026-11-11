"""The value-model port against numbers stated in the repository and derived from its code."""

import pytest

from xviral.xalgo import params
from xviral.xalgo import value_model as vm


def test_params_parse_from_repo():
    hm = params.home_mixer()
    assert hm["FavoriteWeight"] == 0.708 and hm["ReplyWeight"] == 5.0 and hm["ReportWeight"] == -234.0
    assert hm["BidirectionalFollowReplyWeightBoost"] == 15.0 and hm["ShareViaCopyLinkWeight"] == 20.0
    assert hm["MinVideoDurationMs"] == 10_000 and hm["ColdStartFollowerCap"] == 50_000
    assert isinstance(hm["EnableViewerColdStart"], bool)
    v = params.vm_ranker()
    assert v["OonWeightFactor"] == 0.75 and v["AuthorDiversityDecay"] == 0.5 and v["AuthorDiversityFloor"] == 0.25


def test_weight_sums_match_code():
    w = vm.Weights.from_repo()
    # negative_sum = -(not_interested + block + mute + report + not_dwelled)   (weights.rs:74-80)
    assert w.negative_sum() == pytest.approx(53.07 + 31.2 + 58.8 + 234.0 + 0.02)
    # positive_sum uses the base reply weight, not the boosted one (weights.rs:51-72)
    assert w.positive_sum() == pytest.approx(0.708 + 5 + 1 + 0.05 + 0 + 0.274 + 0.2 + 0 + 0 + 2.118 + 5 + 20 + 0.05 + 5 + 0.05 + 0 + 4 + 0.015 + 0 + 0)


def test_worked_example_from_research_report():
    # research/02_scoring_weights.md Appendix B: non-mutual out-of-network original.
    w = vm.Weights.from_repo()
    scores = {"favorite": 0.10, "reply": 0.01, "retweet": 0.02, "share_via_copy_link": 0.002, "report": 0.0005, "cont_click_dwell_time": 1.0}
    r = vm.fuse(scores, w, in_network=False)
    assert r["weighted"] == pytest.approx(0.0708 + 0.05 + 0.02 + 0.04 + 0.385 - 0.117)
    assert r["offset"] == pytest.approx(r["weighted"] + 0.001)
    assert r["score"] == pytest.approx(r["offset"] * 0.75)
    second = vm.fuse(scores, w, in_network=False, same_author_rank=1)
    assert second["multiplier"] == pytest.approx(0.75 * 0.625)


def test_mutual_follow_reply_boost_and_gates():
    w = vm.Weights.from_repo()
    base = vm.weighted_terms({"reply": 0.1}, w)["reply"]
    assert base == pytest.approx(0.5)
    assert vm.weighted_terms({"reply": 0.1}, w, mutual_follow=True)["reply"] == pytest.approx(2.0)   # 5 + 15
    assert vm.weighted_terms({"reply": 0.1}, w, mutual_follow=True, is_reply=True)["reply"] == pytest.approx(0.5)
    assert vm.weighted_terms({"post_unexplored": 1.0}, w, in_network=False)["post_unexplored"] == 0.0
    assert vm.weighted_terms({"post_unexplored": 1.0}, w, in_network=True)["post_unexplored"] == pytest.approx(0.015)
    assert vm.weighted_terms({"vqv": 1.0}, w, vqv_eligible=True)["vqv"] == w.vqv


def test_negative_scores_compress_below_every_positive():
    w = vm.Weights.from_repo()
    neg = vm.offset_score(-10.0, w)
    assert 0.0 <= neg < 0.001 <= vm.offset_score(0.0, w)
    assert vm.offset_score(-w.negative_sum(), w) == pytest.approx(0.0)
    assert vm.offset_score(-1.0, w) > vm.offset_score(-2.0, w)  # order preserved among negatives


def test_diversity_sequence():
    w = vm.Weights.from_repo()
    assert [round(vm.diversity_multiplier(w, k), 4) for k in range(4)] == [1.0, 0.625, 0.4375, 0.3438]


def test_head_vector_reproduces_weighted_sum():
    import numpy as np

    w = vm.Weights.from_repo()
    p = np.zeros(64)
    p[1], p[4], p[6], p[10] = 0.1, 0.01, 0.02, 0.0005
    v = vm.head_weight_vector(w, mutual_share=0.0, in_network=False)
    assert p @ v == pytest.approx(0.0708 + 0.05 + 0.02 - 0.117)
    v_in = vm.head_weight_vector(w, mutual_share=0.4, in_network=True)
    assert v_in[4] == pytest.approx(5 + 0.4 * 15)


def test_gates_from_repo_constants():
    from xviral.xalgo import gates

    row = {"followers_at_post": 10_000, "kind": "original", "hours_since_prev_original": 2.0, "has_video": True,
           "video_index_eligible": False, "video_duration_ms": 8_000, "url_blocked": True, "is_self_repeat": True}
    g = {x["gate"]: x for x in gates.check(row, originals_last_48h=50)}
    assert g["in_network_store"]["status"] == "fail" and g["author_diversity"]["status"] == "warn"
    assert g["video_index"]["status"] == "fail" and g["link_safety"]["status"] == "fail" and g["self_repeat"]["status"] == "warn"
    assert g["cold_start_slot"]["status"] == "pass"
    # 3% of followers within 24 hours on an 8-hour half-life: half the saturating curve's mass is reached well before 8 h.
    need = g["exploration_target"]["views_needed"]
    assert need["24h"] == 300 and 150 < need["8h"] < 200 and need["1h"] < need["2h"] < need["8h"]
    assert all(x["source"] for x in g.values())
    quiet = {x["gate"]: x for x in gates.check({"followers_at_post": 0, "kind": "quote"}, 0)}
    assert "video_index" not in quiet and "exploration_target" not in quiet and quiet["cold_start_slot"]["status"] == "info"
