"""The speak / stay-silent rule (xviral/selective.py): features, model evaluation from JSON, and the advisor hook."""
import json
import math

import numpy as np
import polars as pl
import pytest

from tests.test_serve import embed, trained  # noqa: F401  (the trained-model fixture)
from xviral import selective as S

HOUR = 3_600_000


def _version(q50, spread=0.02, n=10, seed=0, **kw):
    rng = np.random.default_rng(seed)
    qs = q50 + spread * rng.standard_normal(n)
    hist = np.zeros(70)
    hist[int(q50 * 10)] = 1.0
    return {"members": [{"likes_q50": float(q), "likes_q10": float(q - 0.5), "likes_q90": float(q + 0.6), "rel_like_q50": float(q - 1.0), "p_own_top10": 0.1,
                         "likes_hist": hist} for q in qs],
            "features": {"has_media": kw.get("media", 0), "has_video": 0, "has_photo": kw.get("media", 0), "has_url": 1, "ts_url_first_line": 0, "is_quote": 0,
                         "ts_lines": 2, "hashtags_n": 0, "mentions_n": 0, "ts_emoji": 0, "ts_open_question": 0, "ts_open_number": 0, "ts_open_first_person": 0,
                         "ts_open_announce": 0, "ts_call_to_action": 0, "ts_bullet_lines": 0, "pt_launch": 0.0, "pt_any": 0.0, "ts_words": kw.get("words", 20),
                         "kind": "original", "ts_len_vs_own": 0.1, "ts_media_vs_own": 0.0, "ts_link_vs_own": 0.0},
            "history": {"h_n": 48, "h_like_med": 1.0, "h_like_sd": 0.4, "h_like_iqr": 0.5, "h_like_p90": 1.6, "h_like_max": 2.0, "h_span_days": 30.0,
                        "h_media_share": 0.3, "h_words_med": 22.0, "followers_at_post": 12_000, "follower_tier_idx": 3, "cadence_7d": 9, "trailing_media_share": 0.3,
                        "trailing_link_share": 0.5},
            "novelty": {"sim_mean": 0.7, "sim_top3": 0.9, "sim_centroid": kw.get("cen", 0.8), "cohesion": 0.75},
            "embedding": np.eye(4)[kw.get("e", 0)]}


def test_features_do_not_depend_on_the_order_of_the_versions():
    a, b = _version(1.30, media=1, words=40, cen=0.9, e=0), _version(1.10, seed=1, e=1)
    fab, fba = S.pair_features(a, b), S.pair_features(b, a)
    assert fab.keys() == fba.keys()
    for k in fab:
        assert (math.isnan(fab[k]) and math.isnan(fba[k])) or fab[k] == pytest.approx(fba[k]), k
    assert fab["gap"] == pytest.approx(abs(np.mean([m["likes_q50"] for m in a["members"]]) - np.mean([m["likes_q50"] for m in b["members"]])))
    assert fab["d_media"] == 1.0 and fab["d_logwords"] > 0 and fab["sim_centroid_diff"] == pytest.approx(0.1) and fab["cos_ab"] == 0.0
    assert fab["agree"] == 1.0 and fab["ovl_hist"] == 1.0 and fab["log_followers"] == pytest.approx(math.log10(12_001))
    assert fab["width_pick"] == pytest.approx(1.1) and 0 < fab["z_ovl"] < 1
    # a second ensemble's view of the same pair is oriented by the first one's call
    assert S.pair_features(a, b, other=(b, a))["o_gap"] == pytest.approx(-fab["gap"])
    assert S.hist_p_greater(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_models_evaluate_from_json_and_fall_back_when_inputs_are_missing():
    f = S.pair_features(_version(1.30), _version(1.10, seed=1))
    assert S.predict({"type": "logistic_gap", "scale": 2.0}, f) == pytest.approx(1 / (1 + math.exp(-2.0 * f["gap"])))
    lin = {"type": "linear", "features": ["gap", "h_like_sd"], "impute": [0.1, 0.5], "mean": [0.1, 0.5], "std": [0.1, 0.2], "coef": [0.5, -0.3], "intercept": 0.2,
           "calibration": {"a": 1.0, "b": 0.0}}
    z = (f["gap"] - 0.1) / 0.1 * 0.5 + (0.4 - 0.5) / 0.2 * -0.3 + 0.2
    assert S.predict(lin, f) == pytest.approx(1 / (1 + math.exp(-z)))
    tree = {"split_feature": 0, "threshold": 0.15, "default_left": True, "left_child": {"leaf_value": -0.2}, "right_child": {"leaf_value": 0.4}}
    gbm = {"type": "gbm", "features": ["gap"], "init": 0.0, "trees": [tree, tree], "calibration": {"a": 1.0, "b": 0.1}}
    assert S.predict(gbm, f) == pytest.approx(1 / (1 + math.exp(-0.9)))
    mlp = {"type": "mlp", "features": ["gap"], "impute": [0.1], "mean": [0.0], "std": [1.0], "nets": [{"w1": [[1.0]], "b1": [0.0], "w2": [2.0], "b2": 0.0}], "calibration": None}
    assert S.predict(mlp, f) == pytest.approx(1 / (1 + math.exp(-2.0 * math.tanh(f["gap"]))))
    # members' medians only: the rich model cannot run, the stored scale answers
    bare = lambda q: {"members": [{"likes_q50": q}, {"likes_q50": q + 0.01}]}      # noqa: E731
    rule = S.SelectiveRule({"speak": 0.6, "kinds": {"same_topic": {"model": lin, "fallback": {"type": "logistic_gap", "scale": 3.0}}}})
    out = rule.decide(bare(1.5), bare(1.0))
    assert out["method"] == "fallback:logistic_gap" and out["pick"] == "a" and out["p_right"] == pytest.approx(1 / (1 + math.exp(-1.5))) and out["speak"]
    back = rule.decide(bare(1.0), bare(1.5))
    assert back["pick"] == "b" and back["p_a_better"] == pytest.approx(1 - out["p_a_better"])
    assert not rule.decide(bare(1.02), bare(1.0))["speak"] and rule.decide(bare(1.5), bare(1.0), kind="launch")["kind"] == "launch"


def test_shipped_rule_loads_and_decides():
    if not S.RULE_PATH.exists():
        pytest.skip("no rule file in this checkout")
    rule = S.SelectiveRule.load()
    raw = json.loads(S.RULE_PATH.read_text())
    assert {"same_topic", "launch"} <= set(raw["ensembles"][raw["default_ensemble"]]["kinds"]) and 0.5 < rule.speak < 1
    assert S.SelectiveRule.load(ensemble="ft10").rule["kinds"]["same_topic"]["model"]["scale"] < rule.rule["kinds"]["same_topic"]["model"]["scale"]
    for kind in ("same_topic", "launch"):
        close = rule.decide(_version(1.105, seed=2), _version(1.10, seed=1), kind=kind)
        far = rule.decide(_version(1.90, seed=2), _version(1.10, seed=1), kind=kind)
        assert 0.4 < close["p_right"] < far["p_right"] < 1 and not close["speak"] and far["pick"] == "a"
        swapped = rule.decide(_version(1.10, seed=1), _version(1.90, seed=2), kind=kind)
        assert swapped["pick"] == "b" and swapped["p_right"] == pytest.approx(far["p_right"]) and swapped["p_a_better"] == pytest.approx(1 - far["p_a_better"])


def test_advisor_uses_the_rule(trained):
    from xviral import advise, serve

    tmp_path, posts, _, _ = trained
    ck = tmp_path / "run" / "best.pt"
    scorer = serve.DraftScorer([ck, ck], embed)
    history = posts.filter(pl.col("author_id") == 23)
    when = int(history["created_at_ms"].max()) + 30 * HOUR
    account = {"handle": "user23", "followers": 46_000, "following": 300}
    spec = {"type": "linear", "features": ["gap", "h_like_sd", "cos_ab", "d_logwords", "sim_centroid_diff", "width_pick"], "impute": [0.1, 0.4, 0.7, 0.0, 0.0, 1.0],
            "mean": [0.0] * 6, "std": [1.0] * 6, "coef": [6.0, 0, 0, 0, 0, 0], "intercept": 0.0, "calibration": None, "max_missing": 0.0}
    rule = S.SelectiveRule({"speak": 0.6, "kinds": {k: {"model": spec, "fallback": {"type": "logistic_gap", "scale": 6.0}} for k in ("same_topic", "launch")}})
    adv = advise.Advisor(scorer, scales={"likes": 6.0}, selective=rule)
    out = adv.advise(history, {"text": "some thoughts on the week ahead"}, account, when,
                     alternatives=[{"text": "launch day: we shipped the new agent today", "label": "launch version"}])
    lv = {c["change"]: c for c in out["changes"]}["launch version"]
    # every input the rule reads was produced by the serving path, so the fitted model answered, not the fallback
    assert lv["confidence_method"] == "linear" and lv["comparison_kind"] == "same_topic" and lv["status"].startswith("recommended")
    rep = scorer.score(history, [{"text": "some thoughts on the week ahead"}], account, when)[0]
    sel = rep["selective"]
    assert sel["history"]["h_n"] > 0 and not math.isnan(sel["novelty"]["sim_centroid"]) and "likes_q90" in rep["members"][0] and "pt_launch" in sel["features"]
    v = S.version_from_report(rep)
    assert v["members"][0]["likes_q10"] <= v["members"][0]["likes_q50"] <= v["members"][0]["likes_q90"]
    # without the rule nothing changes for the product
    plain = advise.Advisor(serve.DraftScorer([ck, ck], embed), scales={"likes": 6.0})
    assert "confidence_method" not in plain.advise(history, {"text": "some thoughts"}, account, when, alternatives=[{"text": "launch day: shipped"}])["changes"][0]
