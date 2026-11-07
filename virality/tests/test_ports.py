"""Exact ports from the X repository: weighted text length, Phoenix transforms, formulas."""

import math

import numpy as np
import pytest

from xviral import constants as C
from xviral.features import phoenix as ph
from xviral.text.weighted_length import _ASCII_WS, _UNICODE_WS, char_weight, text_metrics

# Samples copied from home-mixer/models/content_features.rs `matches_the_reference_implementation`.
RUST_SAMPLES = [
    "", "   ", "hello world", "  leading and trailing  ", "see https://t.co/abc123 now",
    "https://t.co/start then http://x.y/z and end https://t.co/end",
    "not a url: http:// alone, https:/broken, xhttps://inside, HTTPS://upper",
    "line one\nline two\thttps://t.co/x\r\n", "vertical\x0btab\x0cfeed\x85next-line",
    "日本　語 and ｆｕｌｌｗｉｄｔｈ", "ok 👍🏽 ❤️ 🏳️‍🌈 \U000e0067tag", "Ça va très bien, ünïcödé ÀÉÎÕÜ",
    " em thin\xa0nbsp sep", "‐―“quotes”′‷", "ჿᄀᄁ boundary",
]


def _is_ws(ch: str) -> bool:
    cp = ord(ch)
    return cp in _ASCII_WS or cp in _UNICODE_WS


def _reference(text: str, attach: int) -> tuple[int, int]:
    """The Rust test's own `reference` function, transcribed."""
    ws = sum(char_weight(ord(c)) for c in text if _is_ws(c))
    urls = words = 0
    token = []
    tokens = []
    for c in text:
        if _is_ws(c):
            if token:
                tokens.append("".join(token))
                token = []
        else:
            token.append(c)
    if token:
        tokens.append("".join(token))
    for t in tokens:
        if t.startswith("https://") or t.startswith("http://"):
            urls += 1
        else:
            words += sum(char_weight(ord(c)) for c in t)
    ext = max(urls - attach, 0)
    return ws + words + C.URL_WEIGHTED_LEN * ext, ext


@pytest.mark.parametrize("text", RUST_SAMPLES)
@pytest.mark.parametrize("attach", [0, 1, 2])
def test_weighted_len_matches_rust_reference(text, attach):
    m = text_metrics(text, attach)
    assert (m.weighted_len, m.external_urls) == _reference(text, attach)


def test_weighted_len_rust_assertions():
    w = lambda t, a=0: text_metrics(t, a)  # noqa: E731
    assert w("").weighted_len == 0
    assert w("hello world").weighted_len == 11
    assert w("日本語").weighted_len == 6
    assert w("see https://t.co/abc123 now").weighted_len == 3 + 1 + 23 + 1 + 3
    assert w("ok 👍🏽").weighted_len == 2 + 1 + 2
    assert w("日本　語").weighted_len == 4 + 2 + 2
    mp = w("So nice ❤️ https://t.co/kKPUyA0ztX", 1)
    assert mp.external_urls == 0 and mp.weighted_len == 2 + 1 + 4 + 1 + 2 + 1
    ml = w("read https://t.co/link https://t.co/media", 1)
    assert ml.external_urls == 1 and ml.weighted_len == 4 + 2 + 23
    assert w("line one\nline two https://t.co/x", 1).weighted_len == 8 + 1 + 8 + 1


def test_exploration_target_matches_recsys_batch():
    # At the 24 h window the target is exactly 3% of followers.
    assert C.exploration_view_target(10_000, 24) == pytest.approx(300.0)
    decay = math.log(2) / 8
    expect = 0.03 * 10_000 * (1 - math.exp(-decay * 8)) / (1 - math.exp(-decay * 24))
    assert C.exploration_view_target(10_000, 8) == pytest.approx(expect)
    assert C.exploration_view_target(10_000, 100) == pytest.approx(300.0)  # clamped


def test_popular_quality_matches_rust():
    lam = math.log(2) / 8
    sat = 1 - math.exp(-lam * 24)
    assert C.popular_quality(1000, 2) == pytest.approx(1000 * sat / (1 - math.exp(-lam * 2)))
    assert C.popular_quality(1000, 0.1) == pytest.approx(C.popular_quality(1000, 0.5))  # 0.5 h floor


def test_post_age_bucket():
    base = np.array([1_700_000_000])
    assert ph.post_age_bucket(base, base)[0] == 1          # 0 minutes -> bucket 1
    assert ph.post_age_bucket(base + 3599, base)[0] == 1
    assert ph.post_age_bucket(base + 3600, base)[0] == 2
    assert ph.post_age_bucket(base + 80 * 3600, base)[0] == 81  # overflow bucket
    assert ph.post_age_bucket(base + 500 * 3600, base)[0] == 81
    assert ph.post_age_bucket(base - 60, base)[0] == 0      # negative age -> unknown
    assert ph.post_age_bucket(np.array([0]), base)[0] == 0
    assert C.POST_AGE_NUM_BUCKETS == 82


def test_count_log_features():
    f = ph.count_log_features(np.array([[0, 1, 3, 7, 1023]]))
    np.testing.assert_allclose(f, [[0, 1 / 32, 2 / 32, 3 / 32, 10 / 32, 1.0]], rtol=1e-6)
    assert ph.count_log_features(np.array([[-5, 0, 0, 0, 0]]))[0, 0] == 0


def test_local_hour_index():
    ts = np.array([0, 12 * 3600, 1])
    idx = ph.local_hour_index(ts, np.array([0.0, -8.0, np.nan]))
    assert idx.tolist() == [0, 5, 0]  # ts=0 is unknown; 12:00 UTC is 04:00 PT -> index 5; NaN offset unknown
