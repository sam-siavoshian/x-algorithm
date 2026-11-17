import json

import polars as pl
import pytest

from xviral.model import features
from xviral.sources import home_scrape


def _raw(n=3, **over):
    ids = ["1975320914595111412", "1975221398730928395", "1993663526062186842"][:n]
    created = ["Mon Oct 06 22:03:00 +0000 2025", "Mon Oct 06 15:27:33 +0000 2025", "Wed Nov 26 12:49:59 +0000 2025"][:n]
    d = {"id_str": ids, "text": ["a https://t.co/x", "b", "RT @x: c"][:n], "user_name": ["A", "B", "C"][:n], "screen_name": ["Aa", "bb", "cc"][:n],
         "likes": [904, 100, 55][:n], "impressions": [13744, 3541, 4820][:n], "created_at": created,
         "saved_at": ["2025-12-03 14:26:24.538313+00:00"] * n,
         "media_url_https": [json.dumps(["https://pbs.twimg.com/amplify_video_thumb/1/img/a.jpg"]), json.dumps(["https://pbs.twimg.com/media/G2.jpg", "https://pbs.twimg.com/media/G3.jpg"]), "[]"][:n],
         "post_image_description": ["NaN", "A man at a desk.", None][:n], "creator_is_blue_verified": [True] * n, "creator_followers_count": [399482, 2278, 10][:n],
         "creator_following_count": [55, 990, 1][:n], "creator_posts_count": [1, 2, 3][:n], "creator_media_count": [1, 2, 3][:n]}
    d.update(over)
    return pl.DataFrame(d)


def test_ids_exact_and_fields(tmp_path):
    p = tmp_path / "posts.parquet"
    _raw().write_parquet(p)
    d, rep = home_scrape.read(p)
    assert sorted(d["post_id"].to_list()) == sorted([1975320914595111412, 1975221398730928395, 1993663526062186842])   # no float on the way
    assert rep["timestamp_agrees_with_id"] == 1.0 and rep["bad_ids"] == 0
    r = {x["post_id"]: x for x in d.to_dicts()}
    a = r[1975221398730928395]
    assert a["views"] == 3541 and a["snapshot_kind"] == "exact" and a["source"] == "home_scrape" and a["kind"] == "original"
    assert [m["type"] for m in a["media"]] == ["photo", "photo"] and a["media"][0]["alt"] == "A man at a desk." and a["media"][1]["alt"] is None
    assert r[1975320914595111412]["media"][0]["type"] == "video" and r[1975320914595111412]["media"][0]["alt"] is None       # the string "NaN" is no description
    assert r[1993663526062186842]["kind"] == "repost"
    assert home_scrape.image_text(d).height == 1
    assert home_scrape.text_with_image("b", "A man at a desk.") == "b\n[image: A man at a desk.]"


def test_float_ids_refused(tmp_path):
    p = tmp_path / "posts.parquet"
    _raw().with_columns(pl.col("id_str").cast(pl.Float64)).write_parquet(p)
    with pytest.raises(ValueError):
        home_scrape.read(p)


def test_guard_on_counts(tmp_path):
    p = tmp_path / "posts.parquet"
    _raw(likes=[10**6, 10**6, 10**6]).write_parquet(p)
    with pytest.raises(ValueError):
        home_scrape.read(p)


def test_capture_age_regime():
    d = pl.DataFrame({"author_id": [1] * 4, "source": ["home_scrape"] * 3 + ["community_archive"], "created_at_ms": [0] * 4, "snapshot_at_ms": [int(h * 3.6e6) for h in (0.5, 8, 100, 1)]})
    assert features.with_src_regime(d)["src_regime"].to_list() == ["home_0_1h", "home_6_12h", "home_24h_plus", "stream_sample"]
    assert dict(features.CAT_SPEC)["src_regime"][:4] == ["unknown", "stream_sample", "search_sample", "full_timeline"]     # earlier tables keep their codes
