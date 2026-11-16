"""Readers and rules added by the data track: twitterapi.io pages, twitter100m whole timelines, nitter replies, bio rules."""
import json

import polars as pl
import pytest

from xviral.schema import POSTS_SCHEMA
from xviral.sources import nitter_sample, twitter100m, twitterapi_io
from xviral.text.bio import bio_flags, bio_kind

EPOCH = 1288834974657


def sf(ms: int, seq: int = 0) -> int:
    return ((ms - EPOCH) << 22) + seq


def _tw(i, ms, **kw):
    created = pl.from_epoch(pl.Series([ms]), "ms").dt.strftime("%a %b %d %H:%M:%S +0000 %Y")[0]
    t = {"type": "tweet", "id": str(sf(ms, i)), "text": f"post {i}", "retweetCount": 1, "replyCount": 2, "likeCount": 10, "quoteCount": 0, "viewCount": 1000, "createdAt": created, "lang": "en",
         "bookmarkCount": 3, "isReply": False, "inReplyToId": None, "conversationId": str(sf(ms, i)), "inReplyToUserId": None, "inReplyToUsername": None,
         "author": {"userName": "Acme", "id": "42", "name": "Acme", "followers": 900, "following": 5, "statusesCount": 10, "mediaCount": 1, "isBlueVerified": True,
                    "createdAt": "Sat Nov 16 01:50:21 +0000 2013", "description": "", "profile_bio": {"description": "The API for rockets."}},
         "extendedEntities": {}, "entities": {"urls": [{"url": "https://t.co/x", "expanded_url": "https://acme.dev/launch"}], "user_mentions": [], "hashtags": [{"text": "a"}]},
         "quoted_tweet": None, "retweeted_tweet": None}
    t.update(kw)
    return t


def test_twitterapi_io_reader(tmp_path):
    ms = 1735689600000   # 2025-01-01
    root = _tw(1, ms)
    reply = _tw(2, ms + 60_000, isReply=True, inReplyToId=root["id"], inReplyToUserId="42", inReplyToUsername="Acme", conversationId=root["id"])
    old = _tw(3, 1600000000000, viewCount=0)                       # before view counts: views must be null
    dup = dict(root)                                                # the same post on a second page
    f = tmp_path / "Acme_42.json"
    f.write_text(json.dumps([{"data": {"tweets": [root, reply]}}, {"data": {"tweets": [old, dup]}}]))
    posts, side, rep = twitterapi_io.read_file(f, "2025-02-01", expect_author_id=42)
    assert list(posts.schema.items()) == list(POSTS_SCHEMA.items())
    assert posts.height == 3 and rep["raw"] == 4 and rep["time_mismatch"] == 0
    r = posts.filter(pl.col("post_id") == int(root["id"])).row(0, named=True)
    assert (r["views"], r["likes"], r["bookmarks"], r["author_followers"], r["author_bio"], r["kind"]) == (1000, 10, 3, 900, "The API for rockets.", "original")
    assert r["created_at_ms"] == ms and r["urls"][0]["final_domain"] == "acme.dev"
    assert posts.filter(pl.col("post_id") == int(old["id"]))["views"][0] is None
    s = side.filter(pl.col("post_id") == int(reply["id"])).row(0, named=True)
    assert s["is_self_reply"] and s["in_reply_to_post_id"] == int(root["id"])


def test_twitterapi_io_refuses_shifted_counts(tmp_path):
    ms = 1735689600000
    tweets = [_tw(i, ms + i * 1000, likeCount=5000, viewCount=10) for i in range(250)]     # "likes" above "views": not the fields the reader expects
    f = tmp_path / "Acme_42.json"
    f.write_text(json.dumps([{"data": {"tweets": tweets}}]))
    with pytest.raises(ValueError):
        twitterapi_io.read_file(f, "2025-02-01")


def test_twitter100m_whole_timeline(tmp_path):
    ms = 1640995200000   # 2022-01-01
    iso = lambda m: pl.from_epoch(pl.Series([m]), "ms").dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")[0]
    rows = [("Maker", sf(ms, 1), "We just launched our API", 1, 2, 30, 0, iso(ms)), ("Maker", sf(ms + 5000, 2), "@friend thanks!", 0, 0, 1, 0, iso(ms + 5000)),
            ("Maker", sf(ms + 9000, 3), "RT @x: hi", 0, 9, 0, 0, iso(ms + 9000)), ("Maker", sf(ms + 20000, 4), "bad clock", 0, 0, 0, 0, iso(ms + 999_000)), ("Other", sf(ms, 5), "hello", 0, 0, 2, 0, iso(ms))]
    tp = tmp_path / "t.parquet"
    pl.DataFrame(rows, schema=["user", "id", "tweet", "replies", "retweets", "likes", "quotes", "date"], orient="row").write_parquet(tp)
    up = tmp_path / "u.parquet"
    pl.DataFrame({"user": ["Maker", "Other"], "id": [7, 8], "verified": [False, False], "followers": [120, 5], "description": ["Founder of @acme, the API for rockets", ""], "location": ["", ""]}).write_parquet(up)
    users = twitter100m.all_users(up)
    df = twitter100m.read_shard_accounts(tp, users, pl.Series(["maker"]))
    assert list(df.schema.items()) == list(POSTS_SCHEMA.items())
    assert sorted(df["kind"].to_list()) == ["original", "reply"]            # the repost and the row whose date disagrees with its id are gone
    assert df["author_id"].unique().to_list() == [7] and df["author_followers"][0] == 120 and df["author_bio"][0].startswith("Founder")
    assert twitter100m.check_shard(tp)["time_mismatch"] == 1


def test_nitter_replies_and_parent_link(tmp_path):
    ms = 1767225600000   # 2026-01-01
    iso = lambda m: pl.from_epoch(pl.Series([m]), "ms").dt.strftime("%Y-%m-%d %H:%M:00+00:00")[0]
    def row(i, m, text, to, ext=None):
        return {"id": str(sf(m, i)), "text": text, "name": "Acme", "username": "@Acme", "date": iso(m), "is-retweet": False, "replying-to": to, "external-link": ext,
                "quoted-post": {"date": None, "link": None, "text": None, "user": None, "pictures": None, "videos": None, "gifs": None}, "comments": 1, "retweets": 2, "quotes": 30, "likes": 4000,
                "pictures": [], "videos": [], "gifs": []}
    rows = [row(1, ms, "Introducing Acme", []), row(2, ms + 120_000, "Try it here", ["@Acme"], "https://acme.dev"), row(3, ms + 300_000, "thanks", ["@someone"])]
    p = tmp_path / "s.parquet"
    pl.DataFrame(rows, schema_overrides={"replying-to": pl.List(pl.String), "pictures": pl.List(pl.String), "videos": pl.List(pl.String), "gifs": pl.List(pl.String), "external-link": pl.String}).write_parquet(p)
    posts, side = nitter_sample.read_shard_accounts(p, pl.Series(["acme"]))
    assert posts.height == 3 and posts["is_reply"].sum() == 2
    assert posts.filter(pl.col("kind") == "original").select("likes", "views").row(0) == (30, 4000)       # the column shift is undone
    assert side.filter(pl.col("is_self_reply")).height == 1 and side.filter(pl.col("is_self_reply"))["has_link"][0]
    link = nitter_sample.link_self_replies(posts, side)
    assert link.height == 1 and link["parent_post_id"][0] == sf(ms, 1) and link["parent_is_root"][0]


def test_bio_rules():
    d = pl.DataFrame({"author_bio": ["Founder @acme (YC W21). Building dev tools.", "The open-source platform for AI agents. Join us: https://x.dev", "Dad, runner, engineer at BigCo. Views my own.",
                                     "Founder of MoonCoin, the best crypto token", "Life coach & founder of Thrive Coaching"], "author_display_name": [None] * 5})
    k = d.with_columns(bio_flags()).with_columns(bio_kind().alias("k"))["k"].to_list()
    assert k == ["founder", "company", "none", "none", "none"]
