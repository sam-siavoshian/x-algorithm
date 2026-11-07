import json

import polars as pl
import pytest

from tests.conftest import REAL_USC
from xviral.sources import analytics, usc, user_scrape
from xviral.util.core import parse_pyrepr, snowflake_ms


def test_parse_pyrepr_handles_datetime_and_rejects_code():
    v = parse_pyrepr("{'created': datetime.datetime(2017, 12, 18, 21, 28, 43, tzinfo=datetime.timezone.utc), 'n': 3}")
    assert v == {"created": "2017-12-18T21:28:43+00:00", "n": 3}
    assert parse_pyrepr("__import__('os').system('true')") is None
    assert parse_pyrepr("") is None and parse_pyrepr("nan") is None


def test_usc_mini_chunk(usc_mini_chunk):
    r = usc.read_chunk(usc_mini_chunk)
    p = r["posts"]
    # Ad row dropped, reply dropped by prefilter, original and quote kept.
    assert sorted(p["kind"].to_list()) == ["original", "quote"]
    o = p.filter(pl.col("kind") == "original").row(0, named=True)
    assert o["author_handle"] == "founder" and o["author_id"] == 42
    assert o["author_display_name"] == "Ada Founder"
    assert o["author_followers"] == 2000 and o["author_following"] == 246 and o["author_posts_count"] == 390
    assert o["author_verified"] is True
    assert o["views"] == 15000 and o["likes"] == 120 and o["quotes"] == 2 and o["views_state"] == "count"
    assert o["created_at_ms"] == snowflake_ms(o["post_id"])
    assert abs(o["created_at_ms"] - o["created_at_src_ms"]) <= 1000
    assert o["hashtags_n"] == 1
    m = o["media"][0]
    assert m["type"] == "video" and m["duration_ms"] == 42000 and m["video_url"].endswith("mid.mp4")
    assert m["width"] == 1080 and m["height"] == 1920
    doms = {u["final_domain"]: u["is_media"] for u in o["urls"]}
    assert doms == {"github.com": False, "x.com": True}
    q = p.filter(pl.col("kind") == "quote").row(0, named=True)
    assert q["views"] is None and q["views_state"] == "hidden" and q["lang_src"] == "qme"
    a = r["activity"]
    assert a.filter(pl.col("author_id") == 43)["replies"][0] == 1
    assert r["audit"]["rows_ads"][0] == 1


@pytest.mark.skipif(not REAL_USC.exists(), reason="real USC chunk not checked out")
def test_usc_real_chunk():
    r = usc.read_chunk(REAL_USC)
    p = r["posts"]
    assert p.height > 10_000
    assert p["author_id"].null_count() == 0 and p["author_followers"].null_count() == 0
    assert p["author_handle"].null_count() == 0
    assert (p["created_at_ms"] - p["created_at_src_ms"]).abs().max() <= 1000
    assert abs(r["audit"]["audit_rho"][0]) < 0.1  # counts are mature


def test_user_scrape_detects_nested_layout(tmp_path):
    lines = []
    for i in range(30):
        lines.append({
            "tweet": {"id_str": str(1801041792923578484 + i), "text": f"launch {i}", "created_at": "Wed Jun 12 23:59:59 +0000 2024",
                      "likes": 10 + i, "impressions": 1000, "media_url_https": "https://pbs.twimg.com/media/X.jpg"},
            "author": {"screen_name": "acme", "followers_count": 900, "following_count": 100, "posts_count": 50,
                       "media_count": 5, "is_blue_verified": True},
            "label": {"is_viral": i % 2 == 0},
            "text_embedding": [0.1] * 8,
        })
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines))
    mp = user_scrape.ingest([p], tmp_path / "out")
    assert mp["post_id"] == "tweet.id_str" and mp["handle"] == "author.screen_name" and mp["views"] == "tweet.impressions"
    df = pl.read_parquet(tmp_path / "out" / "posts")
    assert df.height == 30 and df["views"][0] == 1000 and df["author_followers"][0] == 900
    assert df["media"][0][0]["type"] == "photo"
    assert df["created_at_ms"][0] == snowflake_ms(df["post_id"][0])
    assert not df["synthetic_id"].any()
    assert pl.read_parquet(tmp_path / "out" / "old_embeddings").height == 30


def test_user_scrape_without_ids_synthesizes(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps({"text": f"t{i}", "created_at": "2025-01-0%dT10:00:00Z" % (i % 9 + 1),
                                       "screen_name": "acme", "likes": 1, "impressions": 10}) for i in range(10)))
    user_scrape.ingest([p], tmp_path / "out")
    df = pl.read_parquet(tmp_path / "out" / "posts")
    assert df["synthetic_id"].all() and (df["post_id"] < 0).all() and (df["author_id"] < 0).all()
    assert df["post_id"].n_unique() == 10


@pytest.mark.parametrize("headers", [
    ["Post id", "Date", "Post text", "Post Link", "Impressions", "Likes", "Engagements", "Bookmarks", "Shares",
     "New follows", "Replies", "Reposts", "Profile visits", "Detail Expands", "URL Clicks"],
    ["Tweet id", "Tweet permalink", "Tweet text", "time", "impressions", "engagements", "retweets", "replies",
     "likes", "user profile clicks", "url clicks", "detail expands", "follows"],
])
def test_analytics_layouts(tmp_path, headers):
    pid = 1801041792923578484
    vals = {h: "7" for h in headers}
    for h in headers:
        hl = h.lower()
        if "id" in hl and "text" not in hl:
            vals[h] = str(pid)
        if "link" in hl or "permalink" in hl:
            vals[h] = f"https://x.com/acme/status/{pid}"
        if "text" in hl:
            vals[h] = "launch day"
        if hl in ("date", "time"):
            vals[h] = "2024-06-12 23:59"
        if hl == "impressions":
            vals[h] = "1,234"
    p = tmp_path / "a.csv"
    pl.DataFrame([vals]).write_csv(p)
    df = analytics.read_export(p)
    r = df.row(0, named=True)
    assert r["post_id"] == pid and r["views"] == 1234 and r["author_handle"] == "acme"
    assert r["profile_clicks"] == 7 and r["detail_expands"] == 7 and r["source"] == "analytics"


def test_analytics_unknown_layout_fails(tmp_path):
    p = tmp_path / "a.csv"
    pl.DataFrame([{"foo": 1, "bar": 2}]).write_csv(p)
    with pytest.raises(ValueError):
        analytics.read_export(p)


class _FakeCA:
    """PostgREST stand-in for the tweets table: honours order, gte on created_at, and limit."""

    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: (r["created_at"], r["tweet_id"]))
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        start = params["created_at"].split(".", 1)[1]
        out = [r for r in self.rows if r["created_at"] >= start][: int(params["limit"])]

        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self_inner):
                return out

        return R()


def test_ca_time_keyset_pagination_has_no_gaps_or_duplicates(tmp_path, monkeypatch):
    from xviral.config import CommunityArchiveCfg
    from xviral.sources.community_archive import RestDownloader

    monkeypatch.setenv("CA_ANON_KEY", "x")
    # 23 tweets over 7 distinct timestamps, with ties straddling page boundaries (page size 4).
    rows = [{"tweet_id": str(1000 + i), "created_at": f"2025-01-0{1 + i // 4}T00:00:0{i % 2}+00:00", "account_id": "1",
             "full_text": "t", "retweet_count": 0, "favorite_count": 0, "reply_to_tweet_id": None,
             "reply_to_user_id": None, "archive_upload_id": 1} for i in range(23)]
    cfg = CommunityArchiveCfg(page_size=4, requests_per_second=1e6, min_created="2025-01-01")
    n = RestDownloader(cfg, tmp_path, session=_FakeCA(rows)).download("tweets", flush_rows=5)
    got = pl.read_parquet(tmp_path / "tweets" / "*.parquet")["tweet_id"].to_list()
    assert n == 23 and sorted(got) == sorted(r["tweet_id"] for r in rows) and len(set(got)) == 23
