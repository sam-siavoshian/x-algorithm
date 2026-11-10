"""Phase 0 end to end on synthetic data in every source format (USC CSV, CA tables,
scrape JSONL, analytics CSV), with the CPU relevance embedder."""

import csv
import gzip
import io
import json

import numpy as np
import polars as pl

from tests.conftest import snowflake
from xviral import pipeline
from xviral.config import Config

DAY = 86_400_000
T0 = 1_735_689_600_000  # 2025-01-01

TECH = ["shipping our new open source agent framework", "we raised a seed round to build developer tools",
        "launching an api for fine tuning llms", "our yc batch demo day is tomorrow", "new gpu inference benchmark results",
        "hiring founding engineers in sf", "the model now supports function calling", "we open sourced the eval harness"]
OTHER = ["what a game last night", "my cat knocked over the plant again", "best pizza in town", "traffic is terrible",
         "watching the new season tonight", "gym session done", "rainy sunday mood", "coffee first then everything"]


def _usc_user(uid, name, followers):
    return ("{'id': %d, 'id_str': '%d', 'url': 'https://twitter.com/%s', 'username': '%s', 'rawDescription': 'bio', "
            "'created': datetime.datetime(2015, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc), 'followersCount': %d, "
            "'friendsCount': 300, 'statusesCount': 1000, 'favouritesCount': 10, 'listedCount': 0, 'mediaCount': 5, "
            "'verified': False, 'blue': False}") % (uid, uid, name, name, followers)


def build_sources(root, rng):
    header = ['', 'id', 'text', 'url', 'epoch', 'media', 'retweetedTweet', 'retweetedTweetID', 'retweetedUserID', 'id_str',
              'lang', 'rawContent', 'replyCount', 'retweetCount', 'likeCount', 'quoteCount', 'conversationId',
              'conversationIdStr', 'hashtags', 'mentionedUsers', 'links', 'viewCount', 'quotedTweet',
              'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id_str', 'location',
              'cash_app_handle', 'user', 'date', '_type', 'type']
    usc_dir = root / "usc" / "part_1"
    usc_dir.mkdir(parents=True)
    seq = 0
    for chunk in range(3):
        rows = []
        for i in range(700):
            seq += 1
            aid = 1000 + int(rng.integers(0, 120))
            tech = aid < 1040
            ms = T0 + int(rng.integers(0, 120)) * DAY + int(rng.integers(0, DAY))
            pid = snowflake(ms, seq)
            followers = int(10 ** rng.uniform(2, 5))
            quality = 1.5 if tech else 1.0
            views = int(followers * 10 ** rng.normal(-0.5, 0.6) * quality)
            text = (TECH if tech else OTHER)[int(rng.integers(0, 8))] + f" post {i}"
            reply = rng.random() < 0.3
            rows.append(['0', str(pid), text, f'https://twitter.com/u{aid}/status/{pid}', str(ms / 1000), '[]', 'False', '', '',
                         str(pid), 'en', text, '1', '2', str(float(int(views * 0.02))), '0.0', '', '', '[]', '[]', '[]',
                         "{'count': '%d', 'state': 'EnabledWithCount'}" % views, 'False', 'x' if reply else '', '1.0e18' if reply else '',
                         '', '', '', _usc_user(aid, f"u{aid}", followers), '2025-01-01', '', 'tweet-'])
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(header)
        w.writerows(rows)
        (usc_dir / f"chunk_{chunk}.csv.gz").write_bytes(gzip.compress(buf.getvalue().encode()))

    # Community Archive tables in the REST schema.
    ca = root / "ca"
    tw, acc, fol, fing = [], [], [], []
    for aid in range(5000, 5040):
        acc.append({"account_id": str(aid), "username": f"ca{aid}", "created_at": "2014-01-01T00:00:00+00:00",
                    "account_display_name": f"CA {aid}", "num_tweets": 500, "num_following": 400, "num_followers": 800,
                    "num_likes": 10, "updated_at": "2025-08-01T00:00:00+00:00", "is_tombstone": False})
        for j in range(30):
            seq += 1
            ms = T0 + int(rng.integers(0, 120)) * DAY
            tw.append({"tweet_id": str(snowflake(ms, seq)), "account_id": str(aid), "created_at": None,
                       "full_text": TECH[(j + aid) % 8] + f" post {j}", "retweet_count": int(rng.integers(0, 20)),
                       "favorite_count": int(rng.integers(0, 200)), "reply_to_tweet_id": None, "reply_to_user_id": None,
                       "archive_upload_id": 1, "updated_at": "2025-08-01T00:00:00+00:00"})
        for k in range(20):
            fol.append({"id": len(fol), "account_id": str(aid), "follower_account_id": str(9000 + k)})
            if k % 2 == 0:
                fing.append({"id": len(fing), "account_id": str(aid), "following_account_id": str(9000 + k)})
    from datetime import datetime, timezone

    for t in tw:
        ms = (int(t["tweet_id"]) >> 22) + 1288834974657
        t["created_at"] = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
    # One stream-captured tweet read 5 seconds after posting: must be dropped as immature.
    ms = T0 + 50 * DAY
    tw.append({"tweet_id": str(snowflake(ms, 4_000_000)), "account_id": "5000", "created_at": None, "full_text": TECH[0] + " stream",
               "retweet_count": 0, "favorite_count": 0, "reply_to_tweet_id": None, "reply_to_user_id": None,
               "archive_upload_id": None, "updated_at": datetime.fromtimestamp(ms / 1000 + 5, tz=timezone.utc).isoformat()})
    for t in tw:
        if t["created_at"] is None:
            t["created_at"] = datetime.fromtimestamp(((int(t["tweet_id"]) >> 22) + 1288834974657) / 1000, tz=timezone.utc).isoformat()
    up = [{"id": 1, "account_id": "5000", "username": "ca5000", "archive_at": "2025-08-01T00:00:00+00:00", "upload_phase": "completed"}]
    (ca / "archive_upload").mkdir(parents=True)
    pl.DataFrame(up).write_parquet(ca / "archive_upload" / "part-00000.parquet")
    for name, rows in (("tweets", tw), ("all_account", acc), ("followers", fol), ("following", fing)):
        (ca / name).mkdir(parents=True)
        pl.DataFrame(rows).write_parquet(ca / name / "part-00000.parquet")

    # The tech scrape: seed accounts.
    lines = []
    for aid in range(7000, 7030):
        for j in range(25):
            seq += 1
            ms = T0 + int(rng.integers(0, 120)) * DAY
            followers = int(10 ** rng.uniform(3, 5))
            views = int(followers * 10 ** rng.normal(-0.3, 0.6))
            lines.append({"id_str": str(snowflake(ms, seq)), "text": TECH[(j * 3 + aid) % 8] + f" post {j}", "created_at": datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(),
                          "likes": int(views * 0.03), "impressions": views, "screen_name": f"seed{aid}",
                          "followers_count": followers, "following_count": 500, "is_viral": views > followers})
    scrape = root / "scrape.jsonl"
    scrape.write_text("\n".join(json.dumps(x) for x in lines))
    return root / "usc", ca, scrape


def test_phase0_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    usc_root, ca, scrape = build_sources(tmp_path / "raw", rng)
    import os
    import time

    future = time.time() + 400 * 86400  # scrape file "taken" well after every post: counts mature
    os.utime(scrape, (future, future))
    cfg = Config.model_validate({
        "paths": {"work": str(tmp_path / "work"), "usc_root": str(usc_root), "community_archive": str(ca), "user_scrape": [str(scrape)]},
        "relevance": {"embedder": "hashing", "negatives_sample": 5000},
        "workers": 2,
    })
    head = pipeline.phase0(cfg)
    w = tmp_path / "work"
    for p in ("unified/posts.parquet", "filtered/posts.parquet", "authors/trailing.parquet", "relevance/posts.parquet",
              "relevance/audit_to_label.csv", "features/posts.parquet", "dataset/rows.parquet", "dataset/pairs.parquet",
              "report/counts.csv", "report/headline.json", "report/baseline.json"):
        assert (w / p).exists(), p
    rows = pl.read_parquet(w / "dataset" / "rows.parquet")
    assert set(rows["split"].unique()) == {"train", "val", "test"}
    assert (rows["kind"].is_in(["original", "quote"])).all()
    assert rows["source"].is_in(["usc", "community_archive", "user_scrape"]).all()
    # Relevance keeps tech accounts it has never seen (USC tech accounts, CA donors) and drops off-topic ones.
    usc_rows = rows.filter(pl.col("source") == "usc")
    assert usc_rows.filter(pl.col("author_id") < 1040).height > 100
    assert usc_rows.filter(pl.col("author_id") >= 1040).height < 0.1 * usc_rows.filter(pl.col("author_id") < 1040).height
    assert rows.filter(pl.col("source") == "community_archive").height > 600
    assert rows.filter(pl.col("text").str.ends_with(" stream")).height == 0  # immature stream capture dropped
    # Mutual share is exact for CA donors: 10 of 20 followers followed back.
    a = pl.read_parquet(w / "authors" / "authors.parquet").filter(pl.col("author_id") == 5000)
    assert abs(a["mutual_share_exact"][0] - 0.5) < 1e-9
    b = head["baseline"]
    assert b["n_test"] > 0 and not b["leak_suspected"]
    assert abs(b["shuffled_auc_y_reach_10x_test"] - 0.5) < 0.1 or np.isnan(b["shuffled_auc_y_reach_10x_test"])
