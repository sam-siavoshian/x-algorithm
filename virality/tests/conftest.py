import gzip
import io
import random
from pathlib import Path

import polars as pl
import pytest

from xviral.schema import conform
from xviral.util.core import TWITTER_EPOCH_MS

REAL_USC = Path("/home/user/sinking8/usc-x-24-us-election/part_1/may_july_chunk_1.csv.gz")


def snowflake(ms: int, seq: int = 0) -> int:
    return ((ms - TWITTER_EPOCH_MS) << 22) | (seq & 0x3FFFFF)


def make_post(post_id: int, author_id: int, ms: int, text: str = "hello", **kw) -> dict:
    base = {
        "post_id": post_id, "author_id": author_id, "author_handle": f"user{author_id}", "created_at_ms": ms,
        "created_at_src_ms": ms, "snapshot_at_ms": ms + 5 * 86_400_000, "snapshot_kind": "exact", "text": text,
        "lang_src": "en", "kind": "original", "is_reply": False, "media": [], "urls": [], "mentions_n": 0,
        "hashtags_n": 0, "views": 1000, "views_state": "count", "likes": 10, "replies": 1, "reposts": 1, "quotes": 0,
        "author_followers": 500, "author_following": 300, "source": "usc", "source_file": "f", "synthetic_id": False,
    }
    base.update(kw)
    return base


def frame(rows: list[dict]) -> pl.DataFrame:
    return conform(pl.DataFrame(rows, infer_schema_length=None, schema_overrides={
        "media": conform(pl.DataFrame()).schema["media"], "urls": conform(pl.DataFrame()).schema["urls"]}))


@pytest.fixture
def usc_mini_chunk(tmp_path) -> Path:
    """A 4-row chunk in USC's exact on-disk format: Python reprs, datetime calls, float IDs."""
    header = ['', 'id', 'text', 'url', 'epoch', 'media', 'retweetedTweet', 'retweetedTweetID', 'retweetedUserID', 'id_str',
              'lang', 'rawContent', 'replyCount', 'retweetCount', 'likeCount', 'quoteCount', 'conversationId',
              'conversationIdStr', 'hashtags', 'mentionedUsers', 'links', 'viewCount', 'quotedTweet',
              'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id_str', 'location',
              'cash_app_handle', 'user', 'date', '_type', 'type']
    ms = 1_718_236_799_000
    pid = snowflake(ms, 1)

    def user(uid, handle_display, followers):
        return ("{'id': %d, 'id_str': '%d', 'url': 'https://twitter.com/%s', 'username': '%s', "
                "'rawDescription': 'Founder. Building AI tools.', 'created': datetime.datetime(2017, 12, 18, 21, 28, 43, "
                "tzinfo=datetime.timezone.utc), 'followersCount': %d, 'friendsCount': 246, 'statusesCount': 390, "
                "'favouritesCount': 19, 'listedCount': 0, 'mediaCount': 1, 'location': 'SF', 'verified': False, "
                "'blue': True, 'blueType': None}") % (uid, uid, handle_display, handle_display, followers)

    media = ("[{'media_url_https': 'https://pbs.twimg.com/media/A.jpg', 'type': 'video', 'url': 'https://t.co/m1', "
             "'expanded_url': 'https://twitter.com/founder/status/%d/video/1', 'original_info': {'height': 1920, 'width': 1080}, "
             "'video_info': {'aspect_ratio': [9, 16], 'duration_millis': 42000, 'variants': [{'bitrate': 256000, "
             "'content_type': 'video/mp4', 'url': 'https://video.twimg.com/low.mp4'}, {'bitrate': 832000, "
             "'content_type': 'video/mp4', 'url': 'https://video.twimg.com/mid.mp4'}]}}]") % pid
    links = "[{'url': 'https://t.co/l1', 'expanded_url': 'https://github.com/acme/tool', 'display_url': 'github.com/acme'}]"
    rows = [
        ['0', str(pid), 'We launched today https://t.co/l1 https://t.co/m1', f'https://twitter.com/founder/status/{pid}', str(ms / 1000),
         media, 'False', '', '', str(pid), 'en', 'We launched today https://t.co/l1 https://t.co/m1', '3', '5', '120.0', '2.0',
         '1.8e+18', '1.8e+18', "['launch']", '[]', links, "{'count': '15000', 'state': 'EnabledWithCount'}", 'False',
         '', '', '', '', '', user(42, 'Ada Founder', 2000), '2024-06-12', '', 'tweet-'],
        ['1', str(pid + 1), '@x replying', f'https://twitter.com/bob/status/{pid + 1}', str(ms / 1000), '[]', 'False', '', '',
         str(pid + 1), 'en', '@x replying', '0', '0', '0.0', '0.0', '1.8e+18', '1.8e+18', '[]',
         "[{'id_str': '1', 'screen_name': 'x'}]", '[]', "{'count': '10', 'state': 'EnabledWithCount'}", 'False',
         'x', '1.8006810262489418e+18', '1.4e+18', '', '', user(43, 'Bob', 100), '2024-06-12', '', 'tweet-'],
        ['2', str(pid + 2), 'quote of something', f'https://twitter.com/carol/status/{pid + 2}', str(ms / 1000), '[]', 'False',
         '', '', str(pid + 2), 'qme', 'quote of something', '0', '0', '4.0', '0.0', '', '', '[]', '[]', '[]',
         "{'state': 'Enabled'}", 'True', '', '', '', '', '', user(44, 'Carol', 50), '2024-06-12', '', 'tweet-'],
        ['3', '', 'Buy now ad text', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', ''],
    ]
    buf = io.StringIO()
    import csv

    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    p = tmp_path / "part_1" / "may_july_chunk_9.csv.gz"
    p.parent.mkdir(parents=True)
    p.write_bytes(gzip.compress(buf.getvalue().encode()))
    return p


@pytest.fixture
def rng():
    return random.Random(0)
