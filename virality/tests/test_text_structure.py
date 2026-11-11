"""Candidate text-structure features: values on a known post, and no look-ahead in the self-relative group."""

import polars as pl

from xviral.features import text_structure as ts

T0 = 1_760_000_000_000
HOUR = 3_600_000
LAUNCH = "Introducing Foo: the fastest agent runtime.\n\n- 10x faster\n- $5M raised\n\nTry it https://foo.dev"


def _posts(texts):
    return pl.DataFrame([{"post_id": i + 1, "author_id": 1, "created_at_ms": T0 + i * 7 * HOUR, "text": t, "has_media": i % 2 == 0, "has_url": "http" in t}
                         for i, t in enumerate(texts)])


def test_values_on_a_launch_post():
    r = ts.text_structure(_posts([LAUNCH, "Why did we not ship this sooner? 🧵\n1/ I think you never know", "https://x.com/a first\nthen text", ""]))
    a, b, c, d = r.sort("post_id").iter_rows(named=True)
    assert a["ts_open_announce"] == 1 and a["ts_first_line_words"] == 6 and a["ts_lines"] == 4 and a["ts_blank_lines"] == 2
    assert a["ts_bullet_lines"] == 2 and a["ts_money"] == 1 and a["ts_url_at_end"] == 1 and a["ts_url_first_line"] == 0
    assert a["ts_superlative"] == 1 and a["ts_call_to_action"] == 1 and 0.8 < a["ts_chars_before_url_share"] < 0.9
    assert b["ts_open_question"] == 1 and b["ts_thread_marker"] == 1 and b["ts_emoji"] == 1 and b["ts_we"] == 1 and b["ts_negation"] == 2
    assert c["ts_url_first_line"] == 1 and c["ts_chars_before_url_share"] == 0 and b["ts_chars_before_url_share"] is None
    assert d["ts_words"] == 0 and d["ts_lines"] == 0                      # an empty post is all zeros, not an error
    assert set(r.columns) == {"post_id", *ts.ALL} and len(ts.ALL) == len(set(ts.ALL))


def test_self_relative_group_uses_only_earlier_posts():
    texts = [f"post number {i} " + "word " * (i % 7) for i in range(30)]
    base = ts.text_structure(_posts(texts)).sort("post_id")
    changed = list(texts)
    changed[20] = LAUNCH * 3
    alt = ts.text_structure(_posts(changed)).sort("post_id")
    cols = ts.GROUPS["self_relative"]
    assert base.head(20).select(cols).equals(alt.head(20).select(cols))   # rows before the edit are untouched
    assert base[20, "ts_len_vs_own"] != alt[20, "ts_len_vs_own"]
    assert base.head(5).select(cols[0]).null_count().item() == 5          # no value until five earlier posts exist
