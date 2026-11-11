import io
import shutil
import subprocess

import numpy as np
import polars as pl
import pytest

from tests.conftest import frame, make_post, snowflake
from xviral.embed import mm
from xviral.embed.store import EmbeddingStore
from xviral.features import audience, content, suppression
from xviral.features.neighbors import author_centroid_and_selfsim, global_neighbors
from xviral.media import asr, video

DAY = 86_400_000
T0 = 1_735_689_600_000


def _png(color=(255, 0, 0)) -> bytes:
    from PIL import Image

    b = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(b, format="PNG")
    return b.getvalue()


# --- the X-faithful encoder inputs -------------------------------------------------


class _Capture:
    def __init__(self):
        self.calls = []

    def encode(self, inputs, prompt=None, batch_size=None):
        self.calls.append((inputs, prompt))
        return np.ones((len(inputs), 4096), dtype=np.float32)


def _conv_signature(conv):
    """Comparable form of a conversation: PIL images replaced by their bytes."""
    out = []
    for msg in conv:
        parts = []
        for c in msg["content"]:
            parts.append(("image", c["image"].tobytes()) if c["type"] == "image" else ("text", c["text"]))
        out.append((msg["role"], parts))
    return out


def test_batched_inputs_identical_to_mm_encoder(monkeypatch):
    import sys
    import types

    # mm_encoder imports sentence_transformers before consulting its model cache; a stub lets the
    # reference path run with the capture model below and no weights.
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = object
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)
    m = mm.mm()
    row = {"post_id": 1, "text": "We launched https://t.co/abc today", "author_handle": "acme", "author_display_name": "Acme"}
    imgs = [_png(), _png((0, 255, 0))]
    cases = [
        (mm.build_post(row), None),
        (mm.build_post(row, images=imgs), None),
        (mm.build_post(row, video_frames=[[_png(), _png(), _png(), _png((0, 0, 255))]]), "hello from the demo"),
    ]
    for post, transcript in cases:
        # Reference: X's own path, with the model replaced by a capture.
        ref = _Capture()
        text, images = m.render_post(post, max_images=8)
        if transcript:
            text += f"\nTranscript: {transcript}"
        m._ST_MODEL_CACHE[(m.QWEN_MODEL_ID, "cpu")] = ref
        m._encode_transformers(text, images, system_prompt=m.DEFAULT_SYSTEM_PROMPT, model_id=m.QWEN_MODEL_ID, device="cpu")
        ours = _Capture()
        enc = mm.MMEncoder(model=ours)
        out = enc.encode([mm.render(post, transcript=transcript, max_images=8)])
        (ref_in, ref_prompt), (our_in, our_prompt) = ref.calls[0], ours.calls[0]
        assert ref_prompt == our_prompt
        if isinstance(ref_in[0], str):
            assert ref_in == our_in
        else:
            assert _conv_signature(ref_in[0]) == _conv_signature(our_in[0])
        assert out.shape == (1, 1024) and np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert "https://t.co" not in mm.render(cases[0][0])[0] and "Acme @acme" in mm.render(cases[0][0])[0]


def test_embedding_store_roundtrip(tmp_path):
    st = EmbeddingStore(tmp_path / "e", dim=4)
    st.write(np.array([5, 7]), np.eye(4)[:2])
    st.write(np.array([9]), np.eye(4)[2:3])
    v, found = st.get([7, 1, 9])
    assert found.tolist() == [True, False, True]
    assert np.allclose(v[0], np.eye(4)[1]) and np.isnan(v[1]).all()
    assert st.missing([5, 6, 9]) == [6]


# --- video ---------------------------------------------------------------------------


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_video_sampling(tmp_path):
    p = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=20:size=320x568:rate=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=20", "-shortest", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(p)], check=True)
    s = video.sample(p, side=128, embed_n=4, temporal_n=16)
    assert abs(s.probe.duration_s - 20) < 0.5 and s.probe.has_audio and s.probe.height == 568
    assert s.probe.resolution_class == "sd"
    assert len(s.embed) == 4 and len(s.temporal) == 16 and s.thumb
    assert s.audio_flac and s.audio_flac[:4] == b"fLaC"
    from PIL import Image

    im = Image.open(io.BytesIO(s.embed[0]))
    assert max(im.size) == 128 and im.size[1] > im.size[0]  # vertical video stays vertical


def test_sample_times_short_and_long():
    e, t = video.sample_times(8.0, 4, 16)
    assert len(e) == 4 and t == [float(i) for i in range(8)]
    e, t = video.sample_times(120.0, 4, 16)
    assert len(t) == 16 and t[0] == 0.0 and 119 <= t[-1] <= 119.5


def test_asr_summary():
    text, ratio, first3 = asr.summarize([(0.5, 2.0, " hi ", 0.1), (5.0, 9.0, "noise", 0.9), (10.0, 12.0, "there", 0.2)], 20.0)
    assert text == "hi there" and ratio == pytest.approx(3.5 / 20) and first3


# --- content, audience, suppression --------------------------------------------------


def test_content_features():
    vid = {"type": "video", "url": "u", "video_url": "v", "width": 1080, "height": 1920, "duration_ms": 42000, "alt": None}
    short = dict(vid, duration_ms=8000)
    rows = [
        make_post(snowflake(T0, 1), 1, T0, text="Launch! https://t.co/a https://t.co/m\nnew line", media=[vid],
                  urls=[{"tco": "https://t.co/a", "expanded": "https://github.com/x/y", "final_domain": "github.com", "is_media": False},
                        {"tco": "https://t.co/m", "expanded": "https://x.com/a/status/1/video/1", "final_domain": "x.com", "is_media": True}]),
        make_post(snowflake(T0, 2), 1, T0, text="tiny clip", media=[short]),
        make_post(snowflake(T0, 3), 1, T0, text="x" * 300, kind="quote"),
    ]
    f = content.content_features(frame(rows)).sort("post_id").to_dicts()
    a, b, c = f
    assert a["has_video"] and a["video_gt_10s"] and a["video_index_eligible"] and a["video_duration_bucket"] == "10_60s"
    assert a["external_url_count"] == 1 and a["has_url"] and str(a["domain_class"]) == "code"
    assert a["newline_count"] == 1 and a["media_aspect"] == "vertical" and a["thunder_has_video"]
    # "Launch! " + space between the URLs + newline + "new line"; one external URL at 23; media t.co excluded.
    assert a["weighted_text_len"] == len("Launch! ") + 1 + 1 + len("new line") + 23
    assert not b["video_gt_10s"] and b["video_duration_bucket"] == "le_10s" and not b["video_index_eligible"]
    assert c["is_long_post"] and c["is_quote"] and not c["has_media"]


def test_audience_zone_estimation():
    rows = []
    for d in range(20):
        for h in (17, 18, 20, 22):  # 9-14 Pacific
            t = T0 + d * DAY + h * 3600_000
            rows.append(make_post(snowflake(t, d * 10 + h), 1, t))
    z = audience.estimate_audience_zone(frame(rows))
    assert z["audience_zone"][0] == "us_pacific"
    tf = audience.audience_time_features(frame(rows[:1]), z)
    assert tf["local_hour_idx"][0] == ((17 - 8) % 24) + 1


def test_pagerank_and_spam_rules():
    edges = pl.DataFrame({"src_author_id": [1, 2, 3, 4], "dst_author_id": [9, 9, 9, 1]})
    pr = suppression.pagerank(edges).sort("graph_pagerank", descending=True)
    assert pr["author_id"][0] == 9
    t = pl.DataFrame({"p": ["SpamEngagementBaiting", "SpamHashTagAbuse", "SpamHashTagAbuse", "NoViolation"], "hp": [True, True, False, False]})
    got = t.select(suppression.spam_label_would_apply(pl.col("p"), pl.col("hp")).alias("x"))["x"].to_list()
    assert got == [True, False, True, False]


def test_author_nsfw_flag_uses_prior_posts_only():
    s = pl.DataFrame({"post_id": [1, 2, 3, 4, 5], "author_id": [1] * 5, "created_at_ms": [1, 2, 3, 4, 5],
                      "p_nsfw_max": [0.99, 0.99, 0.99, 0.1, 0.1]})
    f = suppression.author_nsfw_flag(s).sort("post_id")["author_nsfw_label"].to_list()
    assert f == [False, False, False, True, True]


# --- neighbours ---------------------------------------------------------------------


def test_neighbors_are_strictly_from_earlier_days_and_other_authors():
    rng = np.random.default_rng(0)
    n = 300
    t = T0 + rng.integers(0, 20 * DAY, n)
    meta = pl.DataFrame({
        "post_id": np.arange(n), "author_id": rng.integers(0, 30, n), "created_at_ms": t,
        "log_reach_ratio": rng.normal(size=n), "rate_fav": rng.random(n), "rate_reply": rng.random(n),
        "follower_tier_idx": rng.integers(0, 6, n).astype(float),
    })
    emb = rng.normal(size=(n, 16)).astype(np.float32)
    nb = global_neighbors(meta, emb, k=5)
    # Query blocking never changes the result (cosines agree to float rounding, which depends on the block shape).
    nb2 = global_neighbors(meta, emb, k=5, max_block_cells=7)
    assert nb["nbr_post_id"].equals(nb2["nbr_post_id"]) and nb["nbr_density_7d"].equals(nb2["nbr_density_7d"])
    for a, b in zip(nb["nbr_cos"].to_list(), nb2["nbr_cos"].to_list()):
        assert (a is None and b is None) or np.allclose(a, b, atol=1e-5)
    day = dict(zip(meta["post_id"], meta["created_at_ms"] // DAY))
    auth = dict(zip(meta["post_id"], meta["author_id"]))
    checked = 0
    for r in nb.to_dicts():
        for q in r["nbr_post_id"] or []:
            assert day[q] < day[r["post_id"]] and day[q] >= day[r["post_id"]] - 7
            assert auth[q] != auth[r["post_id"]]
            checked += 1
        if r["nbr_cos"]:
            assert r["nbr_cos"] == sorted(r["nbr_cos"], reverse=True)
    assert checked > 100
    cent, ss = author_centroid_and_selfsim(meta, emb, window=50)
    order = meta.with_row_index("i").sort("author_id", "created_at_ms")
    first_rows = order.group_by("author_id", maintain_order=True).first()["i"].to_list()
    assert np.isnan(cent[first_rows]).all()


def test_neighbors_work_without_views():
    rng = np.random.default_rng(1)
    n = 200
    meta = pl.DataFrame({"post_id": np.arange(n), "author_id": rng.integers(0, 20, n), "created_at_ms": T0 + rng.integers(0, 15 * DAY, n),
                         "log_reach_ratio": pl.Series([None] * n, dtype=pl.Float64), "log_like_ratio": rng.random(n)})
    nb = global_neighbors(meta, rng.normal(size=(n, 8)).astype(np.float32), k=4, density_cos=-1.0)
    assert nb.filter(pl.col("nbr_post_id").list.len() > 0).height > 150
    assert nb["nbr_best_reach_7d"].null_count() == n and nb["nbr_best_like_7d"].null_count() < 50
