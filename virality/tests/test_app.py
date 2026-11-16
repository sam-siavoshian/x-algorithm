"""The launch advisor product: the fast request path gives the same numbers as the original, the launch planner,
version comparison, media inputs, analytics import and the HTTP layer."""

import copy
import io
import json
import logging
import math
import shutil
import subprocess

import numpy as np
import polars as pl
import pytest

from tests.test_serve import HOUR, embed, trained  # noqa: F401  (trained is a module fixture)
from xviral import serve
from xviral.model import train
from xviral.model.config import TrainCfg

V = [{"text": "some thoughts on the week ahead"},
     {"text": "launch day: we shipped the new agent today", "image": True},
     {"text": "Introducing Atlas 2: agents that file your taxes\n\nIt reads receipts.\nIt fills the forms.", "video": True},
     {"text": "We shipped a thing. It is fast. It is cheap. https://example.com/atlas"},
     {"text": "launch: Atlas 2 is out today #ai #agents"},
     {"text": "Tired of slow builds? We fixed it. Launching today, link below https://example.com/x"}]


@pytest.fixture(scope="module")
def ensemble(trained):  # noqa: F811
    """A second nano model with another seed, so the ensemble paths (members, agreement) are exercised."""
    tmp_path, posts, table, mcfg = trained
    train.train(table, tmp_path / "run2", mcfg, TrainCfg(steps=200, batch_size=32, lr=2e-3, warmup_steps=20, eval_every=200, ema_decay=0.98, seed=7), log_every=10_000)
    return [tmp_path / "run" / "best.pt", tmp_path / "run2" / "best.pt"]


def _services(trained, ensemble, **kw):  # noqa: F811
    from xviral.app import service

    tmp_path, posts, _, _ = trained
    posts.write_parquet(tmp_path / "posts.parquet")
    mk = lambda fast: service.Service(serve.DraftScorer([str(c) for c in ensemble], embed), {"likes": 6.0, "views": 6.0},  # noqa: E731
                                      posts=tmp_path / "posts.parquet", fast=fast, **kw)
    return mk(False), mk(True)


def _diff(a, b, path="", skip=("seconds", "timing_ms", "generated_at")):
    """Largest relative difference between two reports, and every non-numeric difference."""
    worst, other = (0.0, ""), []

    def walk(x, y, p):
        nonlocal worst
        if isinstance(x, dict):
            if x.keys() != y.keys():
                other.append((p, sorted(x.keys() ^ y.keys())))
                return
            for k in x:
                if k not in skip:
                    walk(x[k], y[k], f"{p}/{k}")
        elif isinstance(x, list):
            if len(x) != len(y):
                other.append((p, len(x), len(y)))
                return
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{p}[{i}]")
        elif isinstance(x, float) and isinstance(y, float):
            if not (math.isnan(x) and math.isnan(y)):
                r = abs(x - y) / max(abs(x), abs(y), 1e-12)
                if r > worst[0]:
                    worst = (r, p)
        elif x != y:
            other.append((p, x, y))
    walk(a, b, path)
    return worst, other


# --- speed without changing a number ---------------------------------------------------------------------


def test_fast_path_gives_the_same_scores(trained, ensemble):  # noqa: F811
    """The scorer's fast mode (rows the model reads embedded, drafts featurised in one pass, members run once)
    returns the same reports as the original path, for one model and for an ensemble, 1 to 6 drafts, with history
    longer and shorter than the model's window."""
    tmp_path, posts, _, mcfg = trained
    for ck in ([str(ensemble[0])], [str(c) for c in ensemble]):
        a, b = serve.DraftScorer(ck if len(ck) > 1 else ck[0], embed), serve.DraftScorer(ck if len(ck) > 1 else ck[0], embed)
        b.fast = True
        for author, cut in ((3, 40), (7, 12), (11, 1)):
            hist = posts.filter(pl.col("author_id") == author).sort("created_at_ms").head(cut)
            assert (hist.height > mcfg.max_history) == (cut == 40)
            when = int(hist["created_at_ms"].max()) + HOUR
            for n in (1, 2, 6):
                drafts = [{"text": v["text"], "media": [{"type": "photo"}] if v.get("image") else []} for v in V[:n]]
                ra, rb = a.score(hist, drafts, {"handle": f"u{author}", "followers": 9000}, when), b.score(hist, drafts, {"handle": f"u{author}", "followers": 9000}, when)
                worst, other = _diff(ra, rb)
                assert worst[0] < 1e-6 and not other, (author, n, worst, other)
        # The batched feature frame is the per-draft frame, row for row, column for column.
        hist = posts.filter(pl.col("author_id") == 5).sort("created_at_ms")
        acct = {"author_id": 5, "handle": "u5", "followers": 9000}
        when = int(hist["created_at_ms"].max()) + HOUR
        now = pl.DataFrame({"author_id": [5], "snapshot_at_ms": [when], "followers": [9000]}, schema={"author_id": pl.Int64, "snapshot_at_ms": pl.Int64, "followers": pl.Int64})
        drafts = [{"text": v["text"]} for v in V]
        assert b._feature_rows(hist, drafts, acct, when, now).equals(b._feature_rows_batched(hist, drafts, acct, when, now), null_equal=True)
        # Only the newest max_history history rows and the drafts are embedded.
        rows = b._feature_rows(hist, drafts, acct, when, now)
        use = b._rows_the_model_reads(rows, len(drafts))
        assert len(use) == mcfg.max_history + len(drafts) and use[-1] == rows.height - 1


def test_fast_service_reports_match_and_skip_the_encoder(trained, ensemble):  # noqa: F811
    """Whole reports from the product service, fast against original: same numbers, same verdicts, same text. The
    fast service embeds only what the model reads and, on a repeat request, nothing."""
    orig, fast = _services(trained, ensemble)
    calls = {"n": 0}
    raw = fast.scorer._embed_raw

    def counting(texts):
        calls["n"] += len(texts)
        return raw(texts)
    fast.scorer._embed_raw = counting
    past = [{"text": f"launch day: we shipped the new agent {i}" if i % 3 == 0 else f"some thoughts on the week {i}", "posted_at": f"2026-09-{1 + i:02d}T15:00:00Z",
             "likes": 400 if i % 3 == 0 else 60, "views": 16_000 if i % 3 == 0 else 2_400} for i in range(24)]
    # Every request names its posting time: without one a report is scored at "now", and two calls a few milliseconds
    # apart would not be the same request.
    reqs = [{"handle": "user23", "versions": V[:2], "when": "2026-02-01T15:00:00Z"}, {"handle": "user7", "versions": V, "when": "2026-01-05T15:00:00Z"},
            {"handle": "newco", "followers": 9_000, "versions": V[:3], "past_posts": past, "when": "2026-09-25T16:00:00Z"},
            {"handle": "newco", "followers": 9_000, "versions": V[:2], "past_posts": past, "when": "2026-09-24T15:30:00Z"},   # 30 minutes after the last post: second pass
            {"handle": "brandnew", "followers": 500, "versions": V[2:5], "when": "2026-03-01T15:00:00Z"}]
    for req in reqs:
        before = calls["n"]
        a, b = orig.analyze(copy.deepcopy(req)), fast.analyze(copy.deepcopy(req))
        worst, other = _diff(a, b)
        assert worst[0] < 1e-6 and not other, (req["handle"], worst, other)
        assert calls["n"] - before <= 16 + len(req["versions"]) + 2       # max_history of the nano model, the drafts, the spacing variant
    before = calls["n"]
    fast.analyze(copy.deepcopy(reqs[1]))
    assert calls["n"] == before                                              # the account's history is cached, embeddings too
    spaced = fast.analyze(copy.deepcopy(reqs[3]))
    assert any(c["operator"] == "wait_for_gap" for c in spaced["changes"])
    assert set(spaced["timing_ms"]) >= {"history", "features", "embedding", "model", "total"}


# --- comparing versions ---------------------------------------------------------------------------------


def _rep(likes, members=None):
    return {"likes": {"q50": likes}, "views_24h": {"q50": likes * 40}, "members": [{"likes_q50": m, "views_q50": m * 40} for m in (members or [likes, likes])]}


def test_compare_versions_names_a_winner_only_through_the_gates(trained, ensemble):  # noqa: F811
    _, svc = _services(trained, ensemble)
    one = svc.compare_versions([_rep(10)])
    assert one["status"] == "one_version" and one["statement"] is None
    c = svc.compare_versions([_rep(10), _rep(40), _rep(12)])             # version 2 far ahead of both
    assert c["status"] == "clear_winner" and c["winner"] == 2 and c["eliminated"] == [1, 3] and "every other version" in c["statement"]
    assert c["confidence"] == min(p["p_b_better"] if p["b"] == 2 else 1 - p["p_b_better"] for p in c["pairs"] if 2 in (p["a"], p["b"]))
    assert len(c["pairs"]) == 3
    c = svc.compare_versions([_rep(10), _rep(40)])
    assert c["winner"] == 2 and c["statement"].startswith("Version 2 is predicted to do better than version 1 (")
    c = svc.compare_versions([_rep(40), _rep(10)])
    assert c["winner"] == 1                                               # the first version can win
    c = svc.compare_versions([_rep(10), _rep(40), _rep(41)])            # two leaders, too close between them
    assert c["status"] == "too_close" and c["contenders"] == [2, 3] and c["eliminated"] == [1]
    assert c["statement"] == "Too close to call between versions 2 and 3. Version 1 is predicted to do worse than another version."
    c = svc.compare_versions([_rep(10), _rep(10.5), _rep(11)])
    assert c["status"] == "too_close" and c["contenders"] == [1, 2, 3] and c["statement"].startswith("The versions are too close to call")
    c = svc.compare_versions([_rep(10), _rep(40, members=[60, 8])])     # the models disagree: no winner
    assert c["status"] == "too_close" and c["winner"] is None


# --- the launch planner ---------------------------------------------------------------------------------


def test_launch_plan_is_built_from_measured_evidence(trained, ensemble):  # noqa: F811
    _, svc = _services(trained, ensemble)
    ev = svc.launch_evidence
    good = {"text": "Introducing Atlas 2: agents that file your taxes\n\nIt reads receipts.\nIt fills the forms.", "media": [{"type": "video"}]}
    bad = {"text": "Tired of filing taxes? We fixed it, link below https://atlas.example/launch", "media": []}
    plan = svc.launch_plan([good, bad], hours_since_previous=0.5)
    keys = [i["key"] for i in plan["items"]]
    assert set(keys) == {"names_product", "lines", "video", "announcement_first", "original_post", "no_link", "two_hours"}
    assert keys[0] == "names_product" and keys[-1] == "two_hours"                       # grade A first, grade C last
    by = {i["key"]: i for i in plan["items"]}
    for k, ek in (("names_product", "names_product_first_line"), ("lines", "lines3"), ("announcement_first", "announcement_not_problem_first"),
                  ("original_post", "original_not_quote"), ("no_link", "no_link_in_post"), ("two_hours", "two_hours_after_previous")):
        assert by[k]["grade"] == ev[ek]["grade"]
        src = ev[ek]["all_accounts"] if k == "two_hours" else ev[ek]
        assert by[k]["numbers"]["pairs"] == src["pairs"] and f"{src['pairs']:,} pairs" in by[k]["measured"]
    assert by["names_product"]["met"] == [True, False] and by["lines"]["met"] == [True, False] and by["video"]["met"] == [True, False]
    assert by["announcement_first"]["met"] == [True, False] and by["no_link"]["met"] == [True, False] and by["two_hours"]["met"] == [False, False]
    assert by["video"]["numbers"]["against_image"]["pairs"] == ev["video_vs_image"]["pairs"]
    assert "0.71x" in " ".join(by["two_hours"]["notes"]) and "0.5 hours after" in " ".join(by["two_hours"]["notes"])
    assert plan["met_by_version"] == [5, 0] and plan["checks_on_the_text"] == 5
    assert svc.launch_plan([good], None)["items"][-1]["met"] == [None]                  # no time given: not checked
    # Nothing in the plan comes from anywhere but the evidence file: drop a grade and the line disappears.
    svc.launch_evidence = {**ev, "lines3": {**ev["lines3"], "grade": None}}
    assert "lines" not in [i["key"] for i in svc.launch_plan([good], 3.0)["items"]]
    svc.launch_evidence = ev


def test_size_benchmark_and_report_sections(trained, ensemble):  # noqa: F811
    _, svc = _services(trained, ensemble)
    tiers = {t["tier"]: t for t in svc.launch_by_size["tiers"]}
    b = svc.size_benchmark(46_000, 120.0)
    t = tiers["10k-100k"]
    assert b["tier"] == "10k-100k" and b["likes_vs_usual"] == t["likes_vs_usual"] and abs(b["applied_to_your_usual"]["likes"] - 120 * t["likes_vs_usual"]) < 1e-9
    assert f"{t['posts']:,} posts from {t['accounts']} accounts" in b["statement"] and "10,000 to 100,000 followers" in b["statement"]
    assert svc.size_benchmark(500, None)["too_thin"] and "too few" in svc.size_benchmark(500, None)["statement"]
    assert svc.size_benchmark(2_000_000, None)["tier"] == "100k+"
    r = svc.analyze({"handle": "user23", "versions": [V[2], V[5]], "when": "2026-01-20T15:00:00Z"})
    assert r["is_launch"] and r["launch_plan"] and r["benchmarks"]["launch_for_size"]["tier"] == "10k-100k"
    assert r["benchmarks"]["own_usual"]["forecast_vs_usual"]["likes"][0] == pytest.approx(r["versions"][0]["likes_median"] / r["odds"]["own_median"]["likes"])
    assert r["versions"][0]["plan_checks_met"] == 5 and r["versions"][1]["plan_checks_met"] < 5
    assert r["comparison"]["pairs"] and r["pick"]["statement"] == r["comparison"]["statement"]
    assert "850 held-out pairs" in r["method"]["launch_validation"]
    plain = svc.analyze({"handle": "user23", "versions": [V[0]]})
    assert not plain["is_launch"] and plain["launch_plan"] is None and plain["pick"] is None
    assert svc.analyze({"handle": "user23", "versions": [V[0]], "launch": True})["launch_plan"] is not None
    json.dumps(r)                                                         # the JSON export is the report itself


# --- media ----------------------------------------------------------------------------------------------


def _png(w=640, h=360):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (20, 90, 200)).save(buf, "PNG")
    return buf.getvalue()


def _gif():
    from PIL import Image

    buf = io.BytesIO()
    frames = [Image.new("RGB", (120, 80), (i * 40, 0, 0)) for i in range(5)]
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:], duration=200, loop=0)
    return buf.getvalue()


@pytest.fixture(scope="module")
def mp4(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    p = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=2", "-f", "lavfi", "-i", "sine=duration=2",
                    "-shortest", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p.read_bytes()


def test_media_probe_and_analyzer_seam(mp4):
    from xviral.app import media

    im = media.probe(_png(), "shot.png")
    assert (im.kind, im.mime, im.width, im.height, im.animated) == ("image", "image/png", 640, 360, False)
    g = media.probe(_gif())
    assert g.kind == "video" and g.animated and g.duration_s == pytest.approx(1.0)
    v = media.probe(mp4, "clip.mp4")
    assert v.kind == "video" and (v.width, v.height) == (320, 240) and v.duration_s == pytest.approx(2.0, abs=0.15) and v.has_audio
    for bad, msg in ((b"", "empty"), (b"not a picture at all" * 10, "not an image")):
        with pytest.raises(media.MediaError, match=msg):
            media.probe(bad)
    with pytest.raises(media.MediaError, match="limit"):
        media.probe(_png(), max_bytes=100)
    seen = []

    def analyzer(info, data):
        seen.append((info.kind, len(data)))
        return media.MediaUnderstanding(attributes={"content": "product_screenshot", "shows_product_ui": True}, embedding=np.ones(8, np.float32))
    u = media.understand(im, _png(), analyzer)
    assert seen == [("image", len(_png()))] and u.attributes["content"] == "product_screenshot"
    assert im.public()["attributes"]["content"] == "product_screenshot" and im.public()["embedding_dim"] == 8 and "filename" not in im.public()
    boom = media.probe(_png())
    assert media.understand(boom, _png(), lambda i, d: 1 / 0) is None and boom.attributes == {}      # a failing plug-in does not fail the request
    assert media.load_analyzer("tests.test_app:_png") is _png and media.load_analyzer("") is None
    assert set(media.ATTRIBUTE_VOCABULARY["content"]) >= {"product_screenshot", "product_demo", "talking_head"}


def test_uploaded_video_length_reaches_x_rules(trained, ensemble):  # noqa: F811
    """X's code admits a video to the out-of-network index only above 10 seconds. A probed file carries its length to
    that rule; a ticked box carries none, and the report says the length is unknown instead of calling it a failure."""
    _, svc = _services(trained, ensemble)
    short = svc.analyze({"handle": "user23", "versions": [{"text": "launch day: demo", "media_info": {"kind": "video", "duration_s": 2.0, "width": 320, "height": 240}}]})
    g = next(r for r in short["rules"] if r["gate"] == "video_index")
    assert g["status"] == "fail" and g["duration_ms"] == 2000
    long_ = svc.analyze({"handle": "user23", "versions": [{"text": "launch day: demo", "media_info": {"kind": "video", "duration_s": 34.0}}]})
    assert not any(r["gate"] == "video_index" for r in long_["rules"])             # passes: not listed among warnings
    ticked = svc.analyze({"handle": "user23", "versions": [{"text": "launch day: demo", "video": True}]})
    g = next(r for r in ticked["rules"] if r["gate"] == "video_index")
    assert g["status"] == "warn" and "length is not known" in g["detail"]


# --- accounts we do not know ------------------------------------------------------------------------------


def _analytics_csv(n=30, start_ms=1_788_000_000_000):
    rows = ["Post id,Date,Post text,Post Link,Impressions,Likes,Engagements,Bookmarks,Shares,New follows,Replies,Reposts,Profile visits,Detail Expands,URL Clicks"]
    for i in range(n):
        ms = start_ms + i * 30 * HOUR
        pid = (ms - 1288834974657) << 22
        text = f"launch day: we shipped the new agent {i}" if i % 3 == 0 else f"some thoughts on the week {i}"
        if i == 5:
            text = "@someone thanks!"                                       # a reply: skipped
        rows.append(f'{pid},2026-09-01,"{text}",https://x.com/newco/status/{pid},{16000 if i % 3 == 0 else 2400},{400 if i % 3 == 0 else 60},10,1,1,0,2,3,4,5,6')
    rows.append(f'{((start_ms - 1288834974657) << 22) + 7},2026-09-01,,https://x.com/newco/status/1,100,1,1,0,0,0,0,0,0,0,0')
    return "\n".join(rows).encode()


def test_analytics_import(trained, ensemble):  # noqa: F811
    from xviral.app.service import Service

    out = Service.import_analytics(_analytics_csv())
    s = out["summary"]
    assert s["rows"] == 31 and s["posts"] == 29 and s["skipped_replies"] == 1 and s["skipped_without_text"] == 1 and s["handle"] == "newco"
    assert out["past_posts"][0]["posted_at"] < out["past_posts"][-1]["posted_at"] and out["past_posts"][0]["views"] == 16000
    assert Service.import_analytics(_analytics_csv(), limit=10)["summary"]["skipped_older"] == 19
    with pytest.raises(ValueError, match="analytics export"):
        Service.import_analytics(b"foo,bar\n1,2\n")
    _, svc = _services(trained, ensemble)
    r = svc.analyze({"handle": "newco", "followers": 9_000, "versions": V[:2], "past_posts": out["past_posts"]})
    assert r["account"]["history_posts"] == 29 and r["account"]["history_source"] == "posts you supplied" and not r["account"]["new_account"]


# --- HTTP ---------------------------------------------------------------------------------------------------


def test_http_validation_media_logging_and_limits(trained, ensemble, mp4, caplog):  # noqa: F811
    from fastapi.testclient import TestClient

    from xviral.app import accounts, server

    _, svc = _services(trained, ensemble)
    client = TestClient(server.create_app(svc, settings=server.Settings(rate_per_minute=600, rate_burst=100)))
    assert client.get("/api/health").json()["loaded"] and client.get("/api/ready").status_code == 200
    page = client.get("/").text
    assert "Launch advisor" in page and 'name="viewport"' in page and "prefers-color-scheme: dark" in page

    def err(body):
        r = client.post("/api/analyze", json=body)
        assert r.status_code == 422, r.text
        return r.json()["detail"]
    assert "at most six versions" in err({"handle": "user23", "versions": [{"text": "a"}] * 7})
    assert "at least one version" in err({"handle": "user23", "versions": [{"text": "  "}]})
    assert "15 letters" in err({"handle": "not a handle!", "versions": [{"text": "a"}]})
    assert "version 1, text: is too long" in err({"handle": "user23", "versions": [{"text": "a" * 5000}]})
    assert "date and time" in err({"handle": "user23", "versions": [{"text": "a"}], "when": "next tuesday"})
    assert "not a field" in err({"handle": "user23", "versions": [{"text": "a", "colour": "red"}]})
    assert "follower count is required" in err({"handle": "nobody", "versions": [{"text": "a"}]})
    assert client.post("/api/analyze", content=b"{not json", headers={"content-type": "application/json"}).status_code == 422

    # Media: files go with the request, are probed, and only their properties come back. Draft text is not logged.
    caplog.set_level(logging.INFO, logger="xviral.app")
    secret = "Introducing Atlas 2: a secret draft"
    body = {"handle": "user23", "versions": [{"text": secret}, {"text": secret + "\n\nnow with video"}, {"text": "plain one"}]}
    res = client.post("/api/analyze", data={"request": json.dumps(body)}, files={"media_1": ("shot.png", _png(), "image/png"), "media_2": ("clip.mp4", mp4, "video/mp4")})
    assert res.status_code == 200, res.text
    r = res.json()
    m1, m2 = r["versions"][0]["media"], r["versions"][1]["media"]
    assert (m1["kind"], m1["width"], m1["height"]) == ("image", 640, 360) and m2["kind"] == "video" and m2["duration_s"] == pytest.approx(2.0, abs=0.15)
    assert r["versions"][2]["media"] is None and "filename" not in m1
    assert {i["key"]: i for i in r["versions"][1]["checklist"]}["video"]["ok"]                 # the uploaded video counts as a video
    logs = [json.loads(rec.getMessage()) for rec in caplog.records if rec.name == "xviral.app" and rec.getMessage().startswith("{")]
    entry = next(e for e in logs if e["path"] == "/api/analyze" and e["status"] == 200)
    assert entry["versions"] == 3 and entry["media"] == ["image", "video", None] and entry["chars"][0] == len(secret)
    assert all(secret not in rec.getMessage() and "plain one" not in rec.getMessage() for rec in caplog.records)
    bad = client.post("/api/analyze", data={"request": json.dumps(body)}, files={"media_1": ("x.png", b"garbage" * 20, "image/png")})
    assert bad.status_code == 422 and bad.json()["detail"].startswith("version 1: ")
    assert client.post("/api/analyze", data={"request": json.dumps(body)}, files={"media_5": ("shot.png", _png(), "image/png")}).status_code == 422

    # Analytics import over HTTP, then the account lookup.
    imp = client.post("/api/import/analytics", files={"file": ("export.csv", _analytics_csv(), "text/csv")})
    assert imp.status_code == 200 and imp.json()["summary"]["posts"] == 29
    assert client.post("/api/import/analytics", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")}).status_code == 422
    a = client.get("/api/account", params={"handle": "@User23"}).json()
    assert a["known"] and a["history_posts"] == 40 and a["followers"] == 46_000
    assert client.get("/api/account", params={"handle": "nobody"}).json()["known"] is False

    # Rate limit: the burst, then 429 with Retry-After.
    tight = TestClient(server.create_app(svc, settings=server.Settings(rate_per_minute=2, rate_burst=2)))
    ok = {"handle": "user23", "versions": [{"text": "a"}]}
    assert [tight.post("/api/analyze", json=ok).status_code for _ in range(3)] == [200, 200, 429]
    r429 = tight.post("/api/analyze", json=ok)
    assert int(r429.headers["retry-after"]) >= 1 and "too many requests" in r429.json()["detail"]
    lim = server.RateLimiter(60, 1)
    assert lim.take("a", 0.0) == 0 and lim.take("a", 0.5) > 0 and lim.take("a", 1.01) == 0

    # The accounts seam: with keys set, the API needs one; the page and health checks do not.
    keyed = TestClient(server.create_app(svc, settings=server.Settings(), gate=accounts.Gate(["k1"])))
    assert keyed.post("/api/analyze", json=ok).status_code == 401 and keyed.get("/").status_code == 200 and keyed.get("/api/health").status_code == 200
    assert keyed.post("/api/analyze", json=ok, headers={"Authorization": "Bearer k1"}).status_code == 200
    used = []

    class Billing(accounts.Gate):
        def record_usage(self, caller, usage):
            used.append((caller.id, usage.endpoint, usage.versions))
    TestClient(server.create_app(svc, settings=server.Settings(), gate=Billing())).post("/api/analyze", json=ok)
    assert used == [("ip:testclient", "analyze", 1)]

    # An unexpected failure: JSON with the request ID, not a bare 500 page.
    class Broken:
        def analyze(self, req):
            raise RuntimeError("boom")
    crash = TestClient(server.create_app(Broken(), settings=server.Settings()), raise_server_exceptions=False).post("/api/analyze", json=ok)
    assert crash.status_code == 500 and crash.headers["x-request-id"] in crash.json()["detail"] and "boom" not in crash.text

    # Unconfigured server: a clear 503, health still answers.
    bare = TestClient(server.create_app(settings=server.Settings()))
    assert bare.get("/api/health").json()["loaded"] is False and bare.get("/api/ready").status_code == 503
    r = bare.post("/api/analyze", json=ok)
    assert r.status_code == 503 and "XVIRAL_CHECKPOINTS" in r.json()["detail"]


def test_settings_from_env(monkeypatch):
    from xviral.app import server

    monkeypatch.setenv("XVIRAL_CHECKPOINTS", "a.pt, b.pt")
    monkeypatch.setenv("XVIRAL_RATE_PER_MIN", "5")
    monkeypatch.setenv("XVIRAL_LOG_TEXT", "1")
    s = server.Settings.from_env()
    assert s.checkpoints == ["a.pt", "b.pt"] and s.rate_per_minute == 5 and s.log_text and s.encoder == "BAAI/bge-small-en-v1.5" and s.max_upload_mb == 64
