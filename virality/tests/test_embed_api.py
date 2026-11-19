"""The API text embedding in the product: the encoder is read from the checkpoints, vectors are cached per account,
a request reports what was sent and what it cost, and a failing provider never yields a report built on missing vectors.
The embedding provider is always the built-in stub or a fake HTTP session: nothing leaves the process."""

import base64
import copy
import json
import logging

import numpy as np
import polars as pl
import pytest
import torch

from tests.test_app import V, ensemble  # noqa: F401  (module fixtures)
from tests.test_serve import embed, trained  # noqa: F401
from xviral import serve
from xviral.embed import api

STUB16 = "openai:stub-embedding@16"


@pytest.fixture(scope="module")
def api_checkpoints(trained, ensemble):  # noqa: F811
    """The two nano checkpoints re-saved with an API encoder named in their metadata, as a table built by
    scripts/combine/build_table.py records it. (They were trained on the stand-in vectors: the plumbing is under test.)"""
    api.ENCODERS[STUB16] = {"name": STUB16, "provider": "openai", "model": "stub-embedding", "dimensions": 16, "max_chars": 8000, "price_per_mtok": 0.13,
                            "parts": [{"src": "stub", "dim": 16}]}
    tmp_path = trained[0]
    out = []
    for i, c in enumerate(ensemble):
        ck = torch.load(c, map_location="cpu", weights_only=False)
        ck["table_meta"] = {**ck["table_meta"], "encoder": dict(api.ENCODERS[STUB16])}
        p = tmp_path / f"api_{i}.pt"
        torch.save(ck, p)
        out.append(str(p))
    return out


def _service(trained, cks, enc, **kw):  # noqa: F811
    from xviral.app import service

    tmp_path, posts, _, _ = trained
    posts.write_parquet(tmp_path / "posts.parquet")
    return service.Service(serve.DraftScorer(cks, encoder=enc), {"likes": 6.0, "views": 6.0}, posts=tmp_path / "posts.parquet", **kw)


def _local(trained, ensemble):  # noqa: F811
    from xviral.app import service

    tmp_path, posts, _, _ = trained
    return service.Service(serve.DraftScorer([str(c) for c in ensemble], embed), {"likes": 6.0, "views": 6.0}, posts=tmp_path / "posts.parquet", model_set="local")


# --- the client ---------------------------------------------------------------------------------------------


def test_stub_is_deterministic_unit_length_and_counts_cost():
    s = api.StubEmbedder(64, batch=4)
    texts = [f"launch day number {i}" for i in range(10)]
    a, b = s.encode(texts), api.StubEmbedder(64).encode(list(reversed(texts)))[::-1]
    assert a.shape == (10, 64) and np.allclose(a, b, atol=1e-3) and np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-3)
    assert len(s.calls) == 3 and sorted(len(c["input"]) for c in s.calls) == [2, 4, 4]          # batches of 4, in parallel
    assert all(c["model"] == "stub-embedding" and c["dimensions"] == 64 and c["encoding_format"] == "base64" for c in s.calls)
    u = s.take_usage()
    assert u["calls"] == 3 and u["texts"] == 10 and u["tokens"] > 0 and u["usd"] == pytest.approx(u["tokens"] * 0.13 / 1e6) and u["error"] is None
    assert s.take_usage()["calls"] == 0 and s.usage.total["calls"] == 3                               # per-request usage resets, the total stays
    near, far = s.encode(["we shipped the new agent today", "we shipped the new agent today!", "quarterly tax filing deadline"]), None
    assert float(near[0] @ near[1]) > float(near[0] @ near[2])                                        # similar texts, similar vectors
    assert s.encode([]).shape == (0, 64)


class _Session:
    """A fake `requests`: a scripted list of (status, body) or exceptions, one per call."""

    def __init__(self, script):
        self.script, self.sent = list(script), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.sent.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step

        class R:
            status_code, text = step[0], "x"

            def json(self_inner):
                return step[1]
        return R()


def _ok(vectors, tokens=7, as_list=False):
    data = [{"index": i, "embedding": [float(x) for x in v] if as_list else base64.b64encode(np.asarray(v, "<f4").tobytes()).decode()} for i, v in enumerate(vectors)]
    return 200, {"data": list(reversed(data)), "usage": {"total_tokens": tokens}}                       # out of order on purpose


def test_requests_retries_errors_and_cost(monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda s: None)
    v = np.eye(4, dtype=np.float32)[:2] * 3.0
    sess = _Session([(503, {"error": {"message": "busy"}}), _ok(v, tokens=1000)])
    e = api.OpenAIEmbedder("text-embedding-3-large", 4, session=sess, retries=2, max_chars=5, api_key="k")
    out = e.encode(["abcdefghij", "b"])
    assert np.allclose(out, np.eye(4)[:2], atol=1e-3)                                                # index order restored, unit length
    body = sess.sent[0]["json"]
    assert sess.sent[0]["url"].endswith("/v1/embeddings") and sess.sent[0]["headers"]["Authorization"] == "Bearer k" and sess.sent[0]["timeout"] == 20.0
    assert body == {"model": "text-embedding-3-large", "input": ["abcde", "b"], "encoding_format": "base64", "dimensions": 4}     # cut at max_chars
    u = e.take_usage()
    assert u["calls"] == 1 and u["tokens"] == 1000 and u["usd"] == pytest.approx(0.00013) and len(sess.sent) == 2
    assert np.allclose(api.OpenAIEmbedder("m", 4, session=_Session([_ok(v, as_list=True)])).encode(["a", "b"]), np.eye(4)[:2], atol=1e-3)   # plain lists too
    # a refused request is not retried; a timeout is; both end in one clear error and no vectors
    refused = _Session([(401, {"error": {"message": "bad key"}}), _ok(v)])
    with pytest.raises(api.EmbeddingError, match="HTTP 401: bad key"):
        api.OpenAIEmbedder("m", 4, session=refused, retries=2).encode(["a", "b"])
    assert len(refused.sent) == 1
    flaky = _Session([TimeoutError("slow"), TimeoutError("slow"), TimeoutError("slow")])
    e2 = api.OpenAIEmbedder("m", 4, session=flaky, retries=2)
    with pytest.raises(api.EmbeddingError, match="did not return vectors: TimeoutError"):
        e2.encode(["a", "b"])
    assert len(flaky.sent) == 3 and e2.take_usage()["error"] == "TimeoutError"
    with pytest.raises(api.EmbeddingError, match="shape"):                                           # fewer vectors than texts
        api.OpenAIEmbedder("m", 4, session=_Session([_ok(v[:1])]), retries=0).encode(["a", "b"])
    # two batches, one fails: the whole call fails (no partial result to score with)
    half = api.StubEmbedder(8, batch=2, fail={1})
    with pytest.raises(api.EmbeddingError):
        half.encode(["a", "b", "c", "d"])


def test_encoder_is_read_from_checkpoint_metadata():
    assert api.encoder_spec({"emb_dim": 384})["name"] == api.BGE_SMALL and api.encoder_spec({})["provider"] == "local"       # every older checkpoint
    recipe = api.encoder_spec({"emb_dim": 1024, "encoder": {"parts": [{"src": "te3l", "dim": 1024}]}})                      # the embedding track's form
    full = api.encoder_spec({"emb_dim": 1024, "encoder": dict(api.ENCODERS["openai:text-embedding-3-large@1024"])})
    assert recipe == full and full["provider"] == "openai" and full["model"] == "text-embedding-3-large" and full["dimensions"] == 1024 and full["price_per_mtok"] == 0.13
    assert api.encoder_spec({"emb_dim": 1024, "encoder": "openai:text-embedding-3-small@1024"})["model"] == "text-embedding-3-small"
    with pytest.raises(ValueError, match="does not match"):
        api.encoder_spec({"emb_dim": 384, "encoder": {"parts": [{"src": "te3l", "dim": 1024}]}})
    with pytest.raises(ValueError, match="composite"):
        api.encoder_spec({"emb_dim": 1408, "encoder": {"parts": [{"src": "bge"}, {"src": "te3l", "dim": 1024}]}})
    with pytest.raises(ValueError, match="does not know"):
        api.encoder_spec({"encoder": "some-other-encoder"})
    with pytest.raises(ValueError, match="different encoders"):
        api.common_encoder([{"emb_dim": 384}, {"emb_dim": 1024, "encoder": {"parts": [{"src": "te3l", "dim": 1024}]}}])
    assert isinstance(api.make_encoder(full, env={"XVIRAL_EMBED_STUB": "1"}), api.StubEmbedder)
    real = api.make_encoder(full, env={"XVIRAL_EMBED_TIMEOUT": "3", "XVIRAL_EMBED_BATCH": "32"})
    assert type(real) is api.OpenAIEmbedder and real.timeout == 3.0 and real.batch == 32 and real.dim == 1024 and real.describe()["provider"] == "OpenAI"


# --- the scorer ---------------------------------------------------------------------------------------------


def test_scorer_takes_its_encoder_from_the_checkpoints_and_caches_per_account(trained, api_checkpoints, monkeypatch):  # noqa: F811
    tmp_path, posts, _, mcfg = trained
    monkeypatch.setenv("XVIRAL_EMBED_STUB", "1")
    sc = serve.DraftScorer(api_checkpoints)                               # no encoder given: built from the name in the checkpoints
    assert isinstance(sc.encoder, api.StubEmbedder) and sc.encoder_spec["name"] == STUB16 and sc.encoder.dim == 16
    with pytest.raises(ValueError, match="trained with 16"):              # a model is never served with another encoder's vectors
        serve.DraftScorer(api_checkpoints, encoder=api.StubEmbedder(1024))
    sc.fast = True
    hist = posts.filter(pl.col("author_id") == 3).sort("created_at_ms")
    when = int(hist["created_at_ms"].max()) + 3_600_000
    acct = {"handle": "u3", "followers": 9000}
    r1 = sc.score(hist, [{"text": V[0]["text"]}, {"text": V[1]["text"]}], acct, when)
    first = sum(len(c["input"]) for c in sc.encoder.calls)
    assert first == mcfg.max_history + 2 and len(sc.encoder.calls) == 1                 # the history the model reads and the drafts, one call
    assert all(t.splitlines()[0].endswith("@u3") or "u3" in t for c in sc.encoder.calls for t in c["input"][-2:])   # drafts carry the author line
    r2 = sc.score(hist, [{"text": V[0]["text"]}, {"text": V[2]["text"]}], acct, when)
    assert sum(len(c["input"]) for c in sc.encoder.calls) == first + 1                 # repeat request: only the new draft is sent
    assert r1[0]["likes"]["q50"] == pytest.approx(r2[0]["likes"]["q50"], rel=1e-6)
    assert sc.embed_counts == {"texts": 2 * (mcfg.max_history + 2), "sent": mcfg.max_history + 3}
    # least recently used account is dropped when more accounts are held than the limit
    sc.emb_accounts_limit = 2
    for a in (5, 7):
        h = posts.filter(pl.col("author_id") == a).sort("created_at_ms")
        sc.score(h, [{"text": V[0]["text"]}], {"handle": f"u{a}", "followers": 9000}, int(h["created_at_ms"].max()) + 3_600_000)
    assert list(sc._emb_scopes) == ["u5", "u7"] and not any("u3" in t for t in sc._emb_cache)
    # a failed call caches nothing and raises the embedding error
    n = len(sc._emb_cache)
    sc.encoder.fail = True
    with pytest.raises(api.EmbeddingError):
        sc.score(hist, [{"text": "a brand new draft"}], acct, when)
    assert len(sc._emb_cache) == n


# --- the service --------------------------------------------------------------------------------------------


def test_report_states_cost_and_what_was_sent_first_and_repeat(trained, api_checkpoints):  # noqa: F811
    stub = api.StubEmbedder(16)
    svc = _service(trained, api_checkpoints, stub, model_set="api")
    req = {"handle": "user23", "versions": V[:2], "when": "2026-02-01T15:00:00Z"}
    a = svc.analyze(copy.deepcopy(req))
    e = a["embedding"]
    assert a["model_set"] == "api" and e["encoder"] == STUB16 and e["external"] and e["model"] == "stub-embedding" and e["dimensions"] == 16
    assert e["texts_sent"] > 2 and e["texts_from_cache"] >= 0 and e["tokens"] > 0 and e["cost_usd"] == pytest.approx(e["tokens"] * 0.13 / 1e6, abs=1e-6)
    assert e["sent"]["to"] == stub.provider and "author name and handle" in e["sent"]["what"] and not e["fallback_used"] and e["error"] is None
    assert a["cost_usd"] == {"embedding": e["cost_usd"], "judge": 0.0, "total": e["cost_usd"]}
    b = svc.analyze({**copy.deepcopy(req), "versions": [V[0], V[3]]})                  # same account again, one new version
    f = b["embedding"]
    assert f["texts_sent"] <= 3 and f["texts_from_cache"] >= e["texts_sent"] - 2 and f["cost_usd"] < e["cost_usd"] and f["calls"] >= 1
    c = svc.analyze(copy.deepcopy(req))                                                # identical request: nothing is sent, nothing is billed
    assert c["embedding"]["texts_sent"] == 0 and c["embedding"]["cost_usd"] == 0.0 and c["embedding"]["sent"] is None and c["embedding"]["calls"] == 0
    assert c["versions"][0]["likes_median"] == pytest.approx(a["versions"][0]["likes_median"], rel=1e-6)
    sent_texts = [t for call in stub.calls for t in call["input"]]
    assert not any(isinstance(t, (int, float)) for t in sent_texts) and not any("9000" in t for t in sent_texts)      # texts only: no counts, no follower numbers


def test_provider_failure_falls_back_or_fails_clearly(trained, ensemble, api_checkpoints):  # noqa: F811
    from xviral.app import service

    req = {"handle": "user23", "versions": V[:2], "when": "2026-02-01T15:00:00Z"}
    local = _local(trained, ensemble)
    plain = local.analyze(copy.deepcopy(req))
    assert not plain["embedding"]["external"] and plain["embedding"]["cost_usd"] == 0.0 and plain["embedding"]["sent"] is None and plain["embedding"]["provider"] == "local"
    # no fallback configured: a clear error, never a report
    alone = _service(trained, api_checkpoints, api.StubEmbedder(16, fail=True))
    with pytest.raises(service.EmbeddingUnavailable, match="did not return vectors.*Nothing was scored"):
        alone.analyze(copy.deepcopy(req))
    # with a fallback: its report, marked, with the error and the fact that the texts had been sent
    both = _service(trained, api_checkpoints, api.StubEmbedder(16, fail=True), fallback=local)
    out = both.analyze(copy.deepcopy(req))
    e = out["embedding"]
    assert e["fallback_used"] and e["fallback_reason"] == "the embedding provider failed" and e["primary_encoder"] == STUB16 and "stub failure" in e["error"]
    assert not e["external"] and e["cost_usd"] == 0.0 and e["primary_attempt"]["calls"] == 1 and out["model_set"] == "local"
    assert out["versions"][0]["likes_median"] == pytest.approx(plain["versions"][0]["likes_median"], rel=1e-6)          # exactly the fallback models' report
    assert out["comparison"]["status"] == plain["comparison"]["status"]
    # the provider recovers: the next request is the primary set's again
    both.scorer.encoder.fail = False
    again = both.analyze(copy.deepcopy(req))
    assert again["embedding"]["external"] and not again["embedding"]["fallback_used"] and again["embedding"]["texts_sent"] > 0


def test_text_is_not_sent_without_permission(trained, ensemble, api_checkpoints):  # noqa: F811
    from xviral.app import service

    past = [{"text": f"launch day: we shipped the new agent {i}" if i % 3 == 0 else f"some thoughts on the week {i}", "posted_at": f"2026-09-{1 + i:02d}T15:00:00Z",
             "likes": 400 if i % 3 == 0 else 60} for i in range(24)]
    up = {"handle": "newco", "followers": 9000, "versions": V[:2], "past_posts": past, "when": "2026-09-25T16:00:00Z"}
    stub = api.StubEmbedder(16)
    svc = _service(trained, api_checkpoints, stub, fallback=_local(trained, ensemble))
    out = svc.analyze(copy.deepcopy(up))                                    # uploaded posts, no permission: the local models answer
    assert out["embedding"]["fallback_used"] and "share_past_posts" in out["embedding"]["fallback_reason"] and not stub.calls and out["model_set"] == "local"
    out = svc.analyze({**copy.deepcopy(up), "share_past_posts": True})      # permission given: the API model set answers
    assert not out["embedding"]["fallback_used"] and out["embedding"]["texts_sent"] > 10 and stub.calls
    n = len(stub.calls)
    out = svc.analyze({"handle": "user23", "versions": V[:2], "when": "2026-02-01T15:00:00Z", "external_embedding": False})     # opt-out for a corpus account
    assert out["embedding"]["fallback_used"] and "external_embedding" in out["embedding"]["fallback_reason"] and len(stub.calls) == n
    bare = _service(trained, api_checkpoints, api.StubEmbedder(16))         # no local models: refused with the reason and the remedy
    with pytest.raises(service.EmbeddingNotAllowed, match="share_past_posts: true"):
        bare.analyze(copy.deepcopy(up))
    assert not bare.scorer.encoder.calls
    local = _local(trained, ensemble)                                       # a local encoder needs no permission
    assert not local.analyze(copy.deepcopy(up))["embedding"]["fallback_used"]


def test_warming_an_account_is_best_effort_and_scoped(trained, api_checkpoints):  # noqa: F811
    stub = api.StubEmbedder(16)
    svc = _service(trained, api_checkpoints, stub)
    assert svc.account("user23")["known"] and stub.calls and "user23" in svc.scorer._emb_scopes
    n = sum(len(c["input"]) for c in stub.calls)
    out = svc.analyze({"handle": "user23", "versions": V[:1]})
    assert out["embedding"]["texts_sent"] <= 2 and sum(len(c["input"]) for c in stub.calls) <= n + 2          # the history was already embedded
    broken = _service(trained, api_checkpoints, api.StubEmbedder(16, fail=True))
    assert broken.account("user7")["known"]                                 # a failing provider does not fail the lookup


def test_bundle_loads_a_model_set_from_one_config(trained, ensemble, api_checkpoints, tmp_path):  # noqa: F811
    from xviral.app import service

    rule = {"version": 1, "speak": 0.6, "default_ensemble": "api2", "ensembles": {"api2": {"kinds": {k: {"model": {"type": "logistic_gap", "scale": s}, "fallback": {"type": "logistic_gap", "scale": s}}
                                                                                                    for k, s in (("same_topic", 3.0), ("launch", 2.0))}}}}
    (tmp_path / "selective_rule.json").write_text(json.dumps(rule))
    (tmp_path / "type_correction.json").write_text(json.dumps({"weights_log10": {"pt_launch": 0.1}}))
    cfg = {"name": "api-set", "checkpoints": api_checkpoints, "encoder": {"name": STUB16}, "scales": {"likes": 3.0}, "selective_rule": "selective_rule.json",
           "selective_ensemble": "api2", "type_correction": "type_correction.json", "validation": {"statement": "Measured for this set.", "launch_statement": "Launch for this set."},
           "fallback": {"name": "local-set", "checkpoints": [str(c) for c in ensemble], "scales": {"likes": 6.0, "views": 6.0}}}
    (tmp_path / "bundle.json").write_text(json.dumps(cfg))
    svc = service.load_bundle(tmp_path, posts=str(trained[0] / "posts.parquet"), embedder=api.StubEmbedder(16), fallback_embedder=_Callable(embed))
    assert svc.model_set == "api-set" and svc.fallback is not None and svc.fallback.model_set == "local-set" and svc.embedding_info()["encoder"] == STUB16
    assert svc.advisor.selective.rule["ensemble"] == "api2" and svc.scorer.type_correction == {"pt_launch": 0.1} and svc.advisor.scales == {"likes": 3.0}
    out = svc.analyze({"handle": "user23", "versions": V[1:3], "when": "2026-02-01T15:00:00Z"})
    assert out["method"]["validation"] == "Measured for this set." and out["method"]["launch_validation"] == "Launch for this set." and out["embedding"]["texts_sent"] > 0
    with pytest.raises(ValueError, match="XVIRAL_ENCODER"):                 # naming another encoder than the checkpoints' is refused
        service.load(api_checkpoints, {"likes": 3.0}, encoder="BAAI/bge-small-en-v1.5")


class _Callable:
    """A local encoder object around a function (the test fixture's stand-in vectors)."""
    local, provider, dim = True, "local", 16

    def __init__(self, fn):
        self.fn = fn

    def encode(self, texts, batch_size=None):
        return self.fn(texts)

    def describe(self):
        return {"provider": "local", "model": "test", "dimensions": 16, "endpoint": None}

    def take_usage(self):
        return {"calls": 0, "texts": 0, "tokens": 0, "usd": 0.0, "seconds": 0.0, "error": None}


def test_http_reports_embedding_and_answers_503_when_the_provider_is_down(trained, ensemble, api_checkpoints, caplog):  # noqa: F811
    from fastapi.testclient import TestClient

    from xviral.app import server

    ok = {"handle": "user23", "versions": [{"text": V[0]["text"]}, {"text": V[1]["text"], "image": True}], "when": "2026-02-01T15:00:00Z"}
    svc = _service(trained, api_checkpoints, api.StubEmbedder(16), fallback=_local(trained, ensemble), model_set="api")
    client = TestClient(server.create_app(svc, settings=server.Settings(rate_per_minute=600, rate_burst=100)))
    with caplog.at_level(logging.INFO, logger="xviral.app"):
        r = client.post("/api/analyze", json=ok)
    assert r.status_code == 200 and r.json()["embedding"]["texts_sent"] > 0 and r.json()["cost_usd"]["total"] > 0
    line = next(json.loads(m.message) for m in caplog.records if m.message.startswith("{") and '"path": "/api/analyze"' in m.message)
    assert line["embedding"]["sent"] is True and line["embedding"]["cost_usd"] > 0 and not line["embedding"]["fallback_used"] and V[0]["text"] not in json.dumps(line)
    r = client.post("/api/analyze", json={**ok, "external_embedding": False})
    assert r.status_code == 200 and r.json()["embedding"]["fallback_used"] and r.json()["model_set"] == "local"
    h = client.get("/api/health").json()
    assert h["model_set"] == "api" and h["embedding"]["external"] and h["fallback_model_set"] is True
    schema = client.get("/openapi.json").json()
    assert "external_embedding" in json.dumps(schema)
    down = TestClient(server.create_app(_service(trained, api_checkpoints, api.StubEmbedder(16, fail=True)), settings=server.Settings(rate_per_minute=600, rate_burst=100)))
    r = down.post("/api/analyze", json=ok)
    assert r.status_code == 503 and "did not return vectors" in r.json()["detail"] and "Nothing was scored" in r.json()["detail"] and r.headers["retry-after"] == "5"
    bare = TestClient(server.create_app(_service(trained, api_checkpoints, api.StubEmbedder(16)), settings=server.Settings(rate_per_minute=600, rate_burst=100)))
    past = [{"text": f"some thoughts on the week {i}", "posted_at": f"2026-09-{1 + i:02d}T15:00:00Z", "likes": 60} for i in range(12)]
    r = bare.post("/api/analyze", json={"handle": "newco", "followers": 900, "versions": [{"text": "x y z"}], "past_posts": past})
    assert r.status_code == 422 and "share_past_posts" in r.json()["detail"]
    monkey = server.Settings.from_env()
    assert monkey.bundle is None and monkey.fallback_checkpoints == []


def test_other_members_decide_launch_comparisons_from_the_same_vectors(trained, api_checkpoints):  # noqa: F811
    """A model set may name other members for one kind of comparison (launch): they are scored on the same features and
    vectors, decide comparisons of that kind with that kind's scale, and leave the forecast and everything else alone."""
    import math

    from xviral.app import service
    from xviral.selective import SelectiveRule

    rule = SelectiveRule({"version": 1, "speak": 0.6, "default_ensemble": "x", "ensembles": {"x": {"kinds": {k: {"model": {"type": "logistic_gap", "scale": sc}, "fallback": {"type": "logistic_gap", "scale": sc}}
                                                                                                     for k, sc in (("same_topic", 3.0), ("launch", 2.0))}}}})
    tmp_path, posts, _, _ = trained
    posts.write_parquet(tmp_path / "posts.parquet")
    plain_stub, stub = api.StubEmbedder(16), api.StubEmbedder(16)
    plain = service.Service(serve.DraftScorer(api_checkpoints, encoder=plain_stub), {"likes": 6.0}, posts=tmp_path / "posts.parquet", selective=rule)
    sc = serve.DraftScorer(api_checkpoints, encoder=stub)
    sc.kind_scorers["launch"] = serve.DraftScorer(api_checkpoints[1], encoder=stub)          # one of the two members alone, as the launch members
    svc = service.Service(sc, {"likes": 6.0}, posts=tmp_path / "posts.parquet", selective=rule)
    launch2 = [{"text": "Introducing Atlas 2: agents that file your taxes\n\nIt reads receipts.", "video": True}, {"text": "Introducing Atlas 2, our new agent for taxes. Launching today."}]
    req = {"handle": "user23", "versions": launch2, "when": "2026-02-01T15:00:00Z"}
    a, b = plain.analyze(copy.deepcopy(req)), svc.analyze(copy.deepcopy(req))
    assert [len(c["input"]) for c in stub.calls] == [len(c["input"]) for c in plain_stub.calls]          # the launch members embed nothing of their own
    assert b["embedding"]["cost_usd"] == a["embedding"]["cost_usd"] and "model_launch" in b["timing_ms"]
    assert [v["likes_median"] for v in b["versions"]] == pytest.approx([v["likes_median"] for v in a["versions"]], rel=1e-6)   # the forecast is the main members'
    pa, pb = a["comparison"]["pairs"][0], b["comparison"]["pairs"][0]
    assert pa["decided_by_members"] == "forecast" and pb["decided_by_members"] == "launch" and pb["predicted_ratio_b_over_a"] == pytest.approx(pa["predicted_ratio_b_over_a"], rel=1e-6)
    # the probability is the launch scale on the launch members' gap
    hist = svc._corpus_account("user23")["history"]
    reps = sc.score(hist, [svc._draft(v) for v in launch2], {"handle": "user23", "followers": int(svc._corpus_account("user23")["followers"])}, service._ms(req["when"]))
    assert all(len(r["kind_reports"]["launch"]["members"]) == 1 and len(r["members"]) == 2 for r in reps)
    lg = lambda m: math.log10(m["likes_q50"] + 1)      # noqa: E731
    gap = lg(reps[1]["kind_reports"]["launch"]["members"][0]) - lg(reps[0]["kind_reports"]["launch"]["members"][0])
    own = sum(lg(x) - lg(y) for x, y in zip(reps[1]["members"], reps[0]["members"])) / 2
    assert abs(gap - own) > 1e-6
    assert pb["p_b_better"] == pytest.approx(1 / (1 + math.exp(-2.0 * gap)), abs=1e-6) and pa["p_b_better"] == pytest.approx(1 / (1 + math.exp(-2.0 * own)), abs=1e-6)
    # a comparison that is not launch against launch is decided by the main members with the same-topic scale
    mixed = svc.analyze({"handle": "user23", "versions": [launch2[0], {"text": "some thoughts on the week ahead"}], "when": "2026-02-01T15:00:00Z"})["comparison"]["pairs"][0]
    assert mixed["decided_by_members"] == "forecast"
    with pytest.raises(ValueError, match="one bundle, one encoder"):
        service.load(api_checkpoints, {"likes": 3.0}, embedder=api.StubEmbedder(16), kind_checkpoints={"launch": [str(tmp_path / "run" / "best.pt")]})
