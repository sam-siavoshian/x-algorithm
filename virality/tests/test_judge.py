"""The second opinion (xviral/judge.py): what the model is sent, both orders, retries and cost, the fitted blend rule, and
its place in the product: used when configured, off on request, never fed uploaded posts without permission, and replaced
by the forecasting model alone on any failure. The language model is always a stub here; nothing leaves the process."""

import json
import math
import time

import polars as pl
import pytest

from tests.test_app import V, _analytics_csv, _diff, _services, ensemble  # noqa: F401
from tests.test_serve import embed, trained  # noqa: F401
from xviral import judge as J
from xviral import serve

RULE = {"version": 1, "models": {"stub-judge": {"kinds": {
    "same_topic": {"w_surrogate": 2.0, "w_judge": 0.5, "clip": 4.0, "surrogate_scale": 2.77, "judge_scale": 0.6, "speak": {"p": 0.6},
                   "validated": {"statement": "TEST same-topic statement: 61% of 100 pairs."}},
    "launch": {"w_surrogate": 1.5, "w_judge": 0.9, "clip": 4.0, "surrogate_scale": 1.84, "judge_scale": 0.9, "speak": {"p": 0.6, "require_agreement": True},
               "validated": {"statement": "TEST launch statement: 80% of 100 pairs."}}}}}}


WHEN = "2027-03-01T16:00:00Z"          # a fixed posting time, so that two reports score the same moment


# --- what is sent -------------------------------------------------------------------------------------------


def test_rendering_and_prompt():
    a = J.render_post("Introducing Atlas 2 https://t.co/abc123 more at https://www.example.com/atlas?x=1.", [{"type": "video"}])
    assert a == {"text": "Introducing Atlas 2  more at", "media": "video", "links": ["example.com"], "quoted": "", "likes": None}
    row = J.render_post("read this https://t.co/zzz", [{"type": "photo"}], [{"final_domain": "arxiv.org", "is_media": False}, {"final_domain": "x.com", "is_media": True}], "the quoted words", "quote", 12)
    assert row["text"] == "read this" and row["media"] == "image" and row["links"] == ["arxiv.org"] and row["quoted"] == "the quoted words" and row["likes"] == 12
    assert J.render_post("plain", None, None, "ignored", "original")["quoted"] == ""
    assert J.render_post("see http://[abc now")["links"] == []                         # a string no URL parser accepts does not raise
    b = J.render_post("hello @bob")
    hist = [{"text": "old post\nsecond line", "media": "", "links": [], "quoted": "", "likes": 7}]
    m = J.build_messages({"name": "Acme", "handle": "acme", "followers": 12_345}, hist, a, b, "json")
    assert m[0]["role"] == "system" and 'Return JSON {{"p_a": x}}' in m[0]["content"]          # the prompt as measured, braces included
    u = m[1]["content"]
    assert u.startswith("Account: Acme (@acme), about 12,345 followers\n\nEarlier posts by this account, oldest first, with likes received:\n- (7 likes) old post / second line")
    assert "Post A:\nIntroducing Atlas 2  more at\n[video attached] [link: example.com]\n\nPost B:\nhello @bob\n\nWhich post received more likes?" in u
    lp = J.build_messages(None, [], b, a, "logprob", mask=True)
    assert lp[0]["content"].endswith("Answer with a single letter, A or B, and nothing else.") and lp[1]["content"].startswith("Post A:\nhello @user")
    assert "Account:" not in lp[1]["content"] and "Earlier posts" not in lp[1]["content"]
    note = J.evidence_note({"x": {"grade": "A", "win_rate": 0.64, "pairs": 1672, "description": "the first line names the product (a proper name)", "likes_ratio": {"ratio": 1.5}},
                            "y": {"grade": None, "win_rate": 0.9, "pairs": 5, "description": "ungraded"}, "_meta": {}})
    assert "the first line names the product: did better 64% of the time in 1,672 pairs, about 1.5x the likes" in note and "ungraded" not in note
    assert note in J.build_messages(None, [], a, b, "json", note=note)[0]["content"]


def test_history_posts_are_recent_liked_non_replies():
    h = pl.DataFrame({"text": [f"post {i}" for i in range(6)], "likes": [1, None, 3, 4, 5, 6], "created_at_ms": [5, 4, 3, 2, 1, 0], "is_reply": [False, False, False, True, False, False],
                      "media": [[], [], [{"type": "photo"}], [], [], []]}, schema_overrides={"media": pl.List(pl.Struct({"type": pl.String}))})
    out = J.history_posts(h, 3)
    assert [(p["text"], p["likes"]) for p in out] == [("post 4", 5), ("post 2", 3), ("post 0", 1)] and out[1]["media"] == "image"
    assert J.history_posts(h, 0) == [] and J.history_posts(None, 30) == [] and J.history_posts(h.clear(), 30) == []


def test_answer_parsing():
    lp = lambda toks: {"choices": [{"logprobs": {"content": [{"token": toks[0][0], "logprob": math.log(toks[0][1]), "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in toks]}]}}]}  # noqa: E731
    assert J.parse_logprob(lp([("A", 0.75), ("B", 0.25)])) == pytest.approx(0.75)
    assert J.parse_logprob(lp([(" b", 0.6), ("A", 0.3), ("The", 0.1)])) == pytest.approx(1 / 3)
    assert J.parse_logprob(lp([("The", 0.9), ("A", 0.05)])) is None                               # the model did not answer with a letter
    only = {"choices": [{"logprobs": {"content": [{"token": "A", "logprob": math.log(0.9)}]}}]}   # an endpoint that returns the chosen token only
    assert J.parse_logprob(only) == pytest.approx(0.9)
    assert J.parse_logprob({"choices": [{}]}) is None
    js = lambda s: {"choices": [{"message": {"content": s}}]}  # noqa: E731
    assert J.parse_json(js('{"p_a": 0.8}')) == 0.8 and J.parse_json(js('Sure: {"p_a": 1.7}')) == 1.0 and J.parse_json(js("no json")) is None and J.parse_json(js(None)) is None
    assert J.order_average(0.8, 0.2) == pytest.approx(J.logit(0.8)) and J.order_average(0.9, 0.9) == pytest.approx(0.0) and J.order_average(0.8, None) is None


# --- the client -----------------------------------------------------------------------------------------------


class FakeSession:
    """Stands in for `requests`: a list of responses (status, json) or exceptions, replayed in order."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def post(self, url, headers=None, json=None, timeout=None):  # noqa: A002
        self.calls.append({"url": url, "headers": headers, "body": json, "timeout": timeout})
        r = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(r, Exception):
            raise r
        status, body = r

        class R:
            status_code = status
            text = "x"

            @staticmethod
            def json():
                return body
        return R()


def _ok(p, tin=1000, tout=10, cached=0):
    return 200, {"choices": [{"message": {"content": json.dumps({"p_a": p})}}], "usage": {"prompt_tokens": tin, "completion_tokens": tout, "prompt_tokens_details": {"cached_tokens": cached}}}


def test_requests_retries_timeouts_and_cost(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    a, b = J.render_post("one"), J.render_post("two")
    s = FakeSession([(500, {"error": "boom"}), TimeoutError("slow"), _ok(0.8, 1000, 10, 400)])
    j = J.Judge("gpt-6.1-sol", session=s, retries=2, timeout=7)
    r = j.ask(J.build_messages(None, [], a, b, j.mode))
    assert r["p"] == 0.8 and r["error"] is None and len(s.calls) == 3 and s.calls[0]["timeout"] == 7
    assert r["usd"] == pytest.approx((600 * 2.0 + 400 * 0.10 + 10 * 10.0) / 1e6) and j.spend["requests"] == 1 and j.spend["errors"] == 0
    body = s.calls[0]["body"]
    assert s.calls[0]["url"] == "https://api.openai.com/v1/chat/completions" and s.calls[0]["headers"]["Authorization"].startswith("Bearer ")
    assert body["model"] == "gpt-6.1-sol" and body["reasoning_effort"] == "low" and body["max_completion_tokens"] == 3000 and body["response_format"]["type"] == "json_schema"
    # a refused request is not retried; a judge that never answers gives an error, not an exception
    s = FakeSession([(400, {"error": "bad"})])
    j = J.Judge("gpt-6.1-sol", session=s, retries=3)
    r = j.ask([{"role": "user", "content": "x"}])
    assert r["p"] is None and r["error"] == "HTTP 400" and len(s.calls) == 1 and j.spend["errors"] == 1
    s = FakeSession([ConnectionError("down")])
    r = J.Judge("gpt-6-luna", session=s, retries=1).compare(None, [], a, b)
    assert not r["ok"] and r["logodds"] is None and r["error"] == "ConnectionError" and len(s.calls) == 4          # two orders, each tried twice
    # the Publik endpoint: full vendor/model name, its own token-limit field, log-probabilities
    s = FakeSession([(200, {"choices": [{"logprobs": {"content": [{"token": "B", "logprob": math.log(0.7)}]}}], "usage": {"prompt_tokens": 2000, "completion_tokens": 2}})])
    j = J.Judge("deepseek-ai/DeepSeek-V4.1-Flash", session=s)
    r = j.compare({"handle": "acme", "followers": 10}, [], a, b)
    assert j.endpoint == "publik" and j.provider.startswith("Publik") and s.calls[0]["url"] == "https://publikhq.com/api/v1/chat/completions"
    assert s.calls[0]["body"]["max_tokens"] == 4 and s.calls[0]["body"]["logprobs"] is True and "max_completion_tokens" not in s.calls[0]["body"]
    assert r["ok"] and r["logodds"] == pytest.approx(0.0) and r["p_orders"] == [pytest.approx(0.3), pytest.approx(0.3)]      # says "B" in both orders: no preference
    assert r["usd"] == pytest.approx(2 * (2000 * 0.20 + 2 * 0.60) / 1e6) and r["tokens_in"] == 4000
    with pytest.raises(ValueError):
        J.Judge("x", endpoint="elsewhere")
    assert J.Judge.from_env({}) is None and J.Judge.from_env({"XVIRAL_JUDGE_MODEL": "Qwen/Qwen3.5-397B-A17B"}).endpoint == "publik"
    e = J.Judge.from_env({"XVIRAL_JUDGE_MODEL": "gpt-6.1-sol", "XVIRAL_JUDGE_TIMEOUT": "12", "XVIRAL_JUDGE_HISTORY": "10"})
    assert (e.model, e.endpoint, e.timeout, e.history_n, e.mode) == ("gpt-6.1-sol", "openai", 12.0, 10, "json")
    assert isinstance(J.Judge.from_env({"XVIRAL_JUDGE_MODEL": "stub"}), J.StubJudge)


def test_stub_is_deterministic_order_balanced_and_concurrent():
    a, b, c = J.render_post("a much longer first version of the post, with detail"), J.render_post("short"), J.render_post("medium length post")
    j = J.StubJudge()
    r1 = j.compare_many({"handle": "x"}, [], [(a, b), (b, a), (a, c)])
    r2 = J.StubJudge().compare_many({"handle": "x"}, [], [(a, b), (b, a), (a, c)])
    assert [x["logodds"] for x in r1] == [x["logodds"] for x in r2]                                # deterministic
    assert r1[0]["logodds"] == pytest.approx(-r1[1]["logodds"]) and r1[0]["logodds"] > 0            # swapping the pair flips the sign: no position bias left
    assert r1[0]["p_orders"][0] + r1[0]["p_orders"][1] != pytest.approx(1.0)                        # the raw answers do lean to "A"
    assert j.spend["requests"] == 6 and len(j.calls) == 6 and j.spend["usd"] > 0 and r1[0]["usd"] > 0
    slow = J.StubJudge(delay=0.1)
    t = time.perf_counter()
    out = slow.compare_many(None, [], [(a, b)] * 6)
    assert time.perf_counter() - t < 0.6 and all(o["ok"] for o in out)                               # 12 calls of 0.1 s each, in parallel
    half = J.StubJudge(fail={1}).compare_many(None, [], [(a, b), (a, c)])
    assert not half[0]["ok"] and half[0]["error"] == "HTTP 503" and half[1]["ok"]                    # one order missing: that pair has no answer
    batch = J.JudgeBatch(J.StubJudge(delay=0.3), None, [], [(a, b)], deadline=0.05)
    assert batch.results() == [None] and batch.summary()["error"] == "deadline" and batch.summary()["pairs_answered"] == 0
    hist = [J.render_post(f"p{i}", likes=i) for i in range(50)]
    j = J.StubJudge(history_posts=30)
    j.compare(None, hist, a, b)
    assert j.calls[0][1]["content"].count("\n- (") == 30 and "- (49 likes) p49" in j.calls[0][1]["content"] and "- (19 likes)" not in j.calls[0][1]["content"]


# --- the rule ---------------------------------------------------------------------------------------------------


def test_rule_blend_and_speak():
    rule = J.JudgeRule(RULE)
    d = rule.decide(0.1, 2.0, "same_topic", "stub-judge")
    assert d["p_first_better"] == pytest.approx(1 / (1 + math.exp(-(2.0 * 0.1 + 0.5 * 2.0)))) and d["speak"] and d["agree"] is True
    assert d["surrogate"]["p_first_better"] == pytest.approx(1 / (1 + math.exp(-0.277))) and d["judge"]["p_first_better"] == pytest.approx(1 / (1 + math.exp(-1.2)))
    assert rule.decide(0.1, 99.0, "same_topic", "stub-judge")["p_first_better"] == rule.decide(0.1, 4.0, "same_topic", "stub-judge")["p_first_better"]      # clipped
    m = rule.decide(-0.1, -2.0, "same_topic", "stub-judge")
    assert m["p_first_better"] == pytest.approx(1 - d["p_first_better"]) and m["speak"]            # symmetric in the two versions
    close = rule.decide(0.02, 0.1, "same_topic", "stub-judge")
    assert not close["speak"] and close["why_silent"] == "the two versions are too close to call"
    # launch pairs in this test rule need agreement: a confident blend of two opposed opinions stays silent
    split = rule.decide(-0.3, 3.0, "launch", "stub-judge")
    assert split["p_first_better"] > 0.6 and split["agree"] is False and not split["speak"] and "opposite directions" in split["why_silent"]
    assert rule.decide(0.3, 3.0, "launch", "stub-judge")["speak"]
    assert rule.decide(0.1, 2.0, "same_topic", "stub-judge", speak=0.99)["speak"] is False            # the caller may be stricter
    assert rule.decide(0.1, 2.0, "same_topic", "stub-judge", speak=0.10)["threshold"] == 0.6           # never looser than the fitted rule
    same = rule.decide(0.0, 0.0, "same_topic", "stub-judge")                                       # two identical versions: neither opinion prefers one
    assert same["agree"] is None and same["surrogate"]["pick_first"] is None and same["judge"]["pick_first"] is None and not same["speak"]
    assert rule.decide(0.1, 2.0, "other_kind", "stub-judge")["kind"] == "other_kind" and rule.decide(0.1, 2.0, "launch", "unknown-model") is None
    assert rule.validated("stub-judge", "launch")["statement"].startswith("TEST launch")


def test_shipped_rule_file():
    rule = J.JudgeRule.load()
    assert set(rule.rule["models"]) >= {"gpt-6.1-sol", "gpt-6-luna"}
    for model, m in rule.rule["models"].items():
        assert model in J.MODELS, model
        for kind in ("same_topic", "launch"):
            s = m["kinds"][kind]
            assert s["w_surrogate"] > 0 and s["w_judge"] > 0 and s["clip"] > 0 and 0.5 <= s["speak"]["p"] < 1, (model, kind)
            v = s["validated"]
            assert v["statement"] and v["pairs"] > 100 and 0.4 < v["accuracy"] < 1 and model.split("/")[-1] in v["statement"], (model, kind)
            ps = [rule.decide(0.05, x, kind, model)["p_first_better"] for x in (-3, -1, 0, 1, 3)]
            assert ps == sorted(ps) and ps[0] < 0.5 < ps[-1]
            assert rule.decide(0.0, 0.0, kind, model)["p_first_better"] == pytest.approx(0.5) and not rule.decide(0.0, 0.0, kind, model)["speak"]


# --- in the product -----------------------------------------------------------------------------------------------


def _svc(trained, ensemble, judge, **kw):  # noqa: F811
    return _services(trained, ensemble, judge=judge, judge_rule=J.JudgeRule(RULE), **kw)[1]


def _prefer(word, p=0.95):
    """A stub answer: P(A) high when post A contains `word`, low when post B does."""
    def fn(messages):
        a, b = J.StubJudge._posts(messages)
        return p if word in a and word not in b else 1 - p if word in b and word not in a else 0.5
    return fn


def test_comparisons_use_both_opinions_and_say_which_agreed(trained, ensemble):  # noqa: F811
    plain = _services(trained, ensemble)[1]
    base = plain.analyze({"handle": "user23", "versions": V[:2], "when": WHEN})
    assert base["judge"] == {"configured": False, "used": False, "provider": None, "model": None, "why_not": "no language model is configured on this server", "sent": None,
                             "pairs": 0, "pairs_answered": 0, "calls": 0, "cost_usd": 0.0, "added_ms": 0, "error": None}
    assert base["method"]["judge_validation"] is None and base["method"]["comparisons_decided_by"] == "forecasting model"
    assert base["comparison"]["pairs"][0]["decided_by"] == "forecasting model" and "opinions" not in base["comparison"]

    stub = J.StubJudge(fn=_prefer("launch"))
    r = _svc(trained, ensemble, stub).analyze({"handle": "user23", "versions": V[:2], "when": WHEN})
    jb = r["judge"]
    assert jb["used"] and jb["provider"] == stub.provider and jb["model"] == "stub-judge" and jb["pairs"] == 1 and jb["pairs_answered"] == 1 and jb["calls"] == 2 == len(stub.calls)
    assert jb["cost_usd"] > 0 and jb["why_not"] is None and jb["error"] is None and r["timing_ms"]["judge_wait"] >= 0
    assert jb["sent"]["versions"] == 2 and jb["sent"]["past_posts"] == 30 and jb["sent"]["past_posts_source"] == "our corpus of public posts" and jb["sent"]["to"] == stub.provider
    user = stub.calls[0][1]["content"]
    assert user.startswith("Account:  (@user23), about ") and user.count("\n- (") == 30 and V[0]["text"] in user and V[1]["text"] in user
    assert "[image attached]" in user                                                  # version 2 carries an image
    pair = r["comparison"]["pairs"][0]
    o = pair["opinions"]
    assert pair["decided_by"] == "forecasting model and language model" and o["language_model_prefers"] == 2 and o["forecast_model_prefers"] in (1, 2)
    assert o["agree"] == (o["forecast_model_prefers"] == 2)
    # the number is the rule's: forecasting model's gap and the judge's log-odds through the file, nothing else
    d = J.JudgeRule(RULE).decide(_gap(plain, base), J.logit(0.95), "same_topic", "stub-judge")
    assert pair["p_b_better"] == pytest.approx(d["p_first_better"], rel=1e-6) and pair["p_b_better_forecast_model_only"] == pytest.approx(base["comparison"]["pairs"][0]["p_b_better"], rel=1e-6)
    assert r["versions"][1]["vs_version_1"]["p_better"] == pytest.approx(pair["p_b_better"]) and r["versions"][1]["vs_version_1"]["opinions"] == o
    assert r["comparison"]["opinions"]["pairs_with_both"] == 1 and "preferred the same version in" in r["comparison"]["opinions"]["statement"]
    if r["comparison"]["winner"] == 2:
        assert ("both prefer it" in r["comparison"]["statement"]) == o["agree"]
    assert r["method"]["judge_validation"] == "TEST same-topic statement: 61% of 100 pairs." and r["method"]["comparisons_decided_by"] == "forecasting model and language model"
    # the method note leads with the configuration that decided the comparisons; the forecasting model's own record is kept beside it
    assert r["method"]["validation"] == r["method"]["judge_validation"] and base["method"]["validation"] == base["method"]["forecast_model_validation"] == r["method"]["forecast_model_validation"]
    assert base["method"]["validation"].startswith("Version comparisons are checked on 2,852 real pairs")
    # the forecast itself is untouched by the judge
    assert [v["likes_median"] for v in r["versions"]] == pytest.approx([v["likes_median"] for v in base["versions"]], rel=1e-6)
    # the judge can turn the call around: it now prefers version 1 just as strongly
    stub2 = J.StubJudge(fn=_prefer("thoughts", 0.999))
    r2 = _svc(trained, ensemble, stub2).analyze({"handle": "user23", "versions": V[:2], "when": WHEN})
    assert r2["comparison"]["pairs"][0]["p_b_better"] < pair["p_b_better"] and r2["comparison"]["pairs"][0]["opinions"]["language_model_prefers"] == 1


def _gap(svc, report):
    """The forecasting model's gap (version 2 minus version 1, mean over members of log10 median likes), recomputed through the advisor."""
    h, _, f = svc._history("user23", None, [], 0)
    drafts = [svc._draft(v) for v in V[:2]]
    from xviral.app.service import _ms

    when = _ms(WHEN)
    reps = svc.scorer.score(h, drafts, {"handle": "user23", "followers": int(f)}, when)
    return sum(math.log10(b["likes_q50"] + 1) - math.log10(a["likes_q50"] + 1) for a, b in zip(reps[0]["members"], reps[1]["members"])) / len(reps[0]["members"])


def test_six_versions_one_concurrent_batch_and_launch_kind(trained, ensemble):  # noqa: F811
    stub = J.StubJudge(delay=0.05)
    r = _svc(trained, ensemble, stub).analyze({"handle": "user23", "versions": V})
    assert r["judge"]["pairs"] == 15 and r["judge"]["calls"] == 30 == len(stub.calls) and r["judge"]["added_ms"] < 1500      # 30 calls of 50 ms, overlapping the scoring
    assert all(p["decided_by"] == "forecasting model and language model" for p in r["comparison"]["pairs"]) and r["comparison"]["opinions"]["pairs_with_both"] == 15
    # single defined edits (attach an image, move the link, ...) are never sent: every call is one of the 15 version pairs
    texts = {v["text"] for v in V}
    for call in stub.calls:
        a, b = J.StubJudge._posts(call)
        assert any(t.split(" https://")[0].split("\n")[0] in a for t in texts) and any(t.split(" https://")[0].split("\n")[0] in b for t in texts)
    assert all("opinions" not in c for c in r["changes"])
    # two launch posts are compared under the launch rule (it needs agreement in the test rule), a launch and a non-launch under the other
    svc = _svc(trained, ensemble, J.StubJudge(fn=_prefer("Atlas 2: agents")))
    rep = lambda t, likes: {"likes": {"q50": likes}, "views_24h": {"q50": 40 * likes}, "members": [{"likes_q50": likes, "views_q50": 40 * likes}] * 2, "text": t}  # noqa: E731
    la, lb, plain = rep("Introducing Atlas 2: agents that file your taxes", 20), rep("Launching Atlas 2 today: our new product", 30), rep("some thoughts on the week ahead", 30)
    assert svc.advisor._kind(la, lb) == "launch" and svc.advisor._kind(la, plain) == "same_topic"
    c = svc.advisor.compare(lb, la, {"ok": True, "logodds": 3.0})          # la against lb: the judge prefers la, the forecast lb
    assert c["comparison_kind"] == "launch" and c["opinions"]["agree"] is False and c["status"] == "no_confident_difference" and "opposite directions" in c["why"]
    c = svc.advisor.compare(plain, la, {"ok": True, "logodds": 3.0})
    assert c["comparison_kind"] == "same_topic" and c["status"].startswith("recommended") and c["opinions"]["language_model"]["prefers"] == "this version"
    c = svc.advisor.compare(la, lb, {"ok": True, "logodds": -3.0})
    assert c["status"] == "no_confident_difference"
    # a verdict the forecasting model did not share is stated without its "about Nx the likes", which would contradict it
    v = svc.compare_versions([plain, la], {(0, 1): {"ok": True, "logodds": 3.0}})
    assert v["winner"] == 2 and v["statement"] == f"Version 2 is predicted to do better than version 1 ({100 * v['confidence']:.0f}% likely). The language model prefers it; the forecasting model alone does not."
    assert v["pairs"][0]["predicted_ratio_b_over_a"] < 1 and v["pairs"][0]["opinions"] == {**v["pairs"][0]["opinions"], "forecast_model_prefers": 1, "language_model_prefers": 2, "agree": False}
    c = svc.advisor.compare(la, plain, {"ok": True, "logodds": -3.0})        # plain against la: both the blend and the judge say worse
    assert c["status"] == "predicted_worse" and c["p_better"] < 0.4


def test_opt_out_uploaded_posts_and_single_version(trained, ensemble):  # noqa: F811
    stub = J.StubJudge()
    svc = _svc(trained, ensemble, stub)
    base = _services(trained, ensemble)[1].analyze({"handle": "user23", "versions": V[:2], "when": WHEN})
    off = svc.analyze({"handle": "user23", "versions": V[:2], "judge": False, "when": WHEN})
    assert stub.calls == [] and not off["judge"]["used"] and off["judge"]["configured"] and off["judge"]["model"] == "stub-judge" and "opted out" in off["judge"]["why_not"]
    assert off["judge"]["sent"] is None and off["method"]["judge_validation"] is None
    assert off["comparison"] == base["comparison"] and off["versions"] == base["versions"] and off["pick"] == base["pick"]           # exactly the report without a judge
    one = svc.analyze({"handle": "user23", "versions": V[:1]})
    assert stub.calls == [] and not one["judge"]["used"] and "nothing to compare" in one["judge"]["why_not"]
    # posts that came with the request (an X analytics export) stay on the server unless the request allows sending them
    past = svc.import_analytics(_analytics_csv())["past_posts"]
    kept = svc.analyze({"handle": "newco", "followers": 9_000, "versions": V[:2], "past_posts": past})
    assert stub.calls == [] and not kept["judge"]["used"] and "share_past_posts" in kept["judge"]["why_not"] and kept["judge"]["sent"] is None
    assert kept["comparison"]["pairs"][0]["decided_by"] == "forecasting model"
    sent = svc.analyze({"handle": "newco", "followers": 9_000, "versions": V[:2], "past_posts": past, "share_past_posts": True})
    assert len(stub.calls) == 2 and sent["judge"]["used"] and sent["judge"]["sent"]["past_posts"] == 29 and sent["judge"]["sent"]["past_posts_source"] == "posts you supplied"
    user = stub.calls[0][1]["content"]
    assert "- (400 likes) launch day: we shipped the new agent 0" in user and "- (60 likes) some thoughts on the week 1" in user and "@someone thanks" not in user
    assert "2400" not in user and "16000" not in user                                 # like counts go, view counts do not
    # an account we know nothing about: the versions go without any history
    stub.calls.clear()
    new = svc.analyze({"handle": "brandnew", "followers": 500, "versions": V[:2]})
    assert new["judge"]["used"] and new["judge"]["sent"]["past_posts"] == 0 and "Earlier posts" not in stub.calls[0][1]["content"]


def test_only_posts_from_before_the_planned_time_are_sent(trained, ensemble):  # noqa: F811
    from xviral.app.service import _iso

    stub = J.StubJudge()
    svc = _svc(trained, ensemble, stub)
    h = svc._history("user23", None, [], 0)[0].sort("created_at_ms")
    mid = int(h["created_at_ms"][h.height // 4])
    r = svc.analyze({"handle": "user23", "versions": V[:2], "when": _iso(mid + 1000)})
    earlier = h.filter((pl.col("created_at_ms") < mid + 1000) & pl.col("likes").is_not_null() & ~pl.col("is_reply").fill_null(False))
    n = min(30, earlier.height)
    assert 0 < n < 30 and h.height > 30                                                # the planned time cuts the history short
    assert r["judge"]["used"] and r["judge"]["sent"]["past_posts"] == n and stub.calls[0][1]["content"].count("\n- (") == n


def test_judge_failure_falls_back_to_the_forecasting_model(trained, ensemble):  # noqa: F811
    base = _services(trained, ensemble)[1].analyze({"handle": "user23", "versions": V[:3], "when": WHEN})
    for stub, why in ((J.StubJudge(fail=True), "HTTP 503"), (J.StubJudge(delay=1.0), "deadline")):
        svc = _svc(trained, ensemble, stub, judge_deadline=0.2)
        t = time.perf_counter()
        r = svc.analyze({"handle": "user23", "versions": V[:3], "when": WHEN})
        assert time.perf_counter() - t < 20
        assert not r["judge"]["used"] and r["judge"]["pairs"] == 3 and r["judge"]["pairs_answered"] == 0 and r["judge"]["error"] == why
        assert ("calls failed (HTTP 503)" if why == "HTTP 503" else "did not answer in time") in r["judge"]["why_not"] and r["method"]["judge_validation"] is None
        assert r["judge"]["sent"]["versions"] == 3 and "were sent all the same" in r["judge"]["why_not"]          # a failed call still left the server: the report says so
        worst, other = _diff({k: base[k] for k in ("comparison", "versions", "pick", "changes")}, {k: r[k] for k in ("comparison", "versions", "pick", "changes")})
        assert worst[0] < 1e-9 and not other                                            # the same report as with no judge at all
    # one pair of three fails: that pair alone is decided by the forecasting model
    svc = _svc(trained, ensemble, J.StubJudge(fail={0}))
    r = svc.analyze({"handle": "user23", "versions": V[:3]})
    by = [p["decided_by"] for p in r["comparison"]["pairs"]]
    assert sorted(by) == ["forecasting model", "forecasting model and language model", "forecasting model and language model"] and r["judge"]["used"] and r["judge"]["pairs_answered"] == 2
    assert "decided by the forecasting model alone" in r["comparison"]["opinions"]["statement"]
    # a model the rule file does not know: configured, named, not used
    odd = J.StubJudge(model="some-new-model")
    r = _svc(trained, ensemble, odd).analyze({"handle": "user23", "versions": V[:2]})
    assert odd.calls == [] and not r["judge"]["used"] and "no fitted rule" in r["judge"]["why_not"] and r["judge"]["model"] == "some-new-model"


def test_rewrites_are_judged_against_the_draft(trained):  # noqa: F811
    from xviral.app import service

    tmp_path, posts, table, mcfg = trained
    posts.write_parquet(tmp_path / "posts3.parquet")
    stub = J.StubJudge(fn=_prefer("other"))
    svc = service.Service(serve.DraftScorer(tmp_path / "run" / "best.pt", embed), {"likes": 6.0, "views": 6.0}, posts=tmp_path / "posts3.parquet", judge=stub, judge_rule=J.JudgeRule(RULE))

    class Rewriter:
        model = "stub"

        def candidates(self, text, n=5, launch=False, limit=None):
            return ["some other thoughts on the week ahead", "some thoughts on the week, 99 of them"][:n]

    svc.rewriter = Rewriter()
    r = svc.analyze({"handle": "user23", "versions": [{"text": "some thoughts on the week ahead"}], "suggest": 2})
    rows = r["suggested"]["better"] + r["suggested"]["not_better"]
    assert len(rows) == 1 and len(stub.calls) == 2                                      # the faithful rewrite only, in both orders
    assert rows[0]["decided_by"] == "forecasting model and language model" and rows[0]["opinions"]["language_model_prefers"] == "rewrite"
    assert r["judge"]["used"] and r["judge"]["pairs"] == 1 and r["judge"]["sent"]["rewrites"] == 1 and "second language model" in r["suggested"]["written_by"]
    assert "_judge" not in r["suggested"]
    stub.calls.clear()
    r = svc.analyze({"handle": "user23", "versions": [{"text": "some thoughts on the week ahead"}], "suggest": 2, "judge": False})
    assert stub.calls == [] and (r["suggested"]["better"] + r["suggested"]["not_better"])[0]["decided_by"] == "forecasting model"


def test_http_fields_settings_and_log(trained, ensemble, monkeypatch, caplog):  # noqa: F811
    import logging

    from fastapi.testclient import TestClient

    from xviral.app import server

    monkeypatch.setenv("XVIRAL_JUDGE_MODEL", "gpt-6.1-sol")
    monkeypatch.setenv("XVIRAL_JUDGE_DEADLINE", "15")
    s = server.Settings.from_env()
    assert s.judge_model == "gpt-6.1-sol" and s.judge_endpoint is None and s.judge_deadline_s == 15
    monkeypatch.delenv("XVIRAL_JUDGE_MODEL")
    assert server.Settings.from_env().judge_model is None
    stub = J.StubJudge()
    client = TestClient(server.create_app(_svc(trained, ensemble, stub), settings=server.Settings(rate_per_minute=600, rate_burst=100)))
    with caplog.at_level(logging.INFO, logger="xviral.app"):
        r = client.post("/api/analyze", json={"handle": "user23", "versions": V[:2]})
    assert r.status_code == 200 and r.json()["judge"]["used"] and r.json()["judge"]["provider"] == stub.provider
    rec = [json.loads(m.message) for m in caplog.records if m.message.startswith("{") and '"request"' in m.message][-1]
    assert rec["judge"]["used"] and rec["judge"]["model"] == "stub-judge" and V[0]["text"] not in json.dumps(rec)
    n = len(stub.calls)
    r = client.post("/api/analyze", json={"handle": "user23", "versions": V[:2], "judge": False})
    assert r.status_code == 200 and not r.json()["judge"]["used"] and len(stub.calls) == n
    r = client.post("/api/analyze", json={"handle": "newco", "followers": 100, "versions": V[:2], "past_posts": [{"text": "hello there", "posted_at": "2026-10-01T10:00:00Z", "likes": 3}] * 12})
    assert r.status_code == 200 and not r.json()["judge"]["used"] and len(stub.calls) == n
    r = client.post("/api/analyze", json={"handle": "newco", "followers": 100, "versions": V[:2], "share_past_posts": True,
                                          "past_posts": [{"text": f"hello there {i}", "posted_at": f"2026-10-{i + 1:02d}T10:00:00Z", "likes": 3} for i in range(12)]})
    assert r.status_code == 200 and r.json()["judge"]["used"] and r.json()["judge"]["sent"]["past_posts"] == 12
    schema = client.get("/openapi.json").json()
    assert "share_past_posts" in json.dumps(schema) and "judge" in json.dumps(schema)
