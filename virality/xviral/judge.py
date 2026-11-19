"""A second opinion on which of two versions does better: a language model that reads both versions and the account's
recent posts with their like counts, blended with the forecasting model (research/tracks/judge_product/REPORT.md).

    judge = Judge("gpt-6.1-sol")                                   # or Judge.from_env(): XVIRAL_JUDGE_MODEL
    out = judge.compare_many(account, history_posts, [(a, b), ...])   # both orders of every pair, all calls concurrent
    rule = JudgeRule.load()                                        # xviral/data/judge_rule.json, fitted on development pairs
    d = rule.decide(surrogate_gap, out[0]["logodds"], kind="launch", model=judge.model)

What leaves the machine: the two versions being compared, the account line (name, handle, follower count) and up to
`history_posts` of the account's earlier posts with their like counts. Nothing else. The caller decides whether the
history may be sent (xviral/app/service.py never sends posts that came from an upload unless the request allows it).

The prompt and the rendering of a post are the ones every number in the report was measured with; change them and the
rule file no longer applies.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

RULE_PATH = Path(__file__).parent / "data" / "judge_rule.json"
HANDLE = re.compile(r"@\w+")
TCO = re.compile(r"https?://t\.co/\w+")
LINK = re.compile(r"https?://\S+")

SYSTEM = ("You judge posts on X (Twitter). You are shown two posts, A and B, written by the same account. "
          "One of them received clearly more likes than the other. Decide which one.\n"
          "When earlier posts by this account are listed with the likes each received, use them: they show what this "
          "account's audience rewards and what it ignores. Judge the two posts as that audience would meet them in the feed: "
          "the hook of the first line, how concrete and clear the claim is, whether it stands alone without the link, format and media.\n"
          "{answer}")
ANS_LP = "Answer with a single letter, A or B, and nothing else."
ANS_JSON = 'Return JSON {{"p_a": x}} where x is your probability from 0 to 1 that post A received more likes than post B.'
SCHEMA = {"type": "json_schema", "json_schema": {"name": "judgement", "strict": True, "schema": {
    "type": "object", "properties": {"p_a": {"type": "number"}}, "required": ["p_a"], "additionalProperties": False}}}

ENDPOINTS = {
    "openai": {"base_url": "https://api.openai.com/v1", "provider": "OpenAI", "key_env": "OPENAI_API_KEY"},
    "publik": {"base_url": "https://publikhq.com/api/v1", "provider": "Publik API (models served by DeepInfra)", "key_env": "PUBLIK_API_KEY"},
}
# mode: "logprob" = one output token, P(A) from the token log-probabilities; "json" = {"p_a": x}, the model may reason first.
# price: US dollars per million tokens (input, cached input, output), from the providers' price lists on 2026-11-19.
MODELS = {
    "gpt-6-luna": {"endpoint": "openai", "mode": "logprob", "reasoning_effort": "none", "price": (0.10, 0.01, 0.50), "knowledge_cutoff": "2026-05-18"},
    "gpt-6.1-sol": {"endpoint": "openai", "mode": "json", "reasoning_effort": "low", "price": (2.00, 0.10, 10.00), "knowledge_cutoff": "2026-04-30"},
    "gpt-6-sol": {"endpoint": "openai", "mode": "json", "reasoning_effort": "low", "price": (2.00, 0.20, 10.00), "knowledge_cutoff": None},
    "openai/gpt-oss-120b": {"endpoint": "publik", "mode": "json", "reasoning_effort": None, "price": (0.037, 0.0185, 0.17), "knowledge_cutoff": "2024-06"},
    "deepseek-ai/DeepSeek-V4.1-Flash": {"endpoint": "publik", "mode": "logprob", "reasoning_effort": None, "price": (0.20, 0.006, 0.60), "knowledge_cutoff": None},
    "Qwen/Qwen3.5-397B-A17B": {"endpoint": "publik", "mode": "logprob", "reasoning_effort": "none", "price": (0.45, 0.22, 3.00), "knowledge_cutoff": None},
}


# --- what the model reads ---------------------------------------------------------------------------------


def render_post(text: str | None, media: list | None = None, urls: list | None = None, quoted_text: str | None = None, kind: str | None = None,
                likes: int | None = None) -> dict:
    """A post (a corpus row, an uploaded past post or a draft) as the judge reads it: text without link wrappers, the kind
    of media, the domain of up to two links, the quoted post's text. No time and, for a draft, no outcome."""
    t = text or ""
    dom = [u.get("final_domain") or "" for u in (urls or []) if isinstance(u, dict) and not u.get("is_media")]
    if not urls:                         # a draft or a typed post: the link is written out in the text
        for u in LINK.findall(TCO.sub("", t)):
            try:
                host = urlparse(u.rstrip(".,;:!?)")).netloc.lower()
            except ValueError:              # not a URL a parser accepts (for example an unclosed bracket): no domain tag
                host = ""
            dom.append(host[4:] if host.startswith("www.") else host)
            t = t.replace(u, "")
    t = TCO.sub("", t).strip()
    m = sorted({x.get("type") for x in (media or []) if isinstance(x, dict) and x.get("type")})
    return {"text": t, "media": ("video" if ("video" in m or "animated_gif" in m) else "image" if m else ""), "links": [d for d in dom if d][:2],
            "quoted": (quoted_text or "")[:400] if kind == "quote" else "", "likes": likes}


def history_posts(history, n: int = 30) -> list[dict]:
    """The account's `n` most recent posts with a like count, oldest first, rendered. `history`: a polars frame of the
    post schema (xviral.schema) sorted or not; replies and posts without a like count are left out."""
    if history is None or not getattr(history, "height", 0) or n <= 0:
        return []
    import polars as pl

    h = history.filter(pl.col("likes").is_not_null())
    if "is_reply" in h.columns:
        h = h.filter(~pl.col("is_reply").fill_null(False))
    h = h.sort("created_at_ms").tail(n)
    cols = [c for c in ("text", "media", "urls", "quoted_text", "kind", "likes") if c in h.columns]
    return [render_post(r.get("text"), r.get("media"), r.get("urls"), r.get("quoted_text"), r.get("kind"), int(r["likes"])) for r in h.select(cols).iter_rows(named=True)]


def fmt(p: dict, mask: bool = False, cap: int = 0) -> str:
    t = p["text"]
    if cap and len(t) > cap:
        t = t[:cap] + "…"
    if mask:
        t = HANDLE.sub("@user", t)
    tags = []
    if p.get("media"):
        tags.append(f"[{p['media']} attached]")
    if p.get("links"):
        tags.append("[link: " + ", ".join(p["links"]) + "]")
    if p.get("quoted"):
        q = p["quoted"][:300].replace("\n", " ")
        tags.append("[quotes another post: \"" + (HANDLE.sub("@user", q) if mask else q) + "\"]")
    return (t + ("\n" if tags else "") + " ".join(tags)).strip()


def evidence_note(evidence: dict | None = None) -> str:
    """The measured launch effects (xviral/data/launch_evidence.json) as one paragraph for the prompt variant "ev"."""
    if evidence is None:
        evidence = json.loads((Path(__file__).parent / "data" / "launch_evidence.json").read_text())
    rows = []
    for k, e in evidence.items():
        if k.startswith("_") or not isinstance(e, dict) or not e.get("grade") or not e.get("win_rate") or not e.get("description"):
            continue
        r = (e.get("likes_ratio") or {}).get("ratio")
        rows.append((-e["win_rate"], f"- {e['description'].split(' (')[0]}: did better {100 * e['win_rate']:.0f}% of the time in {e['pairs']:,} pairs"
                     + (f", about {r:.1f}x the likes" if r else "")))
    return ("Measured on launch and announcement posts (one account's launch post against another of its launch posts, by likes), these went "
            "with the stronger post. They are tendencies, weaker than what the account's own history shows:\n" + "\n".join(t for _, t in sorted(rows)))


def build_messages(account: dict | None, history: list[dict], a: dict, b: dict, mode: str = "json", mask: bool = False, note: str | None = None) -> list[dict]:
    """account: {"name", "handle", "followers"} or None; history: rendered posts with likes, oldest first; a, b: rendered posts."""
    parts = []
    if account and not mask:
        f = account.get("followers")
        parts.append(f"Account: {account.get('name') or ''} (@{account.get('handle') or ''})" + (f", about {f:,} followers" if f else ""))
    if history:
        parts.append("Earlier posts by this account, oldest first, with likes received:\n"
                     + "\n".join(f"- ({x['likes']} likes) " + fmt(x, mask, 400).replace("\n", " / ") for x in history))
    parts.append(f"Post A:\n{fmt(a, mask)}\n\nPost B:\n{fmt(b, mask)}\n\nWhich post received more likes?")
    answer = ANS_LP if mode == "logprob" else ANS_JSON      # ANS_JSON goes in as written (doubled braces): the measured prompt
    system = SYSTEM.format(answer=(note + "\n" if note else "") + answer)
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 30.0), -30.0)))


def order_average(p0: float | None, p1: float | None) -> float | None:
    """Log-odds that the first post of the pair does better, from P(A) with the pair shown as (a, b) and as (b, a)."""
    if p0 is None or p1 is None:
        return None
    return (logit(p0) - logit(p1)) / 2


def parse_logprob(js: dict) -> float | None:
    """P(A) from the first output token's log-probabilities."""
    try:
        first = js["choices"][0]["logprobs"]["content"][0]
    except (KeyError, IndexError, TypeError):
        return None
    top = first.get("top_logprobs") or [first]
    pa = sum(math.exp(t["logprob"]) for t in top if (t.get("token") or "").strip().upper() == "A")
    pb = sum(math.exp(t["logprob"]) for t in top if (t.get("token") or "").strip().upper() == "B")
    if pa + pb < 0.5:
        return None
    if pa == 0 or pb == 0:              # the other letter is outside what the endpoint returned: it holds at most the remainder
        pa, pb = (pa, max(1 - pa, 1e-7)) if pb == 0 else (max(1 - pb, 1e-7), pb)
    return max(pa, 1e-7) / (max(pa, 1e-7) + max(pb, 1e-7))


def parse_json(js: dict) -> float | None:
    try:
        txt = js["choices"][0]["message"]["content"] or ""
        m = re.search(r"\{[^{}]*\}", txt)
        return min(1.0, max(0.0, float(json.loads(m.group(0) if m else txt)["p_a"])))
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return None


# --- the judge --------------------------------------------------------------------------------------------


class Judge:
    """Asks one chat model, through an OpenAI-compatible endpoint, which of two posts did better. Every pair is asked in
    both orders and the two log-odds are averaged. `spend` accumulates requests, tokens and dollars over the object's life."""

    def __init__(self, model: str, endpoint: str | None = None, mode: str | None = None, reasoning_effort: str | None = "default",
                 history_posts: int = 30, timeout: float = 45.0, retries: int = 2, concurrency: int = 16, api_key: str | None = None,
                 base_url: str | None = None, mask: bool = False, note: str | None = None, session=None):
        spec = MODELS.get(model, {})
        self.model = model
        self.endpoint = endpoint or spec.get("endpoint") or ("publik" if "/" in model else "openai")
        if self.endpoint not in ENDPOINTS:
            raise ValueError(f"unknown judge endpoint {self.endpoint!r}; use one of {sorted(ENDPOINTS)}")
        ep = ENDPOINTS[self.endpoint]
        self.base_url = (base_url or ep["base_url"]).rstrip("/")
        self.provider = ep["provider"]
        self.mode = mode or spec.get("mode") or "json"
        self.reasoning_effort = spec.get("reasoning_effort") if reasoning_effort == "default" else reasoning_effort
        self.price = spec.get("price") or (0.0, 0.0, 0.0)
        self.history_n, self.timeout, self.retries, self.concurrency = history_posts, timeout, retries, concurrency
        self.mask, self.note = mask, note
        self.api_key = api_key or os.environ.get(ep["key_env"]) or "proxy-injected"     # a network proxy may inject the real key
        self._session = session
        self._lock = threading.Lock()
        self.spend = {"requests": 0, "errors": 0, "tokens_in": 0, "tokens_cached": 0, "tokens_out": 0, "usd": 0.0}

    @classmethod
    def from_env(cls, env=None) -> "Judge | None":
        """XVIRAL_JUDGE_MODEL (required to have a judge), XVIRAL_JUDGE_ENDPOINT (openai | publik; default by model name),
        XVIRAL_JUDGE_TIMEOUT (seconds per call), XVIRAL_JUDGE_HISTORY (posts of history sent), XVIRAL_JUDGE_CONCURRENCY."""
        e = (env or os.environ).get
        model = (e("XVIRAL_JUDGE_MODEL") or "").strip()
        if not model:
            return None
        if model.startswith("stub"):
            return StubJudge()
        return cls(model, endpoint=(e("XVIRAL_JUDGE_ENDPOINT") or "").strip() or None, timeout=float(e("XVIRAL_JUDGE_TIMEOUT") or 45),
                   history_posts=int(e("XVIRAL_JUDGE_HISTORY") or 30), concurrency=int(e("XVIRAL_JUDGE_CONCURRENCY") or 16))

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "endpoint": self.base_url}

    # one request -------------------------------------------------------------------------------------------

    def body(self, messages: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages}
        tok = "max_completion_tokens" if self.endpoint == "openai" else "max_tokens"
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.mode == "logprob":
            body.update({"logprobs": True, "top_logprobs": 5, tok: 4})
        else:
            body.update({tok: 3000, "response_format": SCHEMA})
        return body

    def cost(self, usage: dict) -> float:
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        pi, pc, po = self.price
        return ((usage.get("prompt_tokens", 0) - cached) * pi + cached * pc + usage.get("completion_tokens", 0) * po) / 1e6

    def _post(self, body: dict) -> tuple[int, dict]:
        import requests

        s = self._session or requests
        r = s.post(self.base_url + "/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"}, json=body, timeout=self.timeout)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"error": (r.text or "")[:200]}

    def ask(self, messages: list[dict]) -> dict:
        """One request with retries. {"p": P(A) or None, "error": str or None, "tokens_in", "tokens_cached", "tokens_out", "usd", "seconds"}."""
        t0, err, body = time.perf_counter(), None, self.body(messages)
        out = {"p": None, "error": None, "tokens_in": 0, "tokens_cached": 0, "tokens_out": 0, "usd": 0.0}
        for attempt in range(self.retries + 1):
            try:
                status, js = self._post(body)
            except Exception as e:      # noqa: BLE001  timeouts, connection errors
                err = type(e).__name__
            else:
                if status == 200:
                    u = js.get("usage") or {}
                    out["tokens_in"] += u.get("prompt_tokens", 0)
                    out["tokens_cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                    out["tokens_out"] += u.get("completion_tokens", 0)
                    out["usd"] += self.cost(u)
                    p = parse_logprob(js) if self.mode == "logprob" else parse_json(js)
                    if p is not None:
                        out["p"], err = p, None
                        break
                    err = "unreadable answer"
                else:
                    err = f"HTTP {status}"
                    if status not in (408, 409, 429, 500, 502, 503, 504):
                        break               # a refused request will be refused again
            if attempt < self.retries:
                time.sleep(min(0.5 * 2 ** attempt, 8.0))
        out["error"], out["seconds"] = err, time.perf_counter() - t0
        with self._lock:
            self.spend["requests"] += 1
            self.spend["errors"] += out["p"] is None
            for k in ("tokens_in", "tokens_cached", "tokens_out", "usd"):
                self.spend[k] += out[k]
        return out

    # pairs -------------------------------------------------------------------------------------------------

    def compare_many(self, account: dict | None, history: list[dict], pairs: list[tuple[dict, dict]]) -> list[dict]:
        """pairs: (a, b) rendered posts. Each result: {"ok", "logodds" (that a does better), "p_a", "p_orders", "error",
        "tokens_in", "tokens_out", "usd", "seconds" (the slower of its two calls)}. All calls of the batch run concurrently."""
        if not pairs:
            return []
        hist = history[-self.history_n:] if self.history_n else []
        jobs = [build_messages(account, hist, x, y, self.mode, self.mask, self.note) for a, b in pairs for x, y in ((a, b), (b, a))]
        with ThreadPoolExecutor(max_workers=max(1, min(self.concurrency, len(jobs)))) as ex:
            res = list(ex.map(self.ask, jobs))
        out = []
        for i in range(len(pairs)):
            r0, r1 = res[2 * i], res[2 * i + 1]
            j = order_average(r0["p"], r1["p"])
            out.append({"ok": j is not None, "logodds": j, "p_a": None if j is None else sigmoid(j), "p_orders": [r0["p"], r1["p"]],
                        "error": r0["error"] or r1["error"], "tokens_in": r0["tokens_in"] + r1["tokens_in"], "tokens_out": r0["tokens_out"] + r1["tokens_out"],
                        "usd": r0["usd"] + r1["usd"], "seconds": max(r0["seconds"], r1["seconds"])})
        return out

    def compare(self, account: dict | None, history: list[dict], a: dict, b: dict) -> dict:
        return self.compare_many(account, history, [(a, b)])[0]


class JudgeBatch:
    """The judge calls of one request, started in the background so they overlap with the forecasting model's scoring.
    `results()` waits at most until `deadline` seconds after the start and never raises: a pair without an answer is
    None, and the caller falls back to the forecasting model alone for it."""

    def __init__(self, judge: Judge, account: dict | None, history: list[dict], pairs: list[tuple[dict, dict]], deadline: float = 40.0):
        self.judge, self.n, self.deadline, self.t0 = judge, len(pairs), deadline, time.perf_counter()
        self.history_sent, self.error, self._res, self.seconds = min(len(history), judge.history_n), None, None, None
        self._ex = ThreadPoolExecutor(max_workers=1)
        self._f = self._ex.submit(judge.compare_many, account, history, pairs)
        self._ex.shutdown(wait=False)

    def results(self) -> list[dict | None]:
        if self._res is None:
            try:
                res = self._f.result(timeout=max(0.0, self.deadline - (time.perf_counter() - self.t0)))
                self._res = [r if r and r.get("ok") else None for r in res]
                self._raw = res
            except Exception as e:      # noqa: BLE001  the deadline passed, or the judge itself broke
                self.error = "deadline" if type(e).__name__ == "TimeoutError" else type(e).__name__
                self._res, self._raw = [None] * self.n, []
            self.seconds = time.perf_counter() - self.t0
        return self._res

    def summary(self) -> dict:
        res = self.results()
        raw = getattr(self, "_raw", [])
        return {"pairs": self.n, "pairs_answered": sum(r is not None for r in res), "calls": 2 * self.n, "seconds": self.seconds,
                "usd": float(sum(r.get("usd", 0.0) for r in raw)), "tokens_in": int(sum(r.get("tokens_in", 0) for r in raw)),
                "tokens_out": int(sum(r.get("tokens_out", 0) for r in raw)), "error": self.error or next((r["error"] for r in raw if r.get("error")), None),
                "history_posts_sent": self.history_sent}


class StubJudge(Judge):
    """A judge that never leaves the process, for tests and dry runs. Deterministic: by default it prefers the longer
    text, with a small offset derived from a hash of the two texts, and it has a position bias (it leans to "A") that
    asking both orders removes. `fn(messages) -> P(A)` replaces that; `fail` (True, or a set of call numbers) makes calls fail."""

    def __init__(self, fn=None, fail=False, delay: float = 0.0, model: str = "stub-judge", history_posts: int = 30, concurrency: int = 16, price=(1.0, 0.0, 1.0)):
        super().__init__(model, endpoint="openai", mode="json", history_posts=history_posts, concurrency=concurrency, retries=0)
        self.provider, self.base_url, self.price = "none (built-in stub, nothing is sent)", "stub://", price
        self.fn, self.fail, self.delay, self.calls = fn, fail, delay, []

    @staticmethod
    def _posts(messages: list[dict]) -> tuple[str, str]:
        user = messages[-1]["content"]
        a = user.split("Post A:\n", 1)[1].split("\n\nPost B:\n", 1)
        return a[0], a[1].rsplit("\n\nWhich post received more likes?", 1)[0]

    def _post(self, body: dict) -> tuple[int, dict]:
        with self._lock:
            n = len(self.calls)
            self.calls.append(body["messages"])
        if self.delay:
            time.sleep(self.delay)
        if self.fail is True or (isinstance(self.fail, (set, list, tuple)) and n in self.fail):
            return 503, {"error": "stub failure"}
        if self.fn is not None:
            p = float(self.fn(body["messages"]))
        else:
            a, b = self._posts(body["messages"])
            h = int(hashlib.md5("\x00".join(sorted((a, b))).encode()).hexdigest(), 16) % 1000 / 1000 - 0.5
            lean = (1 if a <= b else -1) * 0.4 * h
            p = sigmoid(0.02 * (len(a) - len(b)) + lean + 0.3)
        tin = sum(len(m["content"]) for m in body["messages"]) // 4
        return 200, {"choices": [{"message": {"content": json.dumps({"p_a": p})}}], "usage": {"prompt_tokens": tin, "completion_tokens": 8}}


# --- the blend and the speak rule -------------------------------------------------------------------------


class JudgeRule:
    """P(the first version does better) from the forecasting model's gap and the judge's log-odds, and whether to speak.

    Per judge model and per kind of comparison ("launch", "same_topic") the file holds
        z = w_surrogate * gap + w_judge * clip(judge log-odds, -clip, +clip),    p = sigmoid(z)
    with `gap` in log10 of predicted likes (the scorer's own unit: a fixed scale, nothing is rescaled per request), and
    a speak rule {"p": threshold on max(p, 1 - p), "require_agreement": bool, "or_judge_p": threshold or None}.
    """

    def __init__(self, rule: dict):
        self.rule = rule

    @classmethod
    def load(cls, path: str | Path | None = None) -> "JudgeRule":
        return cls(json.loads(Path(path or RULE_PATH).read_text()))

    def spec(self, model: str, kind: str) -> dict | None:
        m = self.rule.get("models", {}).get(model)
        if m is None:
            return None
        return m["kinds"].get(kind) or m["kinds"].get("same_topic")

    def validated(self, model: str, kind: str) -> dict | None:
        s = self.spec(model, kind)
        return None if s is None else s.get("validated")

    def decide(self, gap: float, judge_logodds: float, kind: str, model: str, speak: float | None = None) -> dict | None:
        """gap: surrogate, first version minus second, log10 predicted likes. Returns None when the file has no rule for the model."""
        s = self.spec(model, kind)
        if s is None:
            return None
        c = float(s.get("clip", 4.0))
        j = max(-c, min(c, float(judge_logodds)))
        zs, zj = s["w_surrogate"] * gap, s["w_judge"] * j
        p = sigmoid(zs + zj + s.get("bias", 0.0))
        p_s = sigmoid(s.get("surrogate_scale", s["w_surrogate"]) * gap)
        p_j = sigmoid(s.get("judge_scale", 1.0) * j)
        sp = s.get("speak") or {}
        thr = max(float(sp.get("p", 0.6)), float(speak or 0.0))          # the caller may be stricter than the fitted rule, never looser
        conf = max(p, 1 - p)
        pick_first = p >= 0.5
        agree = None if gap == 0 or j == 0 else (gap > 0) == (j > 0)
        ok = conf >= thr
        if ok and sp.get("require_agreement"):
            alt = sp.get("or_judge_p")
            ok = bool(agree) or (alt is not None and max(p_j, 1 - p_j) >= alt and (j > 0) == pick_first)
        return {"p_first_better": p, "speak": bool(ok), "threshold": thr, "agree": agree, "kind": kind, "model": model,
                "surrogate": {"pick_first": None if gap == 0 else gap > 0, "p_first_better": p_s, "gap_log10": gap},
                "judge": {"pick_first": None if j == 0 else j > 0, "p_first_better": p_j, "logodds": float(judge_logodds)},
                "why_silent": None if ok else ("the forecasting model and the reading model point in opposite directions" if agree is False
                                               else "the two versions are too close to call")}
