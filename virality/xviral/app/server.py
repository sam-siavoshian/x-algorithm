"""HTTP front for the service: a JSON API and one page. API reference: docs/launch_advisor_api.md; live schema at /api/docs.

    XVIRAL_CHECKPOINTS=a/best.pt,b/best.pt XVIRAL_SCALES=pair_confidence.json [XVIRAL_POSTS=posts.parquet] [XVIRAL_SID=sid_codebook.npz] [XVIRAL_TYPE_CORRECTION=research/tracks/posttype/type_correction.json] \
        uvicorn xviral.app.server:app --port 8000

Every setting is an environment variable; `Settings.from_env` lists them with their defaults.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from xviral.app import accounts, media

PAGE = Path(__file__).with_name("index.html")
log = logging.getLogger("xviral.app")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


@dataclass
class Settings:
    checkpoints: list[str] = field(default_factory=list)   # XVIRAL_CHECKPOINTS   comma-separated best.pt files; several = an ensemble
    scales: str | None = None                              # XVIRAL_SCALES        pair confidence scales (research/15_pair_confidence.json)
    posts: str | None = None                               # XVIRAL_POSTS         corpus parquet for accounts we know (optional)
    sid: str | None = None                                 # XVIRAL_SID           semantic-ID codebook (optional)
    encoder: str | None = None                             # XVIRAL_ENCODER       optional: read from the checkpoints when unset; when set it must match them
    bundle: str | None = None                              # XVIRAL_BUNDLE        a bundle.json (or its folder): checkpoints, encoder, rules and fallback in one config;
                                                           #                      replaces XVIRAL_CHECKPOINTS, XVIRAL_SCALES, XVIRAL_SID, XVIRAL_TYPE_CORRECTION
    fallback_checkpoints: list[str] = field(default_factory=list)   # XVIRAL_FALLBACK_CHECKPOINTS  a model set with a local encoder, used when the embedding
    fallback_scales: str | None = None                     # XVIRAL_FALLBACK_SCALES        provider fails or a request does not allow its text to be sent
    fallback_sid: str | None = None                        # XVIRAL_FALLBACK_SID
    fallback_type_correction: str | None = None            # XVIRAL_FALLBACK_TYPE_CORRECTION
    device: str = "cpu"                                    # XVIRAL_DEVICE
    torch_threads: int = 0                                 # XVIRAL_TORCH_THREADS 0 = torch's default
    preload: bool = False                                  # XVIRAL_PRELOAD=1     load models at start-up instead of on the first request
    rate_per_minute: int = 30                              # XVIRAL_RATE_PER_MIN  analyses per caller per minute; 0 = no limit
    rate_burst: int = 10                                   # XVIRAL_RATE_BURST
    max_upload_mb: int = 64                                # XVIRAL_MAX_UPLOAD_MB per media file
    max_body_mb: int = 200                                 # XVIRAL_MAX_BODY_MB   whole request
    media_analyzer: str | None = None                      # XVIRAL_MEDIA_ANALYZER package.module:function (xviral/app/media.py)
    log_text: bool = False                                 # XVIRAL_LOG_TEXT=1    include draft text in request logs (off by default)
    log_level: str = "INFO"                                # XVIRAL_LOG_LEVEL
    judge_model: str | None = None                         # XVIRAL_JUDGE_MODEL   a language model as second opinion on version comparisons (xviral/judge.py); unset = none
    judge_endpoint: str | None = None                      # XVIRAL_JUDGE_ENDPOINT openai | publik (default: by the model's name)
    judge_deadline_s: int = 40                             # XVIRAL_JUDGE_DEADLINE seconds a request waits for the judge before the forecasting model decides alone

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        return cls(checkpoints=[c.strip() for c in e("XVIRAL_CHECKPOINTS", "").split(",") if c.strip()], scales=e("XVIRAL_SCALES") or None,
                   posts=e("XVIRAL_POSTS") or None, sid=e("XVIRAL_SID") or None, encoder=e("XVIRAL_ENCODER") or None, device=e("XVIRAL_DEVICE") or "cpu",
                   bundle=e("XVIRAL_BUNDLE") or None, fallback_checkpoints=[c.strip() for c in e("XVIRAL_FALLBACK_CHECKPOINTS", "").split(",") if c.strip()],
                   fallback_scales=e("XVIRAL_FALLBACK_SCALES") or None, fallback_sid=e("XVIRAL_FALLBACK_SID") or None,
                   fallback_type_correction=e("XVIRAL_FALLBACK_TYPE_CORRECTION") or None,
                   torch_threads=_env_int("XVIRAL_TORCH_THREADS", 0), preload=e("XVIRAL_PRELOAD", "") in ("1", "true", "yes"),
                   rate_per_minute=_env_int("XVIRAL_RATE_PER_MIN", 30), rate_burst=_env_int("XVIRAL_RATE_BURST", 10),
                   max_upload_mb=_env_int("XVIRAL_MAX_UPLOAD_MB", 64), max_body_mb=_env_int("XVIRAL_MAX_BODY_MB", 200),
                   media_analyzer=e("XVIRAL_MEDIA_ANALYZER") or None, log_text=e("XVIRAL_LOG_TEXT", "") in ("1", "true", "yes"),
                   log_level=e("XVIRAL_LOG_LEVEL") or "INFO", judge_model=e("XVIRAL_JUDGE_MODEL") or None, judge_endpoint=e("XVIRAL_JUDGE_ENDPOINT") or None,
                   judge_deadline_s=_env_int("XVIRAL_JUDGE_DEADLINE", 40))


# --- request bodies ----------------------------------------------------------------------------------------


class Version(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=4000, description="the post as it would be published")
    image: bool = Field(default=False, description="an image will be attached (when no file is uploaded)")
    video: bool = Field(default=False, description="a video or GIF will be attached (when no file is uploaded)")


class PastPost(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str = Field(max_length=25_000)
    posted_at: str
    likes: int | None = Field(default=None, ge=0)
    views: int | None = Field(default=None, ge=0)
    replies: int | None = Field(default=None, ge=0)
    reposts: int | None = Field(default=None, ge=0)
    image: bool = False
    video: bool = False


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    handle: str | None = Field(default=None, max_length=16, description="X handle, with or without @")
    followers: int | None = Field(default=None, ge=0, le=1_000_000_000, description="required when we have no posts for the account")
    versions: list[Version] = Field(min_length=1, max_length=6)
    past_posts: list[PastPost] = Field(default_factory=list, max_length=500, description="the account's past posts (for example from an X analytics export)")
    when: str | None = Field(default=None, description="planned posting time, ISO 8601; enables the 2-hour spacing check")
    suggest: int = Field(default=0, ge=0, le=8, description="how many rewrites of version 1 to propose and judge; needs XVIRAL_REWRITER_MODEL on the server")
    launch: bool | None = Field(default=None, description="force the launch plan on or off; by default it appears when a version reads as a launch")
    judge: bool = Field(default=True, description="allow the second opinion of a language model on version comparisons when the server has one (XVIRAL_JUDGE_MODEL): "
                                                  "the versions, the handle and follower count, and recent posts with like counts are sent to the provider named in the "
                                                  "report's `judge` block. false = nothing is sent to the language model; the forecasting model decides alone "
                                                  "(see external_embedding for the forecasting model's own text embedding)")
    external_embedding: bool = Field(default=True, description="allow the text of the versions and of the account's recent posts to be sent to the embedding provider "
                                                               "named in the report's `embedding` block, when the server's models use one. false = nothing is sent to it: "
                                                               "the request is answered by the server's local fallback models, or refused when it has none")
    share_past_posts: bool = Field(default=False, description="allow `past_posts` (posts you typed in or read from an X analytics export) to be sent to the language model's "
                                                              "provider. Without it a request that carries past_posts is decided by the forecasting model alone")

    @field_validator("handle")
    @classmethod
    def _handle(cls, v):
        if v is None:
            return v
        h = v.strip().lstrip("@")
        if h and not (h.replace("_", "").isalnum() and h.isascii() and len(h) <= 15):
            raise ValueError("an X handle has at most 15 letters, digits or underscores")
        return h or None

    @field_validator("versions")
    @classmethod
    def _versions(cls, v):
        if not any(x.text.strip() for x in v):
            raise ValueError("write at least one version of the post")
        return v

    @field_validator("when")
    @classmethod
    def _when(cls, v):
        if v:
            from xviral.app.service import _ms

            _ms(v)
        return v


# --- abuse protection -------------------------------------------------------------------------------------


class RateLimiter:
    """A token bucket per caller: `per_minute` sustained, `burst` at once. In memory: one process, one bucket set."""

    def __init__(self, per_minute: int, burst: int):
        self.rate, self.burst = per_minute / 60.0, max(burst, 1)
        self._b: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def take(self, key: str, now: float | None = None) -> float:
        """0 when allowed, else the seconds to wait."""
        if self.rate <= 0:
            return 0.0
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._b.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1:
                self._b[key] = (tokens - 1, now)
                if len(self._b) > 50_000:
                    self._b = {k: v for k, v in self._b.items() if now - v[1] < 600}
                return 0.0
            self._b[key] = (tokens, now)
            return (1 - tokens) / self.rate


def _readable(errors: list[dict]) -> str:
    out = []
    for e in errors:
        loc = [str(x) for x in e.get("loc", []) if x not in ("body",)]
        where = ""
        if loc[:1] == ["versions"] and len(loc) > 1 and loc[1].isdigit():
            where = f"version {int(loc[1]) + 1}" + (f", {loc[2]}" if len(loc) > 2 else "")
        elif loc[:1] == ["past_posts"] and len(loc) > 1 and loc[1].isdigit():
            where = f"past post {int(loc[1]) + 1}" + (f", {loc[2]}" if len(loc) > 2 else "")
        elif loc:
            where = loc[0]
        msg = e.get("msg", "invalid").removeprefix("Value error, ")
        if e.get("type") == "string_too_long":
            msg = f"is too long (at most {e.get('ctx', {}).get('max_length')} characters)"
        elif e.get("type") == "too_long" and loc[:1] == ["versions"]:
            msg = "at most six versions can be compared"
        elif e.get("type") == "too_short" and loc[:1] == ["versions"]:
            msg = "write at least one version of the post"
        elif e.get("type") == "extra_forbidden":
            msg = "is not a field this API takes"
        out.append(f"{where}: {msg}" if where else msg)
    return "; ".join(dict.fromkeys(out)) or "the request could not be read"


# --- the app ----------------------------------------------------------------------------------------------


def create_app(service=None, settings: Settings | None = None, gate: accounts.Gate | None = None) -> FastAPI:
    cfg = settings or Settings.from_env()
    logging.basicConfig(level=cfg.log_level, format="%(message)s")
    app = FastAPI(title="Launch advisor", version="2", docs_url="/api/docs", redoc_url=None,
                  description="Forecasts and comparisons for draft X posts. Reference: docs/launch_advisor_api.md.")
    state = {"service": service, "error": None, "started": time.time()}
    lock = threading.Lock()          # the scorer holds caches and a model; one analysis at a time
    load_lock = threading.Lock()
    limiter = RateLimiter(cfg.rate_per_minute, cfg.rate_burst)
    gate = gate or accounts.make_gate()
    analyzer = media.load_analyzer(cfg.media_analyzer)

    def svc():
        if state["service"] is None:
            with load_lock:
                if state["service"] is None:
                    if not cfg.bundle and (not cfg.checkpoints or not cfg.scales):
                        raise HTTPException(503, "the service is not configured: set XVIRAL_BUNDLE, or XVIRAL_CHECKPOINTS and XVIRAL_SCALES")
                    from xviral.app import service as S

                    if cfg.torch_threads:
                        import torch

                        torch.set_num_threads(cfg.torch_threads)
                    try:
                        judge = None
                        if cfg.judge_model:
                            from xviral.judge import Judge

                            judge = Judge.from_env({**os.environ, "XVIRAL_JUDGE_MODEL": cfg.judge_model, "XVIRAL_JUDGE_ENDPOINT": cfg.judge_endpoint or ""})
                        if cfg.bundle:
                            state["service"] = S.load_bundle(cfg.bundle, posts=cfg.posts, judge=judge, judge_deadline=cfg.judge_deadline_s, device=cfg.device)
                        else:
                            fb = None
                            if cfg.fallback_checkpoints and cfg.fallback_scales:
                                fb = S.load(cfg.fallback_checkpoints, cfg.fallback_scales, posts=cfg.posts, sid_codebook=cfg.fallback_sid, device=cfg.device,
                                            type_correction=cfg.fallback_type_correction, judge=judge, judge_deadline=cfg.judge_deadline_s, model_set="fallback")
                            state["service"] = S.load(cfg.checkpoints, cfg.scales, posts=cfg.posts, sid_codebook=cfg.sid, encoder=cfg.encoder, device=cfg.device,
                                                      type_correction=os.environ.get("XVIRAL_TYPE_CORRECTION") or None, judge=judge, judge_deadline=cfg.judge_deadline_s,
                                                      judge_rule=os.environ.get("XVIRAL_JUDGE_RULE") or None, selective_rule=os.environ.get("XVIRAL_SELECTIVE_RULE") or None,
                                                      selective_ensemble=os.environ.get("XVIRAL_SELECTIVE_ENSEMBLE") or None, fallback=fb)
                        if os.environ.get("XVIRAL_REWRITER_MODEL"):      # an OpenAI chat model; the key is OPENAI_API_KEY or proxy-injected
                            from xviral.rewrite import OpenAIRewriter

                            state["service"].rewriter = OpenAIRewriter(os.environ["XVIRAL_REWRITER_MODEL"])
                    except Exception as e:      # noqa: BLE001
                        state["error"] = f"{type(e).__name__}: {e}"
                        log.exception("model load failed")
                        raise HTTPException(503, "the models could not be loaded; see the server log") from e
        return state["service"]

    if cfg.preload and service is None:
        @app.on_event("startup")
        def _preload():
            threading.Thread(target=lambda: _safe(svc), daemon=True).start()

    def _safe(f):
        try:
            f()
        except HTTPException:
            pass

    @app.middleware("http")
    async def frame(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.rid, request.state.log = rid, {}
        t0 = time.perf_counter()
        n = request.headers.get("content-length")
        if n and n.isdigit() and int(n) > cfg.max_body_mb * 1024 * 1024:
            resp = JSONResponse({"detail": f"the request is larger than {cfg.max_body_mb} MB"}, status_code=413)
        else:
            resp = await call_next(request)
        resp.headers["x-request-id"] = rid
        resp.headers["x-content-type-options"] = "nosniff"
        resp.headers["referrer-policy"] = "no-referrer"
        if request.url.path.startswith("/api/") and request.url.path not in ("/api/health", "/api/ready"):
            rec = {"event": "request", "id": rid, "method": request.method, "path": request.url.path, "status": resp.status_code,
                   "ms": round(1000 * (time.perf_counter() - t0)), **request.state.log}
            log.info(json.dumps(rec, default=str))
        return resp

    @app.exception_handler(Exception)
    async def _crash(request: Request, exc: Exception):
        rid = getattr(request.state, "rid", "")
        log.exception(json.dumps({"event": "error", "id": rid, "path": request.url.path, "error": type(exc).__name__}))
        return JSONResponse({"detail": f"something went wrong on our side; the request ID is {rid}"}, status_code=500, headers={"x-request-id": rid})

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError):
        return JSONResponse({"detail": _readable(exc.errors()), "errors": [{k: e.get(k) for k in ("loc", "msg", "type")} for e in exc.errors()]}, status_code=422)

    def enter(request: Request, endpoint: str, limited: bool = True) -> accounts.Caller:
        try:
            caller = gate.authorize({k.lower(): v for k, v in request.headers.items()}, request.client.host if request.client else "unknown", endpoint)
        except accounts.Denied as d:
            raise HTTPException(d.status, d.message) from d
        request.state.log["caller"] = caller.id if not caller.id.startswith("ip:") else "ip"
        if limited:
            wait = limiter.take(caller.id)
            if wait:
                raise HTTPException(429, f"too many requests; try again in {math.ceil(wait)} seconds", headers={"Retry-After": str(math.ceil(wait))})
        return caller

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page() -> str:
        return PAGE.read_text()

    @app.get("/api/health", tags=["operations"], summary="Liveness: the process answers")
    def health() -> dict:
        s = state["service"]
        return {"ok": True, "loaded": s is not None, "load_error": state["error"], "uptime_s": round(time.time() - state["started"]),
                "corpus": bool(s and s.posts), "media_analyzer": analyzer is not None,
                "model_set": s.model_set if s else None, "embedding": s.embedding_info() if s else None, "fallback_model_set": bool(s and s.fallback)}

    @app.get("/api/ready", tags=["operations"], summary="Readiness: models loaded (503 until then)")
    def ready():
        if state["service"] is None:
            return JSONResponse({"ready": False, "load_error": state["error"]}, status_code=503)
        return {"ready": True}

    @app.get("/api/account", tags=["analysis"], summary="What we hold for a handle; embeds its recent posts ahead of the first report")
    def account(request: Request, handle: str) -> dict:
        enter(request, "account", limited=False)
        try:
            AnalyzeRequest(versions=[Version(text="x")], handle=handle)
        except ValidationError as e:
            raise HTTPException(422, _readable(e.errors())) from e
        with lock:
            return svc().account(handle)

    @app.post("/api/import/analytics", tags=["analysis"], summary="Read an X analytics CSV export into past_posts")
    async def import_analytics(request: Request) -> dict:
        enter(request, "import", limited=False)
        form = await request.form()
        f = form.get("file")
        if f is None or not hasattr(f, "read"):
            raise HTTPException(422, "attach the CSV file as the form field 'file'")
        data = await f.read()
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(413, "the CSV is larger than 20 MB")
        from xviral.app.service import Service

        try:
            out = Service.import_analytics(data, handle=form.get("handle"))
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        request.state.log.update(import_rows=out["summary"]["rows"], import_posts=out["summary"]["posts"])
        return out

    @app.post("/api/analyze", tags=["analysis"], summary="Forecast, compare and check one to six versions of a post",
              openapi_extra={"requestBody": {"content": {"application/json": {"schema": AnalyzeRequest.model_json_schema()},
                                                         "multipart/form-data": {"schema": {"type": "object", "properties": {
                                                             "request": {"type": "string", "description": "the JSON body as a string"},
                                                             **{f"media_{i}": {"type": "string", "format": "binary", "description": f"image or video for version {i}"} for i in range(1, 7)}}}}},
                                         "required": True}})
    async def analyze(request: Request) -> dict:
        caller = enter(request, "analyze")
        files: dict[int, tuple[bytes, str | None]] = {}
        ctype = request.headers.get("content-type", "")
        try:
            if ctype.startswith("multipart/form-data"):
                form = await request.form()
                raw = form.get("request")
                if not isinstance(raw, str):
                    raise HTTPException(422, "a multipart request needs the JSON body in the form field 'request'")
                body = json.loads(raw)
                for i in range(1, 7):
                    f = form.get(f"media_{i}")
                    if f is not None and hasattr(f, "read"):
                        data = await f.read()
                        if data:
                            files[i] = (data, getattr(f, "filename", None))
            else:
                body = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(422, "the request body is not valid JSON") from e
        try:
            req = AnalyzeRequest.model_validate(body)
        except ValidationError as e:
            raise HTTPException(422, _readable(e.errors())) from e
        payload = req.model_dump()
        for i, (data, name) in files.items():
            if i > len(payload["versions"]):
                raise HTTPException(422, f"media_{i} is attached but there is no version {i}")
            try:
                info = media.probe(data, name, max_bytes=cfg.max_upload_mb * 1024 * 1024)
            except media.MediaError as e:
                raise HTTPException(422, f"version {i}: {e}") from e
            u = media.understand(info, data, analyzer)
            v = payload["versions"][i - 1]
            v["media_info"] = info.public()
            if u is not None:
                v["media_attributes"], v["media_embedding"] = u.attributes, u.embedding
            del data
        files.clear()                  # the uploaded bytes are not kept past this point
        lg = request.state.log
        lg.update(versions=len(payload["versions"]), past_posts=len(payload["past_posts"]), handle_given=bool(payload["handle"]),
                  chars=[len(v["text"]) for v in payload["versions"]], media=[(v.get("media_info") or {}).get("kind") or ("video" if v["video"] else "image" if v["image"] else None)
                                                                              for v in payload["versions"]])
        if cfg.log_text:
            lg["texts"] = [v["text"] for v in payload["versions"]]
        from starlette.concurrency import run_in_threadpool

        def work():
            with lock:
                return svc().analyze(payload)
        from xviral.app.service import EmbeddingUnavailable

        try:
            out = await run_in_threadpool(work)
        except EmbeddingUnavailable as e:      # the embedding provider failed and there is no fallback model set: say so, score nothing
            lg.update(embedding={"error": "provider failed, no fallback"})
            raise HTTPException(503, str(e), headers={"Retry-After": "5"}) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        for v in payload["versions"]:
            v.pop("media_embedding", None)
        lg.update(history_source=out["account"]["history_source"], history_posts=out["account"]["history_posts"], timing_ms=out.get("timing_ms"))
        eb = out.get("embedding") or {}
        if eb.get("external") or eb.get("fallback_used"):      # what the embedding cost and whether text left the server for it; never the text
            lg.update(embedding={"encoder": eb.get("encoder"), "sent": eb.get("sent") is not None, "texts_sent": eb.get("texts_sent"), "cost_usd": eb.get("cost_usd"),
                                 "ms": eb.get("ms"), "fallback_used": eb.get("fallback_used"), "fallback_reason": eb.get("fallback_reason")})
        jb = out.get("judge") or {}
        if jb.get("configured"):          # what left the server for this request, without any text
            lg.update(judge={"used": jb["used"], "sent": jb["sent"] is not None, "model": jb["model"], "pairs": jb["pairs"], "cost_usd": jb["cost_usd"], "added_ms": jb["added_ms"], "why_not": jb["why_not"]})
        gate.record_usage(caller, accounts.Usage("analyze", versions=len(payload["versions"]), media_files=sum(1 for v in payload["versions"] if v.get("media_info")),
                                                 history_posts=out["account"]["history_posts"], ms=out["timing_ms"]["total"]))
        return out

    return app


app = create_app()
