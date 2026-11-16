"""HTTP front for the service: one JSON endpoint and one page.

    XVIRAL_CHECKPOINTS=a/best.pt,b/best.pt XVIRAL_SCALES=pair_confidence.json [XVIRAL_POSTS=posts.parquet] [XVIRAL_SID=sid_codebook.npz] [XVIRAL_TYPE_CORRECTION=research/tracks/posttype/type_correction.json] \
        uvicorn xviral.app.server:app --port 8000
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

PAGE = Path(__file__).with_name("index.html")


class Version(BaseModel):
    text: str = Field(max_length=4000)
    image: bool = False
    video: bool = False


class PastPost(BaseModel):
    text: str = Field(max_length=4000)
    posted_at: str
    likes: int | None = None
    views: int | None = None
    replies: int | None = None
    reposts: int | None = None
    image: bool = False
    video: bool = False


class AnalyzeRequest(BaseModel):
    handle: str | None = Field(default=None, max_length=50)
    followers: int | None = Field(default=None, ge=0)
    versions: list[Version] = Field(min_length=1, max_length=6)
    past_posts: list[PastPost] = Field(default_factory=list, max_length=500)
    when: str | None = None


def create_app(service=None) -> FastAPI:
    app = FastAPI(title="Launch advisor", docs_url="/api/docs", redoc_url=None)
    state = {"service": service}
    lock = threading.Lock()          # the scorer holds caches and a model; one request at a time

    def svc():
        if state["service"] is None:
            from xviral.app import service as S

            ck = [c for c in os.environ.get("XVIRAL_CHECKPOINTS", "").split(",") if c]
            if not ck or not os.environ.get("XVIRAL_SCALES"):
                raise HTTPException(503, "the service is not configured: set XVIRAL_CHECKPOINTS and XVIRAL_SCALES")
            state["service"] = S.load(ck, os.environ["XVIRAL_SCALES"], posts=os.environ.get("XVIRAL_POSTS"), sid_codebook=os.environ.get("XVIRAL_SID"),
                                         type_correction=os.environ.get("XVIRAL_TYPE_CORRECTION"))
        return state["service"]

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.read_text()

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "loaded": state["service"] is not None}

    @app.post("/api/analyze")
    def analyze(req: AnalyzeRequest) -> dict:
        with lock:
            try:
                return svc().analyze(req.model_dump())
            except ValueError as e:
                raise HTTPException(422, str(e)) from e

    return app


app = create_app()
