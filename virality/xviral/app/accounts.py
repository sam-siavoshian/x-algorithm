"""The seam for accounts and billing. Out of scope today; this is where it goes.

Every API request passes through `Gate.authorize` before any work and `Gate.record_usage` after a report is made.
The default gate lets everyone in as an anonymous caller keyed by IP address (that key is what the rate limit
counts). Setting XVIRAL_API_KEYS=key1,key2 requires `Authorization: Bearer <key>` on the API (not on the page or the
health checks). A real accounts-and-billing layer replaces the gate:

    XVIRAL_GATE=package.module:make_gate        # a zero-argument callable returning an object with these two methods

`authorize` returns a Caller or raises `Denied` (shown to the user with its status code: 401 unknown key, 402 out of
credit, 403 plan does not include the feature). `record_usage` receives the caller and a Usage record with no draft
text in it. Plans and quotas belong in the gate; the rate limit in server.py is only abuse protection.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from dataclasses import dataclass, field


@dataclass
class Caller:
    id: str                     # stable key for rate limiting and usage: an account ID, or "ip:<address>"
    plan: str = "anonymous"
    attributes: dict = field(default_factory=dict)


@dataclass
class Usage:
    endpoint: str
    versions: int = 0
    media_files: int = 0
    history_posts: int = 0
    ms: int = 0


class Denied(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


class Gate:
    def __init__(self, api_keys: list[str] | None = None):
        self._keys = {hashlib.sha256(k.encode()).hexdigest(): k for k in (api_keys or []) if k}

    def authorize(self, headers: dict, client_ip: str, endpoint: str) -> Caller:
        if not self._keys:
            return Caller(id=f"ip:{client_ip}")
        auth = headers.get("authorization", "")
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        digest = hashlib.sha256(key.encode()).hexdigest()
        if not key or digest not in self._keys:
            raise Denied(401, "an API key is required: send Authorization: Bearer <key>")
        return Caller(id=f"key:{digest[:12]}", plan="key")

    def record_usage(self, caller: Caller, usage: Usage) -> None:
        return None


def make_gate() -> Gate:
    spec = os.environ.get("XVIRAL_GATE")
    if spec:
        mod, _, name = spec.partition(":")
        return getattr(importlib.import_module(mod), name)()
    return Gate([k.strip() for k in os.environ.get("XVIRAL_API_KEYS", "").split(",") if k.strip()])
