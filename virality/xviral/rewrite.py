"""Candidate rewrites of a draft, written by a language model and judged by the surrogate.

The language model only proposes text. Whether a proposal is shown as better is decided by the same gates as any
other comparison (xviral/advise.py), and a proposal is dropped before scoring if it adds a number, drops a link or
drifts from what the draft says. Instructions to the model are built from effects measured on real posts
(xviral/data/launch_evidence.json), not from general advice.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable

import numpy as np
import requests

_NUM = re.compile(r"\d[\d,.]*\d|\d")
_LINK = re.compile(r"https?://\S+")
_HANDLE = re.compile(r"@\w+")

RULES = (
    "You rewrite one draft post for X (Twitter) written by a startup. You produce alternative versions of the SAME post.\n"
    "Hard rules, never broken:\n"
    "- Keep every fact. Add no fact, number, customer, benchmark, claim, date, price or feature that is not in the draft.\n"
    "- Keep every link and every @handle exactly as written. Do not add links, hashtags or emoji that the draft does not have.\n"
    "- Keep the author's voice and person (we / I). No hype words the author did not use. No questions to the reader unless the draft has one.\n"
    "- Each version must stand alone as the whole post and stay under {limit} characters.\n"
)
LAUNCH = (
    "This is a launch or announcement post. Inside accounts' own launch posts these traits went with the stronger post, so prefer them where the draft's facts allow:\n"
    "- the first line names the product;\n"
    "- the first line is the announcement itself, not a problem statement or a question;\n"
    "- the text is laid out on three or more short lines;\n"
    "- any link sits at the end, never in the first line.\n"
)
GENERAL = (
    "Traits that went with the stronger of two versions of one message by the same account, so prefer them where the draft's facts allow:\n"
    "- one point per line, three or more lines;\n"
    "- any link at the end, never in the first line;\n"
    "- concrete wording (what exactly, how much, for whom) using only what the draft already states.\n"
)


def numbers(text: str) -> set[str]:
    return {m.replace(",", "").rstrip(".") for m in _NUM.findall(_LINK.sub(" ", text or ""))}


def faithful(original: str, new: str) -> tuple[bool, str]:
    """Mechanical checks a rewrite must pass before it is scored."""
    if not new or not new.strip():
        return False, "empty"
    if set(_LINK.findall(original)) != set(_LINK.findall(new)):
        return False, "links differ from the draft"
    if not set(_HANDLE.findall(new)) <= set(_HANDLE.findall(original)):
        return False, "adds a handle"
    extra = numbers(new) - numbers(original)
    if extra:
        return False, f"adds a number ({', '.join(sorted(extra))})"
    if new.count("#") > original.count("#"):
        return False, "adds a hashtag"
    return True, ""


class OpenAIRewriter:
    """Calls an OpenAI chat model. The key comes from OPENAI_API_KEY, or is injected by a network proxy (any placeholder works then)."""

    def __init__(self, model: str = "gpt-6-luna", api_key: str | None = None, base_url: str = "https://api.openai.com/v1", timeout: float = 60.0):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "proxy-injected"

    def _chat(self, system: str, user: str, json_mode: bool = False) -> str:
        body = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(self.base_url + "/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"}, json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    def __call__(self, text: str, instruction: str) -> str:
        """The `rewrite_fn` of advise.Advisor: one rewrite following one instruction."""
        return self._chat(RULES.format(limit=max(280, int(2.5 * len(text)))) + "Return only the rewritten post, nothing else.", f"Instruction: {instruction}\n\nDraft:\n{text}").strip()

    def candidates(self, text: str, n: int = 5, launch: bool = False, limit: int | None = None) -> list[str]:
        limit = limit or max(280, int(2.0 * len(text)))
        system = RULES.format(limit=limit) + (LAUNCH if launch else GENERAL) + (
            f"Write {n} versions that differ from each other in opening line and layout, not in facts. "
            'Return JSON: {"versions": ["...", "..."]}. Use \\n for line breaks inside a version.')
        try:
            out = json.loads(self._chat(system, f"Draft:\n{text}", json_mode=True)).get("versions") or []
        except (ValueError, KeyError, requests.RequestException):
            return []
        seen, keep = {text.strip()}, []
        for v in out:
            v = str(v).strip()
            if v and v not in seen:
                seen.add(v)
                keep.append(v)
        return keep[:n]


def screen(original: str, proposals: list[str], embed_fn: Callable[[list[str]], np.ndarray] | None = None, min_cos: float = 0.75) -> list[dict]:
    """Each proposal with whether it may be scored, and why not."""
    out = [{"text": p, "ok": (f := faithful(original, p))[0], "why": f[1]} for p in proposals]
    live = [o for o in out if o["ok"]]
    if embed_fn is not None and live:
        E = np.asarray(embed_fn([original] + [o["text"] for o in live]), dtype=np.float64)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        for o, c in zip(live, E[1:] @ E[0]):
            o["meaning_cos"] = float(c)
            if c < min_cos:
                o["ok"], o["why"] = False, "drifts from what the draft says"
    return out
