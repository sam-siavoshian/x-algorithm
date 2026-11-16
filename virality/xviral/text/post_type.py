"""What kind of company post a draft is. Rules, first match wins; the order puts the rarer, more specific kinds first."""

from __future__ import annotations

import re

_F = re.IGNORECASE
TYPES: list[tuple[str, str, re.Pattern]] = [
    ("funding", "Funding announcement", re.compile(
        r"\b(raised|raising|closed)\b.{0,40}(\$|€|£)\s?\d|\b(seed|pre-seed|series [a-e])\b.{0,30}\b(round|funding|led by)\b|\bled by\b.{0,60}\b(capital|ventures|vc|partners)\b|\bbacked by\b", _F)),
    ("hiring", "Hiring", re.compile(r"\bwe(?:'re| are) hiring\b|\bjoin (?:us|our team)\b|\bopen roles?\b|\bhiring (?:a|an|our|for|engineers?|designers?)\b|\bcome work with\b", _F)),
    ("launch", "Product launch", re.compile(
        r"^\W*(introducing|announcing|meet|say hello to|launching)\b|\b(we(?:'re| are| just)?|i(?:'m| am| just)?) (?:excited to |thrilled to |proud to )?(?:launch(?:ed|ing)?|announc(?:e|ed|ing)|introduc(?:e|ed|ing)|unveil(?:ed|ing)?)\b"
        r"|\bis (?:now |finally )?(?:live|here|out|public|generally available)\b|\bnow (?:in )?(?:public )?(?:beta|ga)\b|\blaunch(?:es|ed|ing)? (?:today|now|on product hunt)\b|\bproduct hunt\b|\bwaitlist\b|\bearly access\b", _F)),
    ("open_source", "Open-source release", re.compile(r"\bopen[- ]sourc(?:e|ed|ing)\b|\bgithub\.com/|\bmit licen[sc]e|\bapache 2", _F)),
    ("milestone", "Milestone or metric", re.compile(
        r"\b(just )?(hit|crossed|passed|reached|surpassed)\b.{0,30}(\$?\d[\d,.]*\s?(k|m|b|million|billion)?\b.{0,20}\b(users?|customers?|arr|mrr|revenue|downloads?|stars?|signups?|installs?|requests?))"
        r"|\$\d[\d,.]*\s?(k|m|million)?\s?(arr|mrr)\b|\b\d[\d,.]*\s?(k|m|million)?\+? (users|customers|downloads|github stars)\b", _F)),
    ("feature", "Feature release or update", re.compile(
        r"^\W*new( in| feature)?\b[: ]|\b(just |we )?(shipped|shipping|released|rolling out|rolled out)\b|\bnow (?:supports?|available|with|you can)\b|\bchangelog\b|\bv?\d+\.\d+(\.\d+)? (?:is )?(?:out|released|live)\b|\bupdate:", _F)),
    ("demo", "Demo or walkthrough", re.compile(r"\b(demo|walkthrough|here(?:'s| is) how (?:it works|we)|watch (?:it|this|how)|in action)\b", _F)),
]
LABELS = {k: label for k, label, _ in TYPES} | {"other": "General post"}


def classify(text: str | None) -> dict:
    t = (text or "").strip()
    for key, label, rx in TYPES:
        m = rx.search(t)
        if m:
            return {"type": key, "label": label, "matched": m.group(0).strip()[:60]}
    return {"type": "other", "label": LABELS["other"], "matched": None}
