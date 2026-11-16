"""What kind of company post a draft is, and the traits of a launch post that have a measured effect.

Types use the rules of xviral/text/launch_labels.py, validated by reading samples (research/tracks/launch/REPORT.md).
Traits use the same definitions as the evidence in xviral/data/launch_evidence.json (scripts/launch/build_table.py),
so a checklist line and its number describe the same thing.
"""

from __future__ import annotations

import re

import polars as pl

from xviral.text import launch_labels as L

LABELS = {"funding": "Funding announcement", "hiring": "Hiring", "launch": "Product launch", "open_source": "Open-source release",
          "feature": "Feature release or update", "milestone": "Milestone or metric", "demo": "Demo or walkthrough", "other": "General post"}

_PROBLEM = re.compile(r"\?\s*$|^(tired of|sick of|stop|ever wonder|what if|why|how|most|every|everyone|nobody|no one)\b|\b(problem|broken|pain|painful|struggle|hard to|too slow|slow|expensive|waste|wasting|"
                      r"frustrat|bottleneck|sucks|can't|cannot|don't|doesn't|shouldn't|isn't|never)\b")
_NAMES = re.compile(r"(?:[Ii]ntroducing|[Aa]nnouncing|[Mm]eet|[Ll]aunching|[Ll]aunch(?:ed)?|[Rr]eleas(?:e|ed|ing)|[Uu]nveiling|[Ss]ay hello to|[Oo]pen[- ]?sourc(?:ed|ing))\s+(?:the |our |my |a |an )?(?:new )?[@A-Z0-9]"
                    r"|\b[A-Z][a-z]+[A-Z]\w*|\b[A-Z][A-Za-z]*[- ]?\d+(?:\.\d+)*\b|@\w+")
_NUMBER = re.compile(r"(^|[\s$€£#(])\d")
_LINK = re.compile(r"https?://\S+")
_OWN_LINK = re.compile(r"https?://(www\.)?(x\.com|twitter\.com|t\.co)/")


def classify(text: str | None) -> dict:
    d = L.add_types(pl.DataFrame({"text": [text or ""]}))
    key = d["ptype_raw"][0]
    return {"type": key, "label": LABELS[key]}


def traits(text: str | None, media: list[dict] | None = None, kind: str = "original") -> dict:
    raw = text or ""
    nolink = _LINK.sub("", raw).strip()
    lines = [ln for ln in nolink.split("\n") if ln.strip()]
    first = lines[0] if lines else ""
    lc1 = first.lower().replace("’", "'")
    types = {m.get("type") for m in (media or [])}
    return {
        "names_product_first_line": bool(_NAMES.search(first)),
        "lines3": len(lines) >= 3,
        "video": bool(types & {"video", "animated_gif", "gif"}),
        "image": bool(types) and not bool(types & {"video", "animated_gif", "gif"}),
        "open_problem": bool(_PROBLEM.search(lc1)),
        "open_number": bool(_NUMBER.search(lc1)),
        "ext_link": any(not _OWN_LINK.match(u) for u in _LINK.findall(raw)),
        "is_quote": kind == "quote",
    }
