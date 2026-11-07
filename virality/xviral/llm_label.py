"""LLM labeler for the spam-policy classifier and the relevance audit.

X's Grox prompts are withheld from the repository, but the policy taxonomy is
not (grox/flows/ptos/state.py). This module asks an open model, served on our
own GPU through any OpenAI-compatible endpoint (vLLM, SGLang), to assign one
spam policy type per post. The labels train a small classifier on cheap text
embeddings (xviral.relevance.TextClassifier), which then scores the corpus.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import requests

from xviral.features.suppression import SPAM_TYPES

log = logging.getLogger(__name__)

DEFINITIONS = {
    "NoViolation": "An ordinary post. Promotion, launches, announcements, opinions and calls to try a product are fine.",
    "SpamEngagementBaiting": "Asks for engagement as the point of the post: 'like if', 'retweet if', 'reply with', 'follow for', 'comment X to get', tag-a-friend chains.",
    "SpamEngagementFarming": "Content built to harvest engagement with no substance: giveaways requiring follows/reposts, follow trains, 'drop your handle' threads, recycled viral bait.",
    "SpamCardManipulation": "Link cards whose preview misrepresents the destination, or text that disguises where a link goes.",
    "SpamEngagementTrading": "Offers to exchange follows, likes or reposts.",
    "SpamFinancialScam": "Too-good-to-be-true money offers, crypto doubling, fake investment returns, impersonated giveaways.",
    "SpamHashTagAbuse": "Many unrelated or trending hashtags stuffed in to gain reach.",
    "SpamScamLinks": "Links to phishing, malware, fake login pages or deceptive downloads.",
    "SpamBotProduced": "Text that reads as automated: templated, repetitive, nonsensical, or obviously machine-generated filler.",
    "SpamPlatformManipulation": "Coordinated or fake activity, purchased engagement, or attempts to game ranking.",
    "SpamMentionAbuse": "Mentions many accounts that have no relation to the content, to force notifications.",
    "SpamManipulatedMedia": "Media edited to deceive about real events or people.",
}

PROMPT = """You label X posts for a spam-policy classifier.
Choose exactly one category for the post below.

Categories:
{cats}

Post:
<<<
{text}
>>>

Answer with JSON only: {{"category": "<one of the names above>"}}"""


def build_prompt(text: str) -> str:
    cats = "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS.items())
    return PROMPT.format(cats=cats, text=(text or "")[:2000])


def parse_answer(content: str) -> str | None:
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        return None
    try:
        cat = json.loads(m.group(0)).get("category")
    except (ValueError, AttributeError):
        return None
    return cat if cat in SPAM_TYPES else None


def label(texts: list[str], endpoint: str, model: str, workers: int = 16, timeout: float = 60.0) -> list[str | None]:
    """endpoint: base URL of an OpenAI-compatible server, e.g. http://localhost:8000/v1"""

    def one(t: str) -> str | None:
        body = {"model": model, "messages": [{"role": "user", "content": build_prompt(t)}], "temperature": 0.0, "max_tokens": 32}
        for _ in range(3):
            try:
                r = requests.post(f"{endpoint.rstrip('/')}/chat/completions", json=body, timeout=timeout)
                r.raise_for_status()
                return parse_answer(r.json()["choices"][0]["message"]["content"])
            except (requests.RequestException, KeyError, IndexError):
                continue
        return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, texts))
