"""Cheap, recall-oriented tech-audience prefilters for very large general corpora.

These only cut a 75M-row corpus down to something the embedding-based relevance stage
(xviral.relevance) can afford; that stage makes the real decision. Measured precision of the
post regex on 2026 posts is roughly 60 to 70% (research/08_public_datasets.md).
"""

from __future__ import annotations

import polars as pl

TECH_POST = (
    r"(?i)\b(startup|founder|co-?founder|yc|y combinator|ycombinator|seed round|series [abc]\b|pre-?seed|venture|vc\b|"
    r"launch(ed|ing)?|shipped|shipping|product hunt|waitlist|open[- ]source|github|repo\b|api\b|sdk\b|saas|devtools?|"
    r"developer|software engineer|programming|coding|typescript|javascript|python|rust\b|golang|kubernetes|postgres|"
    r"llm|gpt|chatgpt|openai|anthropic|claude|gemini|mistral|llama|hugging ?face|fine-?tun|rag\b|agents?\b|ai agent|"
    r"machine learning|deep learning|neural net|transformer|diffusion|benchmark|arxiv|paper\b|nvidia|gpu|inference|"
    r"cursor\b|vercel|supabase|stripe|notion|figma|vibe ?coding|mcp\b|robotics|self-driving|autonomous)\b"
)
TECH_BIO = (
    r"(?i)\b(founder|co-?founder|ceo|cto|building|builder|engineer|developer|dev\b|programmer|hacker|researcher|scientist|"
    r"machine learning|ml\b|ai\b|llm|startup|yc\b|y combinator|investor|vc\b|venture|angel|product manager|pm @|"
    r"designer|open[- ]source|software|saas|indie ?hacker|maker|data scien|phd)\b"
)


def is_tech_post(col: str = "text") -> pl.Expr:
    return pl.col(col).fill_null("").str.contains(TECH_POST)


def is_tech_bio(col: str = "author_bio") -> pl.Expr:
    return pl.col(col).fill_null("").str.contains(TECH_BIO)
