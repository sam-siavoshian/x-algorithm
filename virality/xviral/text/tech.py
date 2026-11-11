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


# Unambiguous vocabulary only. TECH_POST is built for recall on single posts and matches everyday words
# ("launch a strike", "research paper", "co-founder of a think tank"); measured on the 2026 view corpus it
# let news, politics and history accounts through an account-level share rule. This pattern is the one
# account selection uses.
TECH_STRONG = (
    r"(?i)\b(startups?|co-?founders?|yc|y combinator|ycombinator|seed round|series [abc] |pre-?seed|product hunt|waitlist|"
    r"open[- ]sourced?|github|apis?|sdk|saas|devtools?|developers?|devs|software engineer(?:s|ing)?|programming|coding|codebase|"
    r"typescript|javascript|kubernetes|postgres|llms?|gpt-?\d\S*|chatgpt|openai|anthropic|claude|deepmind|hugging ?face|"
    r"fine-?tun\w+|ai agents?|agentic|machine learning|deep learning|neural net\w*|diffusion models?|arxiv|gpus?|inference|"
    r"vercel|supabase|vibe ?cod\w+|mcp|open ?weights?|model weights|context window|prompt\w*|evals?|pull request|compiler|"
    r"frontend|backend|full-?stack|b2b|arr|mrr|pmf|founder mode|claude code|codex|copilot|llama\S*|mistral|midjourney|"
    r"stable diffusion|robotics|humanoid)\b"
)
# Vocabulary of neighbouring audiences that share words with tech (AI tokens, "agents", "launch") but are not the
# target: crypto and retail trading, and day-to-day politics. An account is off-audience when this share exceeds
# its tech share.
OFF_AUDIENCE = (
    r"(?i)(\$[A-Z]{2,6}\b|\b(?:airdrop|memecoin|altcoins?|presale|tokenomics|staking|defi|nft|web3|solana|ethereum|bitcoin|btc|eth|"
    r"hodl|bullish|bearish|pump|forex|price target|earnings call|options flow|calls? and puts?|stock market|nasdaq|s&p ?500|"
    r"trump|biden|maga|democrats?|republicans?|congress|senate|election|immigration|gaza|israel|ukraine|putin|nato|"
    r"woke|liberals?|conservatives?)\b)"
)


def is_tech_post(col: str = "text", strong: bool = False) -> pl.Expr:
    return pl.col(col).fill_null("").str.contains(TECH_STRONG if strong else TECH_POST)


def is_off_audience_post(col: str = "text") -> pl.Expr:
    return pl.col(col).fill_null("").str.contains(OFF_AUDIENCE)


def is_tech_bio(col: str = "author_bio") -> pl.Expr:
    return pl.col(col).fill_null("").str.contains(TECH_BIO)
