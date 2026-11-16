"""Rules that read a profile bio (and display name) as a founder, a company account, or neither.

Used to build the startup account list (scripts/data_track/build_startup_accounts.py) from sources that carry bios.
The rules are deliberately narrow: a bio is evidence only when it states a role at a named thing ("founder of X",
"CEO @x", "building x.com") or speaks as an organisation about a product. `tech` restricts either to the product,
software and AI world, which is the audience of this project; crypto, coaching and
agency bios are excluded.
"""
from __future__ import annotations

import polars as pl

_ROLE = r"(co[- ]?founder|founder|founding (engineer|team|member)|ceo|cto|chief executive|chief technology officer|creator|maker|indie ?hacker|solo ?founder|bootstrapp\w+|solopreneur)"
# A role tied to something: "founder of", "founder @x", "ceo at", "founder & ceo, acme", "building @x", "making x.com".
FOUNDER = (
    r"(?i)(\b(co[- ]?founder|founder|ceo|cto)\b[^.|•\n]{0,40}?(\bof\b|\bat\b|@\w|,\s|\s[-–—:]\s|\s?/\s?|\s&\s|\sand\s)"
    r"|\b(founded|co[- ]?founded|started|bootstrapped|bootstrapping)\s+@?\w"
    r"|\b(co[- ]?creator|creator|maker|inventor|author)\s+(of\s+)?(@\w+|https?://)"
    r"|\b(co[- ]?founder|founder|ceo|cto)\s+(of\s+|at\s+)?https?://"
    r"|\b(building|built|making|made|creator of|maker of|created|hacking on|working on)\s+(@\w+|\w+\.(com|ai|io|dev|app|co|so|xyz|sh|tech|gg)\b)"
    r"|\b(indie ?hacker|solo ?founder|solopreneur|indie (maker|dev|developer)|serial entrepreneur|\d+x founder|yc [sw]\d\d|\(yc [sw]\d\d\)|techstars)\b)"
)
# The bio speaks as an organisation about a product or service.
COMPANY = (
    r"(?i)(\b(we|we're|we are|our)\b[^.]{0,60}\b(build|help|make|provide|offer|platform|product|mission|team|customers|users|api|app|software|tools?)\b"
    r"|\b(the|a|an|your|#1|leading|world's|first|open[- ]source|ai[- ]powered|all[- ]in[- ]one|fastest|easiest|simplest|best)\b[^.]{0,40}\b(platform|app|tool|toolkit|software|api|sdk|engine|database|framework|marketplace|assistant|copilot|browser|extension|plugin|startup|company|studio|lab|labs|network|protocol|service|solution|suite|infrastructure|cloud|library|ide|editor|os)\b (for|to|that|built|of)\b"
    r"|\bofficial (twitter |x )?(account|page|handle|feed|home)\b|\b(support|help|status|contact|questions|sales|press|inquiries|hello)(\s*:|\s+(at|via|@)\b|@)"
    r"|\b(join (us|our)|sign up|get started|try it|download (now|the app|free)|book a demo|request a demo|now (hiring|available)|backed by|a \w+ company|part of @\w+|by @\w+|from the (team|makers|creators))\b"
    r"|\b(platform|app|tool|api|sdk|software|product|saas) (for|that|to)\b)"
)
# First-person-singular words: a person's bio, not an organisation's.
PERSON = r"(?i)(\b(i|i'm|i am|my|me|mine|he/him|she/her|they/them|dad|mom|father|mother|husband|wife|views (are )?my own|opinions (are )?(my )?own|personal account|tweets are my own|phd|alum|alumni|ex[- ]@?\w+|prev\.?(iously)?|formerly|born|human|person|guy|girl|nerd|geek|lover|fan|enthusiast|student|professor|engineer @|working (at|on)|i (build|write|make|work))\b)"
TECH = (
    r"(?i)\b(ai|a\.i\.|ml|llms?|gpt|genai|saas|apis?|sdk|software|apps?|startups?|start-ups?|developers?|devs?|devtools?|dev tools|engineer\w*|open[- ]source|"
    r"machine learning|deep learning|robotics|automation|no-?code|low-?code|fintech|cyber\w*|database|analytics|code|coding|programming|github|"
    r"platform|products?|tools?|design|data|ios|android|chrome extension|browser|plugin|hardware|semiconductors?|compute|gpus?|agents?|yc|y combinator|techstars|bootstrapp\w+|indie ?hacker|indie (maker|dev)|cloud|b2b|product hunt)\b"
)
# Bios that use founder words but are not the target: coaches, agencies, crypto promotion, churches, charities, politics.
NOT_STARTUP = r"(?i)\b(crypto\w*|defi|web3|blockchain|token|tokens|coin|dao|metaverse|nfts?|play[- ]to[- ]earn|casino|betting|coach|coaching|speaker|author of|bestselling|real estate|realtor|ministry|pastor|church|foundation|non-?profit|charity|campaign|candidate|congress|senator|podcast host|life coach|motivational|mlm|forex|trader|signals|airdrop|nft|memecoin|giveaway|onlyfans|18\+)\b"


def bio_flags(bio: str = "author_bio", name: str = "author_display_name") -> list[pl.Expr]:
    """Boolean columns bio_founder, bio_company, bio_person, bio_tech, bio_excluded."""
    b = pl.col(bio).fill_null("")
    return [b.str.contains(FOUNDER).alias("bio_founder"), b.str.contains(COMPANY).alias("bio_company"), b.str.contains(PERSON).alias("bio_person"),
            b.str.contains(TECH).alias("bio_tech"), b.str.contains(NOT_STARTUP).alias("bio_excluded")]


def bio_kind() -> pl.Expr:
    """founder | company | none from the bio_* flags. A founder bio wins over a company bio (founders describe their product)."""
    ok = pl.col("bio_tech") & ~pl.col("bio_excluded")
    return (pl.when(ok & pl.col("bio_founder")).then(pl.lit("founder"))
            .when(ok & pl.col("bio_company") & ~pl.col("bio_person")).then(pl.lit("company")).otherwise(pl.lit("none")))
