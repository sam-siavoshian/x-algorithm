"""Audience-relevance filter (DATA_PIPELINE.md 4.6).

Positive seed: every account in the 2025 tech scrape plus an optional handle
list. Negatives: a uniform sample of the public corpus. A logistic model on
cheap text embeddings scores every post; the decision is then made per
account so author histories stay intact:
  share of the account's recent originals that pass >= keep_whole  -> keep all
  share >= keep_partial                                            -> keep passing posts
  otherwise                                                        -> drop
Seed accounts are always kept whole.

Until a manual audit sets the threshold for the target precision, the default
threshold keeps 90% of posts from seed accounts held out of training.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import polars as pl

from xviral.config import RelevanceCfg
from xviral.embed.store import EmbeddingStore
from xviral.embed.text_small import make_embedder

log = logging.getLogger(__name__)


# The audience, described as topics. Posts are compared against these as retrieval queries; a post is
# on-topic when its best target match beats its best off-target match. This removes the part of a seed
# account's output that is not about tech (in election season, a lot), so the seed classifier learns
# the topic and not the accounts.
TARGET_TOPICS = [
    "a startup announcing a new product launch", "a launch video or demo of a new app or feature",
    "founders sharing lessons about building a company", "raising a seed round or venture capital funding",
    "Y Combinator, demo day and startup accelerators", "machine learning research papers and results",
    "large language models, AI agents and prompting", "software engineering, programming and code",
    "developer tools, APIs and SDKs", "open source software releases", "hiring engineers at a startup",
    "SaaS growth, marketing and customer acquisition", "product design and user experience",
    "AI infrastructure, GPUs, training and inference", "tech industry news about technology companies",
    "the San Francisco and Bay Area tech scene", "data engineering, databases and analytics",
    "robotics and hardware startups", "productivity tools, automation and no-code", "angel investing in startups",
]
OFF_TOPICS = [
    "US elections, candidates and voting", "partisan political commentary about Democrats and Republicans",
    "government policy debates on immigration, abortion and guns", "war, military conflict and geopolitics",
    "professional sports, teams and athletes", "celebrities, movies, TV shows and music",
    "religion and faith", "personal daily life, food, family and pets", "crime and police news",
    "conspiracy theories", "jokes and memes unrelated to technology", "memecoin and crypto price speculation",
    "weather and natural disasters",
]


def topic_margin(post_vecs: np.ndarray, embedder) -> np.ndarray:
    t = embedder.encode(TARGET_TOPICS, query=True)
    o = embedder.encode(OFF_TOPICS, query=True)
    return (post_vecs @ t.T).max(axis=1) - (post_vecs @ o.T).max(axis=1)


class TextClassifier:
    """Logistic regression on fixed text embeddings; binary or multiclass."""

    def __init__(self, C: float = 1.0):
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "TextClassifier":
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def classes(self):
        return self.model.classes_


def political_prethin(df: pl.DataFrame, cfg: RelevanceCfg) -> pl.DataFrame:
    """Cheap pre-thin of the USC corpus only: drop posts with >= N political terms."""
    pat = r"(?i)\b(?:" + "|".join(re.escape(w) for w in cfg.political_blocklist) + r")\b"
    hits = pl.col("text").fill_null("").str.count_matches(pat)
    keep = (pl.col("source") != "usc") | (hits < cfg.blocklist_min_hits)
    out = df.filter(keep)
    log.info("political pre-thin: %d -> %d", df.height, out.height)
    return out


def seed_authors(posts: pl.DataFrame, seed_handles: list[str] | None) -> pl.DataFrame:
    by_source = posts.filter(pl.col("source") == "user_scrape").select("author_id").unique()
    if seed_handles:
        hs = [h.lower().lstrip("@") for h in seed_handles]
        by_handle = posts.filter(pl.col("author_handle").str.to_lowercase().is_in(hs)).select("author_id").unique()
        by_source = pl.concat([by_source, by_handle]).unique()
    return by_source


def embed_texts(df: pl.DataFrame, store: EmbeddingStore, embedder, batch: int = 4096) -> None:
    todo = store.missing(df["post_id"].to_list())
    if not todo:
        return
    sub = df.filter(pl.col("post_id").is_in(todo)).select("post_id", "text")
    for i in range(0, sub.height, batch * 8):
        chunk = sub.slice(i, batch * 8)
        vecs = embedder.encode(chunk["text"].fill_null("").to_list(), batch_size=256)
        store.write(chunk["post_id"].to_numpy(), vecs)
        log.info("relevance embed %d/%d", min(i + batch * 8, sub.height), sub.height)


def score(df: pl.DataFrame, cfg: RelevanceCfg, work: Path, seed_handles: list[str] | None = None,
          device: str = "cuda", seed: int = 0) -> tuple[pl.DataFrame, dict]:
    """Returns df with rel_score, topic_margin, is_seed_author; plus a report dict."""
    store = EmbeddingStore(work / "rel_emb")
    embedder = make_embedder(cfg.embedder, device=device)
    embed_texts(df, store, embedder)
    Xa, _ = store.get(df["post_id"].to_list())
    Xa = np.nan_to_num(Xa)
    margin = topic_margin(Xa, embedder)
    seeds = seed_authors(df, seed_handles)
    df = df.with_columns(
        pl.col("author_id").is_in(seeds["author_id"].implode()).alias("is_seed_author"),
        pl.Series("topic_margin", margin, dtype=pl.Float64),
        pl.Series("_row", np.arange(df.height)),
    )
    on_topic = pl.col("topic_margin") > cfg.topic_margin_min
    # Hold out a fifth of the seed accounts and a slice of the corpus. The classifier never sees them,
    # so their scores show how an unseen relevant account, and an unseen ordinary post, score.
    calib_authors = seeds.sample(fraction=0.2, seed=seed, shuffle=True)["author_id"]
    df = df.with_columns((pl.col("is_seed_author") & pl.col("author_id").is_in(calib_authors.implode())).alias("_calib"))
    pool = df.filter(~pl.col("is_seed_author"))
    held_neg = pool.sample(fraction=cfg.heldout_negative_fraction, seed=seed)
    neg = pool.join(held_neg.select("post_id"), on="post_id", how="anti")
    neg = neg.sample(n=min(cfg.negatives_sample, neg.height), seed=seed)
    pos = df.filter(pl.col("is_seed_author") & ~pl.col("_calib") & on_topic)
    calib = df.filter(pl.col("_calib") & on_topic)
    if pos.height < 50 or calib.height < 20:
        raise ValueError(f"relevance: {pos.height} train / {calib.height} held-out on-topic seed posts; need the tech scrape or a seed handle list")
    X = np.vstack([Xa[pos["_row"].to_numpy()], Xa[neg["_row"].to_numpy()]])
    y = np.r_[np.ones(pos.height), np.zeros(neg.height)]
    clf = TextClassifier(C=0.1).fit(X, y)
    s = clf.proba(Xa)[:, 1]
    df = df.with_columns(pl.Series("rel_score", s, dtype=pl.Float64))
    calib_scores = s[calib["_row"].to_numpy()]
    thr = cfg.threshold if cfg.threshold is not None else float(np.quantile(calib_scores, 0.10))
    from sklearn.metrics import roc_auc_score

    hn = s[held_neg["_row"].to_numpy()]
    auc = float(roc_auc_score(np.r_[np.ones(len(calib_scores)), np.zeros(len(hn))], np.r_[calib_scores, hn])) if len(hn) else float("nan")
    keep = (s >= thr) & (margin > cfg.topic_margin_min)
    report = {
        "seed_authors": seeds.height, "train_seed_posts": pos.height,
        "seed_posts_off_topic_excluded": int(df.filter(pl.col("is_seed_author") & ~on_topic).height),
        "heldout_seed_authors": calib_authors.len(), "heldout_seed_posts": calib.height, "negatives": neg.height,
        "threshold": thr, "threshold_source": "config" if cfg.threshold is not None else "heldout_seed_recall_0.90",
        "heldout_auc_vs_corpus": auc,
        "heldout_corpus_pass_rate": float(keep[held_neg["_row"].to_numpy()].mean()) if held_neg.height else float("nan"),
        "corpus_on_topic_rate": float((margin[~df["is_seed_author"].to_numpy()] > cfg.topic_margin_min).mean()),
    }
    log.info("relevance: %s", report)
    return df.drop("_calib", "_row"), report


def decide(df: pl.DataFrame, cfg: RelevanceCfg, threshold: float) -> pl.DataFrame:
    """Account-level keep decision, then per-account and per-duplicate-cluster caps."""
    d = df.with_columns(((pl.col("rel_score") >= threshold) & (pl.col("topic_margin") > cfg.topic_margin_min)).alias("rel_pass"))
    recent = (
        d.sort("created_at_ms", descending=True)
        .group_by("author_id")
        .agg(pl.col("rel_pass").head(cfg.account_window).mean().alias("acct_pass_share"), pl.col("is_seed_author").first())
    )
    d = d.join(recent, on=["author_id", "is_seed_author"], how="left").with_columns(
        pl.when(pl.col("is_seed_author") | (pl.col("acct_pass_share") >= cfg.account_keep_whole)).then(pl.lit("whole"))
        .when(pl.col("acct_pass_share") >= cfg.account_keep_partial).then(pl.lit("partial"))
        .otherwise(pl.lit("drop")).alias("acct_decision")
    )
    kept = d.filter((pl.col("acct_decision") == "whole") | ((pl.col("acct_decision") == "partial") & pl.col("rel_pass")))
    kept = kept.with_columns(pl.col("created_at_ms").rank("ordinal", descending=True).over("author_id").alias("_r")).filter(
        pl.col("_r") <= cfg.max_rows_per_account).drop("_r")
    if "near_dup_cluster_id" in kept.columns:
        kept = kept.with_columns(pl.col("created_at_ms").rank("ordinal").over("near_dup_cluster_id").alias("_c")).filter(
            pl.col("near_dup_cluster_id").is_null() | (pl.col("_c") <= cfg.max_rows_per_dup_cluster)).drop("_c")
    acct = d.select("author_id", "acct_decision").unique()
    counts = dict(acct.group_by("acct_decision").len().iter_rows())
    log.info("relevance decide: %d -> %d rows; accounts %s", df.height, kept.height, counts)
    return kept


def audit_sample(scored: pl.DataFrame, kept_ids: pl.Series, n: int, threshold: float, seed: int = 0) -> pl.DataFrame:
    """Rows for manual review: random kept, random dropped, and a band around the threshold."""
    k = scored.filter(pl.col("post_id").is_in(kept_ids.implode()))
    dr = scored.filter(~pl.col("post_id").is_in(kept_ids.implode()))
    band = scored.with_columns((pl.col("rel_score") - threshold).abs().alias("_d")).sort("_d").head(n)
    parts = [
        k.sample(n=min(n, k.height), seed=seed).with_columns(pl.lit("kept").alias("audit_bucket")),
        dr.sample(n=min(n, dr.height), seed=seed).with_columns(pl.lit("dropped").alias("audit_bucket")),
        band.drop("_d").with_columns(pl.lit("boundary").alias("audit_bucket")),
    ]
    cols = ["audit_bucket", "post_id", "author_handle", "rel_score", "topic_margin", "text", "source"]
    return pl.concat([p.select(cols) for p in parts]).with_columns(pl.lit(None, dtype=pl.Boolean).alias("relevant"))


def threshold_from_audit(audit: pl.DataFrame, target_precision: float) -> float | None:
    """Lowest threshold whose labelled precision (rows at or above it) meets the target."""
    a = audit.filter(pl.col("relevant").is_not_null()).sort("rel_score", descending=True)
    if a.height < 50:
        return None
    rel = a["relevant"].cast(pl.Float64).to_numpy()
    prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
    ok = np.flatnonzero(prec >= target_precision)
    return float(a["rel_score"][int(ok[-1])]) if ok.size else None
