"""Score drafts for an account: the serving path from raw posts to the product report.

Takes the account's post history (unified schema, with outcomes) and one or more drafts, runs the
same feature functions the training pipeline runs, applies the transforms stored in the model's
checkpoint, and scores the drafts against the account's own history.

Drafts given together are treated as alternatives for the same slot: they share the history and
are scored in one pass under candidate isolation, so each draft's score is independent of the
others and the scores are directly comparable.

Not yet wired in here: the pre-flight rule check, retrieved neighbours (needs a corpus index), and
media embeddings. A model trained with neighbours scores without them as it does for a post that
has none.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import torch

from xviral import authors, dedup
from xviral.features import audience, content, derive, suppression, text_structure
from xviral.model import episodes, features
from xviral.model import train as tr
from xviral.model.predict import Predictor
from xviral.schema import MEDIA_STRUCT, URL_STRUCT, conform
from xviral.util.core import stable_hash64


def feature_frame(posts: pl.DataFrame, snapshots: pl.DataFrame | None = None, account: dict | None = None) -> pl.DataFrame:
    """The pipeline's author and feature stages for a small set of posts, in memory.

    snapshots: extra (author_id, snapshot_at_ms, followers) rows, e.g. the account's follower count now.
    account: fallbacks for author fields the posts do not carry (following, created_at_ms, verified).
    Corpus-level columns (near-duplicate clusters, graph features, neighbours) are not produced here.
    """
    account = account or {}
    snaps = posts.select("author_id", pl.col("snapshot_at_ms").fill_null(pl.col("created_at_ms")).alias("snapshot_at_ms"),
                         pl.col("author_followers").alias("followers")).filter(pl.col("followers").is_not_null())
    if snapshots is not None and snapshots.height:
        snaps = pl.concat([snaps, snapshots.select("author_id", "snapshot_at_ms", "followers")], how="vertical_relaxed")
    df = authors.followers_at_post(posts, snaps.unique().sort("author_id", "snapshot_at_ms"))
    df = df.join(authors.trailing_features(df), on="post_id", how="left")
    df = df.join(dedup.author_copypasta(df).select("post_id", "is_self_repeat"), on="post_id", how="left")
    fb = lambda col, key, dt: pl.coalesce(pl.col(col), pl.lit(account.get(key), dtype=dt))   # noqa: E731
    df = derive.author_columns(df.with_columns(
        fb("author_following", "following", pl.Int64), fb("author_created_at_ms", "created_at_ms", pl.Int64),
        fb("author_verified", "verified", pl.Boolean).alias("verified")))
    df = df.join(content.content_features(df), on="post_id", how="left")
    df = df.join(text_structure.admitted(df), on="post_id", how="left")
    real = df.filter(pl.col("source") != "draft")
    zones = audience.estimate_audience_zone(real) if real.height else pl.DataFrame(
        schema={"author_id": pl.Int64, "audience_zone": pl.String, "audience_utc_offset": pl.Float64})
    df = df.join(audience.audience_time_features(df, zones), on="post_id", how="left")
    return df.join(suppression.url_features(df, set()), on="post_id", how="left")


class DraftScorer:
    def __init__(self, checkpoint: str | Path, embed_fn: Callable[[list[str]], np.ndarray], device: str = "cpu",
                 sid_codebook: str | Path | None = None):
        """embed_fn: rendered post texts -> [n, emb_dim] array, from the same encoder the model was trained with."""
        self.model, self.tcfg, ck = tr.load(checkpoint, device)
        self.meta, self.temperature = ck["table_meta"], ck.get("temperature") or {}
        self.embed_fn, self.device, self.sid_codebook = embed_fn, device, sid_codebook

    def _draft_row(self, d: dict, i: int, account: dict, when_ms: int) -> pl.DataFrame:
        row = {
            "post_id": -stable_hash64("draft", account.get("handle"), when_ms, i, d.get("text")),
            "author_id": account["author_id"], "author_handle": account.get("handle"),
            "author_display_name": account.get("display_name"), "created_at_ms": when_ms,
            "text": d.get("text"), "kind": d.get("kind", "original"), "is_reply": False,
            "media": d.get("media", []), "urls": d.get("urls", []), "quoted_text": d.get("quoted_text"),
            "author_followers": account.get("followers"), "author_following": account.get("following"),
            "author_created_at_ms": account.get("created_at_ms"), "author_verified": account.get("verified"),
            "source": "draft", "synthetic_id": True, "views_state": "absent",
        }
        return conform(pl.DataFrame([row], schema_overrides={"media": pl.List(MEDIA_STRUCT), "urls": pl.List(URL_STRUCT)}, infer_schema_length=None))

    def featurize(self, history: pl.DataFrame, drafts: list[dict], account: dict, when_ms: int | None = None) -> tuple[features.FeatureTable, pl.DataFrame]:
        """The table holds the history followed by the drafts. Each draft is featurised alone against the
        history, as the account's next post, so no draft's features see another draft."""
        when_ms = when_ms or int(time.time() * 1000)
        account = {"author_id": int(history["author_id"][0]) if history.height else -1, **account}
        hist = conform(history).filter(pl.col("created_at_ms") < when_ms).with_columns(pl.lit(account["author_id"]).alias("author_id"))
        now = pl.DataFrame({"author_id": [account["author_id"]], "snapshot_at_ms": [when_ms], "followers": [account.get("followers")]},
                           schema={"author_id": pl.Int64, "snapshot_at_ms": pl.Int64, "followers": pl.Int64}).filter(pl.col("followers").is_not_null())
        parts: list[pl.DataFrame] = []
        for i, d in enumerate(drafts):
            f = feature_frame(pl.concat([hist, self._draft_row(d, i, account, when_ms)], how="vertical_relaxed"), now, account)
            if not parts:
                parts.append(f.filter(pl.col("source") != "draft"))
            parts.append(f.filter(pl.col("source") == "draft"))
        posts = pl.concat(parts, how="vertical_relaxed")
        emb = np.asarray(self.embed_fn(features.render_texts(posts))) if self.meta["emb_dim"] else None
        sid = None
        if self.sid_codebook and emb is not None and self.meta["sid_levels"]:
            from xviral.embed import sid as sidmod

            sid = sidmod.phoenix_codes(sidmod.assign(emb, self.sid_codebook))
        return features.transform(posts, self.meta, emb=emb, sid=sid, device=self.device)

    @torch.no_grad()
    def score(self, history: pl.DataFrame, drafts: list[dict], account: dict, when_ms: int | None = None,
              mutual_share: float = 0.0) -> list[dict]:
        """One report per draft, in the order given. account: handle, followers, following, created_at_ms, verified."""
        when_ms = when_ms or int(time.time() * 1000)
        table, rows = self.featurize(history, drafts, account, when_ms)
        n = len(drafts)
        first = table.n - n                        # drafts are the newest rows of the single author
        by_id = {r["post_id"]: r for r in Predictor(self.model, table, self.tcfg, self.temperature).report_joint(first, n, mutual_share=mutual_share)}
        acct = {"author_id": int(rows["author_id"][0]), **account}
        reports = []
        for i, d in enumerate(drafts):             # table order among drafts is by ID; return them in the order given
            r = by_id[int(self._draft_row(d, i, acct, when_ms)["post_id"][0])]
            r["text"] = d.get("text")
            r["history_posts_used"] = int(min(first, self.model.cfg.max_history))
            reports.append(r)
        return reports
