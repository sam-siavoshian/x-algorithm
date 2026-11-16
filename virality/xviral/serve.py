"""Score drafts for an account: the serving path from raw posts to the product report.

Takes the account's post history (unified schema, with outcomes) and one or more drafts, runs the
same feature functions the training pipeline runs, applies the transforms stored in the model's
checkpoint, and scores the drafts against the account's own history.

Drafts given together are treated as alternatives for the same slot: they share the history and
are scored in one pass under candidate isolation, so each draft's score is independent of the
others and the scores are directly comparable.

Each report also carries the deterministic distribution gates read off X's code (xalgo/gates.py).
Not yet wired in here: the classifier-based pre-flight checks, retrieved neighbours (needs a corpus
index), and media embeddings. A model trained with neighbours scores without them as it does for a post that
has none.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import torch

from xviral import authors, dedup
from xviral.features import audience, content, derive, post_type, suppression, text_structure
from xviral.model import episodes, features
from xviral.model import train as tr
from xviral.model.predict import Predictor
from xviral.schema import MEDIA_STRUCT, URL_STRUCT, conform
from xviral.util.core import stable_hash64
from xviral.xalgo import gates


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
    df = df.join(suppression.url_features(df, set()), on="post_id", how="left")
    return with_post_type(df)


def with_post_type(df: pl.DataFrame) -> pl.DataFrame:
    """Post type, launch traits and the account's trailing record by type (features/post_type.py). Drafts in the
    frame never count as history, so this is also correct for a frame holding several alternative drafts."""
    df = df.drop([c for c in post_type.COLUMNS if c in df.columns])
    return df.join(post_type.post_type_features(df).drop("pt_type"), on="post_id", how="left")


_KEEP = ("post_id", "author_id", "author_handle", "author_display_name", "created_at_ms", "source", "kind", "text", "text_note", "quoted_text",
         "in_audience", "w_row", *features.COUNT_COLS, *[c for c, _ in features.NUM_SPEC], *[c for c, _ in features.CAT_SPEC])


def compact(posts: pl.DataFrame) -> pl.DataFrame:
    """The columns the feature table reads, copied into buffers of their own. A frame sliced out of a large
    corpus keeps that corpus's string buffers alive, and pickling it writes them all."""
    f = posts.select([c for c in dict.fromkeys(_KEEP) if c in posts.columns])
    return pl.DataFrame(f.to_dict(as_series=False), schema=f.schema)


class DraftScorer:
    def __init__(self, checkpoint: str | Path | list, embed_fn: Callable[[list[str]], np.ndarray], device: str = "cpu",
                 sid_codebook: str | Path | None = None, type_correction: str | Path | dict | None = None):
        """embed_fn: rendered post texts -> [n, emb_dim] array, from the same encoder the model was trained with.
        type_correction: the JSON written by scripts/posttype/fit_correction.py (or its dict): per post-type column a
        shift in log10 likes, fitted on the model's errors on own announcements before the test cutoff. It moves the
        likes forecast of a draft the rules type as an own announcement and leaves every other draft unchanged."""
        if isinstance(type_correction, (str, Path)):
            import json

            type_correction = json.loads(Path(type_correction).read_text())
        self.type_correction = (type_correction or {}).get("weights_log10", type_correction) or {}
        if isinstance(checkpoint, (list, tuple)):       # several checkpoints: scored as an ensemble (model/ensemble.py)
            from xviral.model import ensemble

            self.model, self.tcfg, ck = ensemble.load(list(checkpoint), device)
            self.temperature = {}
        else:
            self.model, self.tcfg, ck = tr.load(checkpoint, device)
            self.temperature = ck.get("temperature") or {}
        self.meta = ck["table_meta"]
        self._embed_raw, self.device, self.sid_codebook = embed_fn, device, sid_codebook
        self._emb_cache: dict[str, np.ndarray] = {}
        self.emb_cache_limit = 200_000
        self.feature_cache: dict | None = None
        # fast=True is the product's request path (research/tracks/product/REPORT.md). Same numbers up to floating-point
        # noise, less work: only the history rows the model can read are embedded, all drafts are featurised in one pass,
        # and each ensemble member runs once. False keeps the original path, which the equality test compares against.
        self.fast = False
        self.timings: dict[str, float] = {}

    def _tick(self, name: str, t0: float) -> float:
        now = time.perf_counter()
        self.timings[name] = self.timings.get(name, 0.0) + now - t0
        return now

    def embed_fn(self, texts: list[str]) -> np.ndarray:
        """The encoder with a cache by rendered text: an account's history is embedded once across calls."""
        new = [t for t in dict.fromkeys(texts) if t not in self._emb_cache]
        if new:
            for t, v in zip(new, np.asarray(self._embed_raw(new))):
                self._emb_cache[t] = v
        if len(self._emb_cache) > self.emb_cache_limit:
            self._emb_cache = {t: self._emb_cache[t] for t in texts}
        return np.stack([self._emb_cache[t] for t in texts])

    def preencode(self, posts: pl.DataFrame, chunk: int = 2048, log=None) -> int:
        """Embed many posts ahead of scoring, sorted by length. Scoring pair by pair encodes each account's history as
        one small batch padded to its longest post, which on a CPU costs several times more encoder time."""
        texts = sorted({t for t in features.render_texts(conform(posts)) if t not in self._emb_cache}, key=len)
        t0 = time.time()
        for k in range(0, len(texts), chunk):
            part = texts[k:k + chunk]
            for t, v in zip(part, np.asarray(self._embed_raw(part))):
                self._emb_cache[t] = v
            if log is not None and (k // chunk) % 5 == 0:
                log.info("encoded %d/%d texts ahead, %.0fs", k + len(part), len(texts), time.time() - t0)
        return len(texts)

    def prefill_embeddings(self, table_dir: str | Path, post_ids) -> int:
        """Seed the embedding cache with a feature table's stored embeddings for `post_ids` (same encoder, keyed by
        the rendered text), so scoring many posts of known accounts does not embed their histories again."""
        d = Path(table_dir)
        ids = np.load(d / "post_id.npy")
        idx = np.flatnonzero(np.isin(ids, np.asarray(list(post_ids), dtype=np.int64)) & np.load(d / "has_emb.npy"))
        texts = pl.read_parquet(d / "texts.parquet")["text"].gather(idx).to_list()
        emb = np.asarray(np.load(d / "emb.npy", mmap_mode="r")[idx], dtype=np.float32)
        self._emb_cache.update(zip(texts, emb))
        return len(idx)

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

    def _feature_rows(self, hist: pl.DataFrame, drafts: list[dict], account: dict, when_ms: int, now: pl.DataFrame) -> pl.DataFrame:
        """History rows, then one row per draft. Each draft is featurised alone against the history."""
        parts: list[pl.DataFrame] = []
        for i, d in enumerate(drafts):
            f = feature_frame(pl.concat([hist, self._draft_row(d, i, account, when_ms)], how="vertical_relaxed"), now, account)
            if not parts:
                parts.append(f.filter(pl.col("source") != "draft"))
            parts.append(f.filter(pl.col("source") == "draft"))
        return pl.concat(parts, how="vertical_relaxed")

    def _feature_rows_batched(self, hist: pl.DataFrame, drafts: list[dict], account: dict, when_ms: int, now: pl.DataFrame) -> pl.DataFrame:
        """The same rows from one pass over the feature functions. Every feature is computed per author, so each
        draft after the first gets a copy of the history under an author ID of its own: no draft sees another, and
        the history is not recomputed once per draft through the whole chain of feature functions."""
        aid, base = account["author_id"], -(1 << 61)
        frames, snaps, clones = [hist, self._draft_row(drafts[0], 0, account, when_ms)], [now], []
        for i, d in enumerate(drafts[1:], start=1):
            clone = base - i
            clones.append(clone)
            frames.append(hist.with_columns(pl.lit(clone, dtype=pl.Int64).alias("author_id"),
                                            (pl.lit(base - i * (1 << 32), dtype=pl.Int64) - pl.int_range(pl.len(), dtype=pl.Int64)).alias("post_id")))
            frames.append(self._draft_row(d, i, account, when_ms).with_columns(pl.lit(clone, dtype=pl.Int64).alias("author_id")))
            snaps.append(now.with_columns(pl.lit(clone, dtype=pl.Int64).alias("author_id")))
        allp = pl.concat(frames, how="vertical_relaxed")
        if aid in clones or allp["post_id"].n_unique() != allp.height:      # an ID collision: not worth reasoning about, take the plain path
            return self._feature_rows(hist, drafts, account, when_ms, now)
        f = feature_frame(allp, pl.concat(snaps), account)
        own = f.filter((pl.col("author_id") == aid) & (pl.col("source") != "draft"))
        dr = f.filter(pl.col("source") == "draft").with_columns(pl.lit(aid, dtype=pl.Int64).alias("author_id"))
        ids = [int(self._draft_row(d, i, account, when_ms)["post_id"][0]) for i, d in enumerate(drafts)]
        dr = dr.join(pl.DataFrame({"post_id": ids, "_o": list(range(len(ids)))}, schema={"post_id": pl.Int64, "_o": pl.Int64}), on="post_id", how="left").sort("_o").drop("_o")
        return pl.concat([own, dr], how="vertical_relaxed")

    def _rows_the_model_reads(self, posts: pl.DataFrame, n_drafts: int) -> np.ndarray:
        """Indices of the rows whose embedding and semantic ID can reach the model: the drafts and the history rows an
        episode can hold (the newest `max_history`; ties on the timestamp are all kept). Older history enters only
        through the trailing features, which do not use embeddings."""
        n_hist, H = posts.height - n_drafts, self.model.cfg.max_history
        t = posts["created_at_ms"].to_numpy()[:n_hist]
        keep = np.arange(n_hist) if n_hist <= H else np.nonzero(t >= np.sort(t)[-H])[0]
        return np.concatenate([keep, np.arange(n_hist, posts.height)]).astype(np.int64)

    def featurize(self, history: pl.DataFrame, drafts: list[dict], account: dict, when_ms: int | None = None) -> tuple[features.FeatureTable, pl.DataFrame]:
        """The table holds the history followed by the drafts. Each draft is featurised alone against the
        history, as the account's next post, so no draft's features see another draft."""
        when_ms = when_ms or int(time.time() * 1000)
        account = {"author_id": int(history["author_id"][0]) if history.height else -1, **account}
        hist = conform(history).filter(pl.col("created_at_ms") < when_ms).with_columns(pl.lit(account["author_id"]).alias("author_id"))
        now = pl.DataFrame({"author_id": [account["author_id"]], "snapshot_at_ms": [when_ms], "followers": [account.get("followers")]},
                           schema={"author_id": pl.Int64, "snapshot_at_ms": pl.Int64, "followers": pl.Int64}).filter(pl.col("followers").is_not_null())
        # Everything up to here is independent of the model; only the last line applies a checkpoint's
        # transforms. With `feature_cache` set (a dict), that model-independent part is kept, so a second
        # model scores the same drafts without recomputing features or embeddings (the pair benchmark).
        key = None
        if self.feature_cache is not None:
            key = stable_hash64(account["author_id"], when_ms, hist.height, int(hist["post_id"][-1]) if hist.height else 0,
                                *[(d.get("text"), len(d.get("media") or []), len(d.get("urls") or []), d.get("quoted_text"), d.get("kind")) for d in drafts])
        if key is not None and key in self.feature_cache:
            posts, emb, sid = self.feature_cache[key]
            if post_type.COLUMNS[0] in self.meta["num_base"] or self.type_correction:
                posts = with_post_type(posts)     # kept out of the cache (see below): 0.1 s, and never stale
        else:
            t0 = time.perf_counter()
            posts = self._feature_rows_batched(hist, drafts, account, when_ms, now) if self.fast and len(drafts) > 1 else self._feature_rows(hist, drafts, account, when_ms, now)
            t0 = self._tick("features", t0)
            emb = sid = None
            if self.meta["emb_dim"]:
                texts = features.render_texts(posts)
                use = self._rows_the_model_reads(posts, len(drafts)) if self.fast else np.arange(posts.height)
                vec = np.asarray(self.embed_fn([texts[i] for i in use]))
                emb = vec if len(use) == posts.height else np.zeros((posts.height, vec.shape[1]), vec.dtype)
                if emb is not vec:
                    emb[use] = vec               # rows the model never reads stay zero
                t0 = self._tick("embedding", t0)
                if self.sid_codebook and self.meta["sid_levels"]:
                    from xviral.embed import sid as sidmod

                    codes = sidmod.phoenix_codes(sidmod.assign(vec, self.sid_codebook))
                    sid = codes if emb is vec else np.zeros((posts.height, codes.shape[1]), codes.dtype)
                    if emb is not vec:
                        sid[use] = codes
                    self._tick("semantic_ids", t0)
            if key is not None:
                # The post-type columns are cheap and are recomputed from the cached frame on every read, so a cache
                # written before they existed, or before their definition changed, still serves any model.
                self.feature_cache[key] = (compact(posts.drop([c for c in post_type.COLUMNS if c in posts.columns])), emb, sid)
        return features.transform(posts, self.meta, emb=emb, sid=sid, device=self.device)

    @torch.no_grad()
    def score(self, history: pl.DataFrame, drafts: list[dict], account: dict, when_ms: int | None = None,
              mutual_share: float = 0.0) -> list[dict]:
        """One report per draft, in the order given. account: handle, followers, following, created_at_ms, verified."""
        when_ms = when_ms or int(time.time() * 1000)
        table, rows = self.featurize(history, drafts, account, when_ms)
        t0 = time.perf_counter()
        n = len(drafts)
        first = table.n - n                        # drafts are the newest rows of the single author
        members = self.model.models if hasattr(self.model, "models") and len(self.model.models) > 1 else []
        if self.fast and members:                  # one episode, each member run once; the mixture is built from the same outputs
            ep = episodes.build(table, torch.tensor([first], device=table.device), self.model.cfg, self.tcfg, train=False, n_candidates=n)
            outs = [m(table, ep) for m in members]
            by_id = {r["post_id"]: r for r in Predictor(self.model, table, self.tcfg, self.temperature).reports_from(ep, self.model.combine(outs), first, n, mutual_share)}
            for m, T, o in zip(members, self.model.temperatures, outs):
                for r in Predictor(m, table, self.tcfg, T).reports_from(ep, o, first, n, mutual_share):
                    by_id[r["post_id"]].setdefault("members", []).append({"views_q50": r["views_24h"]["q50"], "likes_q50": r["likes"]["q50"]})
        else:
            by_id = {r["post_id"]: r for r in Predictor(self.model, table, self.tcfg, self.temperature).report_joint(first, n, mutual_share=mutual_share)}
            for m, T in zip(members, self.model.temperatures if members else []):      # ensemble: each member's medians, for agreement checks
                for r in Predictor(m, table, self.tcfg, T).report_joint(first, n, mutual_share=mutual_share):
                    by_id[r["post_id"]].setdefault("members", []).append({"views_q50": r["views_24h"]["q50"], "likes_q50": r["likes"]["q50"]})
        self._tick("model", t0)
        acct = {"author_id": int(rows["author_id"][0]), **account}
        feat = {r["post_id"]: r for r in rows.tail(n).iter_rows(named=True)}
        recent = int((rows.head(first)["created_at_ms"] >= when_ms - 48 * 3_600_000).sum())
        reports = []
        for i, d in enumerate(drafts):             # table order among drafts is by ID; return them in the order given
            pid = int(self._draft_row(d, i, acct, when_ms)["post_id"][0])
            r = by_id[pid]
            r["text"] = d.get("text")
            if self.type_correction:
                shift = sum(w * float(feat[pid].get(c) or 0.0) for c, w in self.type_correction.items())
                if shift:
                    k = 10.0 ** shift
                    r["likes"] = {q: (v + 1.0) * k - 1.0 for q, v in r["likes"].items()}
                    if r["lift_vs_own_median"]["likes"] is not None:
                        r["lift_vs_own_median"]["likes"] *= k
                    r["type_correction_log10"] = shift
            r["history_posts_used"] = int(min(first, self.model.cfg.max_history))
            r["gates"] = gates.check(feat[pid], recent)      # deterministic distribution gates from X's code
            reports.append(r)
        return reports

    def timing(self, history: pl.DataFrame, draft: dict, account: dict, start_ms: int | None = None, hours_ahead: int = 48,
               step_hours: int = 2) -> list[dict]:
        """The same draft at each candidate posting time, best first.

        Each time is scored separately because the time features, the gap since the previous post and the
        cadence counts change with it. What moves between the rows is what the model learned about hour,
        weekday and spacing for accounts like this one from observed posts; it is an association, and the
        differences are usually small next to differences in content.
        """
        start_ms = start_ms or int(time.time() * 1000)
        out = []
        for h in range(0, hours_ahead + 1, step_hours):
            when = start_ms + h * 3_600_000
            r = self.score(history, [draft], account, when)[0]
            out.append({"when_ms": when, "hours_from_start": h, "views_q50": r["views_24h"]["q50"], "likes_q50": r["likes"]["q50"],
                        "p_own_top_10pct": r["p_own_top_10pct"], "x_value_score": r["x_value_score"]["in_network"]})
        key = "views_q50" if self.meta.get("has_views") else "likes_q50"
        return sorted(out, key=lambda r: -r[key])

    def link_variants(self, history: pl.DataFrame, draft: dict, account: dict, when_ms: int | None = None) -> list[dict]:
        """A draft with a link, scored as written, with the link moved to the end, and with the link removed
        (as when it goes in a reply). Empty when the draft has no link. Alternatives share one episode."""
        text = draft.get("text") or ""
        links = _URL.findall(text)
        if not links:
            return []
        stripped = " ".join(_URL.sub(" ", text).split(" ")).strip()
        stripped = "\n".join(line.rstrip() for line in stripped.split("\n")).strip()
        variants = [("as written", text)]
        if not text.rstrip().endswith(links[-1]) or len(links) > 1:
            variants.append(("link moved to the end", stripped + "\n\n" + links[0]))
        variants.append(("link removed (put it in a reply)", stripped))
        reports = self.score(history, [{**draft, "text": t, "urls": [] if "removed" in name else draft.get("urls", [])} for name, t in variants],
                             account, when_ms)
        for (name, _), r in zip(variants, reports):
            r["variant"] = name
        return reports


_URL = re.compile(r"https?://\S+")
