#!/usr/bin/env python3
"""Selective track: score "units" through the serving path with many checkpoints and keep everything a deferral
model could read: each member's quantiles and full like histogram for each version, plus the model-independent
feature rows of the versions and summary statistics of the account's visible history.

    python scripts/selective/sel_score.py --units units.parquet --out data/sel/<set>/shard_0 --shard 0/4 \
        --feature-cache cache_0.pkl [--emb-seed cache.pkl|table_dir ...] name=ckpt.pt ...

A unit is one call of DraftScorer: (unit_id, author_id, when_ms, post_ids [1 or 2 posts]). The posts are handed over as
alternative drafts at `when_ms` with the account's previous `--history` posts (exactly tac_serve_eval.py for pairs and
score_launch.py for single posts), so an existing feature cache of those scripts is hit and nothing is recomputed.

Writes <out>.npz: Q [models, units, cands, 8] (likes q10 q50 q90 mean, rel_like q50, p_own_top10_like, views q50, p_reach_10x),
H [models, units, cands, 70] (like histogram, float16), E [units, 8] (embedding statistics), and <out>.parquet with one
row per version: unit_id, cand, post_id and the feature-frame columns, plus history statistics.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle

import numpy as np
import polars as pl
import torch

from xviral.model import episodes
from xviral.model import features as F

log = logging.getLogger("sel")
COLS = ["post_id", "author_id", "author_handle", "author_display_name", "created_at_ms", "snapshot_at_ms", "text", "kind", "media", "urls",
        "quoted_post_id", "quoted_text", "likes", "replies", "reposts", "quotes", "views", "author_followers", "author_following", "source"]
QN = ("likes_q10", "likes_q50", "likes_q90", "likes_mean", "rel_like_q50", "p_own_p90_rel_like", "views_q50", "p_reach_10x")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--history", type=int, default=56)
    ap.add_argument("--posts", default="data/work/filtered/posts.parquet")
    ap.add_argument("--feature-cache", default=None)
    ap.add_argument("--emb-seed", action="append", default=[])
    ap.add_argument("--sid-codebook", default="data/work/emb/sid_codebook.npz")
    ap.add_argument("--fast", action="store_true", help="the product's request path for cache misses")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("ckpt", nargs="+")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    from xviral.serve import DraftScorer

    U = pl.read_parquet(a.units).sort("unit_id")
    i, n = (int(x) for x in a.shard.split("/"))
    U = U.with_row_index("_r").filter(pl.col("_r") % n == i).drop("_r")
    if a.limit:
        U = U.head(a.limit)
    d = pl.scan_parquet(a.posts).select(COLS).filter(pl.col("author_id").is_in(U["author_id"].unique().to_list())).collect()
    by_author = {k: g.sort("created_at_ms") for (k,), g in d.group_by("author_id")}
    want = set(U["post_ids"].explode().to_list())
    by_id = {r["post_id"]: r for r in d.filter(pl.col("post_id").is_in(list(want))).iter_rows(named=True)}
    del d

    enc = {}

    def embed(texts):
        if "m" not in enc:
            from xviral.embed import api as embed_api
            enc["m"] = embed_api.make_encoder(embed_api.common_encoder([sc.meta for sc in scorers]))      # the checkpoints' own encoder
            log.info("encoder loaded (cache miss)")
        return enc["m"].encode(texts)

    fcache = pickle.load(open(a.feature_cache, "rb")) if a.feature_cache and os.path.exists(a.feature_cache) else {}
    n_cache0 = len(fcache)
    embs: dict = {}
    names, scorers = [], []
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        sc = DraftScorer(path, embed, device="cpu", sid_codebook=a.sid_codebook)
        sc._emb_cache, sc.emb_cache_limit, sc.feature_cache, sc.fast = embs, 10 ** 9, fcache, a.fast
        names.append(name)
        scorers.append(sc)
    for seed in a.emb_seed:
        n0 = len(embs)
        if os.path.isdir(seed):
            ids = np.load(os.path.join(seed, "post_id.npy"))
            allp = np.fromiter((p for g in by_author.values() for p in g["post_id"].to_list()), np.int64)
            sel = np.flatnonzero(np.isin(ids, allp) & np.load(os.path.join(seed, "has_emb.npy")))
            E = np.load(os.path.join(seed, "emb.npy"), mmap_mode="r")
            for tt, r_ in zip(pl.read_parquet(os.path.join(seed, "texts.parquet"))["text"].gather(sel).to_list(), sel):
                embs.setdefault(tt, np.asarray(E[r_], np.float32))
        else:
            blob = pickle.load(open(seed, "rb"))
            if blob and isinstance(next(iter(blob.values())), np.ndarray):      # {rendered text: vector} from sel_seed.py
                embs.update(blob)
            else:
                for posts_, emb_, _sid in blob.values():
                    for tt, v in zip(F.render_texts(posts_), emb_):
                        if np.any(v):
                            embs.setdefault(tt, np.asarray(v, np.float32))
        log.info("embeddings seeded from %s: %d", seed, len(embs) - n0)

    nc = int(U["post_ids"].list.len().max())
    M = len(scorers)
    Q = np.full((M, U.height, nc, len(QN)), np.nan, np.float32)
    H = np.zeros((M, U.height, nc, 70), np.float16)
    E = np.full((U.height, 8), np.nan, np.float32)
    DV = np.zeros((U.height, nc, int(scorers[0].meta.get("emb_dim") or 384)), np.float16)      # the versions' own embeddings (for pairs built from two units)
    rows_out = []
    for u, r in enumerate(U.iter_rows(named=True)):
        g = by_author[r["author_id"]]
        xs = [by_id[p] for p in r["post_ids"]]
        hist = g.filter(pl.col("created_at_ms") < r["when_ms"]).tail(a.history)
        account = {"author_id": r["author_id"], "handle": xs[0]["author_handle"], "display_name": xs[0]["author_display_name"],
                   "followers": xs[0]["author_followers"], "following": xs[0]["author_following"]}
        drafts = [{"text": x["text"], "kind": x["kind"], "media": x["media"], "urls": x["urls"], "quoted_text": x["quoted_text"], "aux": {}} for x in xs]
        try:
            for m, sc in enumerate(scorers):
                with torch.no_grad():
                    table, rows = sc.featurize(hist, drafts, account, r["when_ms"])
                    first = table.n - len(drafts)
                    ep = episodes.build(table, torch.tensor([first]), sc.model.cfg, sc.tcfg, train=False, n_candidates=len(drafts))
                    out = sc.model(table, ep)
                    s = sc.model.summarize(ep, out, sc.temperature)
                    ph = torch.softmax(out["hist"]["likes"].float() / (sc.temperature or {}).get("likes", 1.0), -1)[0].numpy()
                    tid = table.t["post_id"][first:first + len(drafts)].tolist()
                    acct = {"author_id": int(rows["author_id"][0]), **account}
                    did = [int(sc._draft_row(dd, j, acct, r["when_ms"])["post_id"][0]) for j, dd in enumerate(drafts)]
                    for j, pid in enumerate(did):
                        c = tid.index(pid)
                        Q[m, u, j] = [float(s[k][0, c]) for k in QN]
                        H[m, u, j] = ph[c]
                if m == 0:      # model-independent: the versions' feature rows, history statistics, embedding statistics
                    key_rows = rows.tail(len(drafts))
                    hrows = rows.head(first)
                    hl = np.log10(hrows["likes"].drop_nulls().to_numpy().astype(float) + 1) if first else np.array([])
                    hs = {"h_n": first, "h_like_med": float(np.median(hl)) if len(hl) else np.nan, "h_like_sd": float(hl.std()) if len(hl) > 1 else np.nan,
                          "h_like_iqr": float(np.subtract(*np.percentile(hl, [75, 25]))) if len(hl) > 3 else np.nan,
                          "h_like_p90": float(np.percentile(hl, 90)) if len(hl) else np.nan, "h_like_max": float(hl.max()) if len(hl) else np.nan,
                          "h_span_days": float((r["when_ms"] - hrows["created_at_ms"].min()) / 86_400_000) if first else np.nan,
                          "h_media_share": float(hrows["has_media"].mean()) if first else np.nan,
                          "h_words_med": float(hrows["ts_words"].median()) if first else np.nan}
                    for j, pid in enumerate(did):
                        fr = key_rows.filter(pl.col("post_id") == pid).row(0, named=True)
                        rows_out.append({"unit_id": r["unit_id"], "cand": j, "real_post_id": r["post_ids"][j], **hs,
                                         **{k: v for k, v in fr.items() if isinstance(v, (int, float, bool)) or v is None},
                                         "kind": fr["kind"], "domain_class": str(fr.get("domain_class"))})
                    # embeddings: cached entry holds (posts, emb, sid) aligned; find it again through the scorer's cache key path
                    ent = _entry(sc, hist, drafts, account, r["when_ms"])
                    if ent is not None and ent[1] is not None:
                        posts_c, emb_c = ent[0], np.asarray(ent[1], np.float32)
                        ids_c = posts_c["post_id"].to_list()
                        dv = [emb_c[ids_c.index(p)] for p in did]
                        hmask = np.array([(p not in did) for p in ids_c])
                        hv = emb_c[hmask]
                        hv = hv[np.abs(hv).sum(1) > 0][-48:]
                        nrm = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)      # noqa: E731
                        dvn = [nrm(v) for v in dv]
                        for j, v in enumerate(dvn):
                            DV[u, j] = v
                        e = [np.nan] * 8
                        if len(hv):
                            hn = nrm(hv)
                            cen = nrm(hn.mean(0))
                            for j, v in enumerate(dvn[:2]):
                                sims = hn @ v
                                e[j * 3:j * 3 + 3] = [float(sims.mean()), float(np.sort(sims)[-3:].mean()), float(cen @ v)]
                            e[7] = float((hn @ cen).mean())
                        if len(dvn) > 1:
                            e[6] = float(dvn[0] @ dvn[1])
                        E[u] = e
        except Exception as ex:  # noqa: BLE001
            log.warning("unit %s skipped: %r", r["unit_id"], ex)
            continue
        if (u + 1) % 100 == 0:
            log.info("scored %d/%d units", u + 1, U.height)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out + ".npz", Q=Q, H=H, E=E, DV=DV, names=np.array(names), unit_id=U["unit_id"].to_numpy(), qn=np.array(QN))
    pl.DataFrame(rows_out, infer_schema_length=None).write_parquet(a.out + ".parquet")
    if a.feature_cache and len(fcache) > n_cache0:
        with open(a.feature_cache, "wb") as fc:
            pickle.dump(fcache, fc)
        log.info("feature cache written: %d entries (%d new)", len(fcache), len(fcache) - n_cache0)
    log.info("done %s: %d units, %d models, cache misses %d", a.out, U.height, M, len(fcache) - n_cache0)


def _entry(sc, hist, drafts, account, when_ms):
    """The cached (posts, emb, sid) of this call, by the same key DraftScorer.featurize builds."""
    from xviral.schema import conform
    from xviral.util.core import stable_hash64
    account = {"author_id": int(hist["author_id"][0]) if hist.height else -1, **account}
    h = conform(hist).filter(pl.col("created_at_ms") < when_ms)
    key = stable_hash64(account["author_id"], when_ms, h.height, int(h["post_id"][-1]) if h.height else 0,
                        *[(d.get("text"), len(d.get("media") or []), len(d.get("urls") or []), d.get("quoted_text"), d.get("kind")) for d in drafts])
    return sc.feature_cache.get(key)


if __name__ == "__main__":
    main()
