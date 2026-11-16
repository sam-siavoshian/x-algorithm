#!/usr/bin/env python3
"""A model trained on the product's own question: of two versions of one message by one account, which did better?

    python scripts/evidence/pair_model.py --posts <work>/filtered/posts.parquet --after 2026-04-23 \
        --surrogate bench2_joined.parquet --out pair_model.json [--emb-cache emb.npz]

Training pairs: same account, same quoted post or link, different wording, at most 48 hours apart, both posted
before the cutoff. Test pairs: the same kind, after the cutoff. Inputs are differences between the two versions
(text-structure columns, media and link flags, text embedding), so nothing about the account or the topic can
carry the answer. Every pair is used in both orders with the sign flipped, which removes posting order.

With --surrogate (per-pair score differences of the surrogate models on the same pairs), the script also fits a
two-input blend on pre-cutoff pairs the surrogate did not train on and reports it on the test pairs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import tac_serve_eval as T  # noqa: E402

from xviral.features import text_structure as ts  # noqa: E402
from xviral.model import evaluate  # noqa: E402


def bands(conf: np.ndarray, correct: np.ndarray) -> dict:
    out = {}
    for lo, hi in ((0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 1.01)):
        m = (conf >= lo) & (conf < hi)
        out[f"{lo:.2f}-{min(hi, 1):.2f}"] = {"pairs": int(m.sum()), "accuracy": float(correct[m].mean()) if m.sum() else None}
    m = conf >= 0.6
    out["0.60+"] = {"share": float(m.mean()), "pairs": int(m.sum()), "accuracy": float(correct[m].mean()) if m.sum() else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--surrogate", default=None, help="parquet with post_id, post_id_b and l_<model> score-difference columns")
    ap.add_argument("--surrogate-members", default=None, help="comma-separated l_ column suffixes to average; default all")
    ap.add_argument("--emb-cache", default=None)
    ap.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb
    from xviral.util.core import date_to_ms

    d = pl.scan_parquet(args.posts).select(T.COLS).collect()
    pairs = T.find_pairs(d, 48.0).sort("post_id", "post_id_b")
    cut = date_to_ms(args.after)
    pairs = pairs.filter((pl.col("created_at_ms") >= cut) | (pl.col("created_at_ms_b") < cut))
    ids = pl.concat([pairs["post_id"], pairs["post_id_b"]]).unique()
    P = d.filter(pl.col("post_id").is_in(ids.implode()))
    F = ts.text_structure(P)
    tcols = [c for c in F.columns if c.startswith("ts_") and F.schema[c].is_numeric()]
    F = F.select("post_id", *tcols).join(P.select(
        "post_id", (pl.col("media").list.len() > 0).cast(pl.Float64).alias("x_media"),
        pl.col("media").list.eval(pl.element().struct.field("type") == "video").list.any().fill_null(False).cast(pl.Float64).alias("x_video"),
        (pl.col("urls").list.len() > 0).cast(pl.Float64).alias("x_link"),
        pl.col("text").fill_null("").str.len_chars().log1p().alias("x_loglen")), on="post_id")
    cols = [c for c in F.columns if c != "post_id"]
    pid = P["post_id"].to_numpy()
    if args.emb_cache and os.path.exists(args.emb_cache):
        z = np.load(args.emb_cache)
        assert (z["post_id"] == pid).all()
        E = z["emb"]
    else:
        from xviral.embed.text_small import SentenceEmbedder
        E = np.asarray(SentenceEmbedder(args.encoder, device="cpu").encode(P["text"].fill_null("").to_list()), dtype=np.float32)
        if args.emb_cache:
            np.savez(args.emb_cache, post_id=pid, emb=E)
    pos = {p: i for i, p in enumerate(pid)}
    Fm = F.join(pl.DataFrame({"post_id": pid}), on="post_id", how="right").select(cols).fill_null(0).to_numpy().astype(np.float64)
    ia = np.array([pos[x] for x in pairs["post_id"].to_list()])
    ib = np.array([pos[x] for x in pairs["post_id_b"].to_list()])
    X = np.hstack([Fm[ia] - Fm[ib], (E[ia] - E[ib]).astype(np.float64)])
    y = np.log1p(pairs["likes"].to_numpy().astype(float)) - np.log1p(pairs["likes_b"].to_numpy().astype(float))
    auth = pairs["author_id"].to_numpy()
    test = pairs["created_at_ms"].to_numpy() >= cut
    clear = np.abs(y) >= np.log(1.25)
    sd = X[~test].std(0) + 1e-9
    X = X / sd
    nf = len(cols)

    def fit(kind: str, tr: np.ndarray):
        Xt, yt = np.vstack([X[tr], -X[tr]]), np.concatenate([y[tr] > 0, y[tr] < 0]).astype(int)
        if kind == "logit_struct":
            m = LogisticRegression(C=0.3, fit_intercept=False, max_iter=2000).fit(Xt[:, :nf], yt)
            return lambda Z: Z[:, :nf] @ m.coef_[0]
        if kind == "logit_all":
            m = LogisticRegression(C=0.02, fit_intercept=False, max_iter=3000).fit(Xt, yt)
            return lambda Z: Z @ m.coef_[0]
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, reg_lambda=5.0, num_threads=args.threads, verbose=-1).fit(Xt, yt)
        return lambda Z: 0.5 * (m.predict(Z, raw_score=True) - m.predict(-Z, raw_score=True))   # antisymmetric by construction

    rng = np.random.default_rng(0)
    ua = np.unique(auth[~test])
    fold_of = dict(zip(ua, rng.integers(0, 5, len(ua))))
    fold = np.array([fold_of.get(a, -1) for a in auth])
    res: dict = {"train_pairs": int((~test & clear).sum()), "test_pairs": int((test & clear).sum()), "inputs": {"structure": nf, "embedding": int(E.shape[1])}, "models": {}}
    S = {}
    for kind in ("logit_struct", "logit_all", "gbdt"):
        s = np.zeros(len(y))
        for k in range(5):                                   # out-of-fold scores for the pre-cutoff pairs
            s[(fold == k) & ~test] = fit(kind, ~test & clear & (fold != k))(X[(fold == k) & ~test])
        s[test] = fit(kind, ~test & clear)(X[test])
        S[kind] = s
        te, tr = test & clear, ~test & clear
        k_ = evaluate.fit_pair_scale(s[tr], y[tr] > 0)       # confidence scale from out-of-fold pre-cutoff pairs
        p = 1 / (1 + np.exp(-k_ * s[te]))
        c = (s[te] * y[te]) > 0
        res["models"][kind] = {"oof_accuracy": float(((s[tr] * y[tr]) > 0).mean()), "test_accuracy": float(c.mean()), "bands": bands(np.maximum(p, 1 - p), c)}
        print(kind, json.dumps(res["models"][kind]))
    if args.surrogate:
        J = pl.read_parquet(args.surrogate)
        lc = [c for c in J.columns if c.startswith("l_")]
        if args.surrogate_members:
            lc = ["l_" + m for m in args.surrogate_members.split(",")]
        J = J.select("post_id", "post_id_b", "part", pl.mean_horizontal(lc).alias("sur"))
        Q = pairs.select("post_id", "post_id_b").with_row_index("i").join(J, on=["post_id", "post_id_b"])
        i = Q["i"].to_numpy()
        sur, part = Q["sur"].to_numpy(), Q["part"].to_numpy()
        for kind in ("logit_all", "gbdt"):
            pm, yy = S[kind][i], y[i]
            cal, te = (part == "calibration") & clear[i], (part == "test") & clear[i]
            Z = np.stack([sur, pm], 1)
            Z = Z / Z[cal].std(0)
            m = LogisticRegression(C=10.0, fit_intercept=False).fit(np.vstack([Z[cal], -Z[cal]]), np.concatenate([yy[cal] > 0, yy[cal] < 0]).astype(int))
            out = {}
            for name, s in (("surrogate", Z[:, 0] * m.coef_[0][0] / max(m.coef_[0][0], 1e-9)), ("pair_model", Z[:, 1]), ("blend", Z @ m.coef_[0])):
                k_ = evaluate.fit_pair_scale(s[cal], yy[cal] > 0)
                p = 1 / (1 + np.exp(-k_ * s[te]))
                c = (s[te] * yy[te]) > 0
                out[name] = {"pairs": int(te.sum()), "test_accuracy": float(c.mean()), "bands": bands(np.maximum(p, 1 - p), c)}
            out["blend_weights"] = [float(x) for x in m.coef_[0]]
            res["models"][kind + "_with_surrogate"] = out
            print(kind, "blend", json.dumps(out))
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
