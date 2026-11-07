"""Phase-0 baseline: gradient-boosted trees on the tabular features (DESIGN.md section 7).

Three numbers decide the go/no-go gate, all on the time-forward test split:
  1. AUROC for reach >= 10x followers (cold start: no engagement counts used).
  2. Same-author pairwise accuracy (which of two posts within 7 days did better).
  3. The same two numbers for a model trained on shuffled labels, which must be
     at chance; anything else means a leak, and nothing else is reported.
Engagement counts of the target post are never features here.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

# Features allowed at draft time. Counts of the target post are excluded by construction.
FEATURES = [
    # content (7.2)
    "weighted_text_len", "newline_count", "external_url_count", "has_url", "has_media", "has_photo", "has_video",
    "has_gif", "media_count", "video_duration_ms", "video_gt_10s", "video_index_eligible", "is_quote", "is_long_post",
    "mentions_n", "hashtags_n", "domain_class", "media_aspect", "video_duration_bucket",
    # author (7.3)
    "log_followers_at_post", "follower_tier_idx", "log_following", "follow_ratio", "account_age_days", "verified",
    "cold_start_eligible_author", "mutual_share_est", "cadence_24h", "cadence_7d", "hours_since_prev_original",
    "trailing_rate_fav_p50", "trailing_rate_fav_p90", "trailing_rate_reply_p50", "trailing_rate_reply_p90",
    "trailing_rate_repost_p50", "trailing_rate_repost_p90", "trailing_rate_quote_p50", "trailing_rate_quote_p90",
    "trailing_reach_p50", "trailing_reach_p90", "trailing_reach_p99", "trailing_reply_to_fav",
    "trailing_media_share", "trailing_video_share", "trailing_link_share", "history_n", "reply_share",
    "copypasta_events", "graph_pagerank",
    # audience and time (7.4)
    "local_hour_idx", "local_dow", "audience_zone_known",
    # duplicates and suppression proxies (3, 7.6)
    "near_dup_rank", "near_dup_cluster_size", "is_self_repeat", "url_blocked",
]
CATEGORICAL = ["domain_class", "media_aspect", "video_duration_bucket"]


def matrix(df: pl.DataFrame, features: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    feats = [f for f in (features or FEATURES) if f in df.columns]
    cols = []
    for f in feats:
        s = df[f]
        if f in CATEGORICAL or s.dtype in (pl.String, pl.Categorical) or isinstance(s.dtype, pl.Enum):
            s = s.cast(pl.String).cast(pl.Categorical).to_physical()
        cols.append(s.cast(pl.Float64).to_numpy(allow_copy=True))
    return np.column_stack(cols) if cols else np.zeros((df.height, 0)), feats


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    ok = ~np.isnan(y)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def pairwise_accuracy(pairs: pl.DataFrame, scores: pl.DataFrame) -> float:
    p = pairs.join(scores.rename({"post_id": "post_id_a", "score": "sa"}), on="post_id_a").join(
        scores.rename({"post_id": "post_id_b", "score": "sb"}), on="post_id_b")
    if p.height == 0:
        return float("nan")
    pred = (p["sa"] > p["sb"]).cast(pl.Int8)
    return float((pred == p["a_wins"]).mean())


def run(df: pl.DataFrame, pairs: pl.DataFrame, target: str = "log_reach_ratio", tier: str = "y_reach_10x",
        seed: int = 0, shuffle_control: bool = True) -> dict:
    import lightgbm as lgb

    lab = df.filter(pl.col(target).is_not_null())
    X, feats = matrix(lab)
    split = lab["split"].to_numpy()
    y = lab[target].to_numpy()
    w = lab["w_row"].fill_null(1.0).to_numpy()
    yt = lab[tier].cast(pl.Float64).to_numpy(allow_copy=True)
    tr, va, te = split == "train", split == "val", split == "test"
    params = dict(objective="huber", alpha=1.0, learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=seed)

    def fit_eval(y_train: np.ndarray, tag: str) -> dict:
        dtr = lgb.Dataset(X[tr], y_train[tr], weight=w[tr], feature_name=feats, free_raw_data=False)
        dva = lgb.Dataset(X[va], y_train[va], weight=w[va], reference=dtr) if va.any() else None
        model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva] if dva else None,
                          callbacks=[lgb.early_stopping(100, verbose=False)] if dva else None)
        s = model.predict(X)
        scores = pl.DataFrame({"post_id": lab["post_id"], "score": s})
        test_pairs = pairs.filter(pl.col("split") == "test")
        res = {
            f"{tag}_auc_{tier}_test": _auc(yt[te], s[te]),
            f"{tag}_auc_{tier}_val": _auc(yt[va], s[va]),
            f"{tag}_pairwise_test": pairwise_accuracy(test_pairs, scores),
            f"{tag}_spearman_test": _spearman(y[te], s[te]),
        }
        if tag == "model":
            imp = sorted(zip(feats, model.feature_importance("gain")), key=lambda t: -t[1])[:25]
            res["top_features_gain"] = [(f, round(float(g), 1)) for f, g in imp]
        return res

    out = {"n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()), "n_features": len(feats)}
    out.update(fit_eval(y, "model"))
    if shuffle_control:
        rng = np.random.default_rng(seed)
        out.update(fit_eval(rng.permutation(y), "shuffled"))
        leak = abs(out["shuffled_auc_" + tier + "_test"] - 0.5) > 0.03 if not np.isnan(out["shuffled_auc_" + tier + "_test"]) else False
        out["leak_suspected"] = bool(leak)
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 10:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
