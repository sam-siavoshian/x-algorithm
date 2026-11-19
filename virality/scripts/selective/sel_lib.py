"""Selective track: turn the rich scores of sel_score.py into one row of deferral features per pair.

Everything is oriented by the ensemble's call ("pick" = the version the calling ensemble prefers, "other" = the one it
does not), never by which version was posted first: posting order is not known for two drafts and leaks the outcome.
The feature functions here are the ones xviral/selective.py uses at serving time (pair_features), so the offline and
the product's inputs are the same numbers.
"""
from __future__ import annotations

import glob

import numpy as np
import polars as pl

from xviral import selective as S

FT = ["pairs_ft_ts46_s1", "pairs_ft_ts46_s2", "pairs_ft_compare_v2_full", "ft_ts0_s0", "ft_ts0_s1", "ft_ts20_s1", "ft_ts20_s2", "ft_cmp_full", "ft_cmp_aud", "ft_cmpv2_aud"]
ST = ["start_ts46_s1", "start_ts46_s2", "start_compare_v2_full", "start_ts0_s0", "start_ts0_s1", "start_ts20_s1", "start_ts20_s2", "start_cmp_full", "start_cmp_aud", "start_cmpv2_aud"]
BASE3 = ST[:3]
ENS = {"ft10": FT, "start10": ST, "base3": BASE3}


def load_set(name: str) -> dict:
    """All shards of one scored set: Q [M, U, C, 8], H [M, U, C, 70], E [U, 8], DV [U, C, 384] or None, rows (one per version)."""
    files = sorted(glob.glob(f"data/sel/{name}/shard_*.npz"))
    zs = [np.load(f) for f in files]
    uid = np.concatenate([z["unit_id"] for z in zs])
    o = np.argsort(uid)
    out = {"names": [str(x) for x in zs[0]["names"]], "unit_id": uid[o],
           "Q": np.concatenate([z["Q"] for z in zs], 1)[:, o], "H": np.concatenate([z["H"] for z in zs], 1)[:, o].astype(np.float32),
           "E": np.concatenate([z["E"] for z in zs])[o],
           "DV": np.concatenate([z["DV"] for z in zs])[o].astype(np.float32) if "DV" in zs[0] else None}
    rows = pl.concat([pl.read_parquet(f.replace(".npz", ".parquet")) for f in files], how="diagonal_relaxed")
    out["rows"] = {(r["unit_id"], r["cand"]): r for r in rows.iter_rows(named=True)}
    return out


def version(D: dict, u: int, c: int, members: list[str]) -> dict | None:
    """One scored version in the shape xviral.selective.pair_features reads (see selective.version_from_report)."""
    r = D["rows"].get((int(D["unit_id"][u]), c))
    if r is None or np.isnan(D["Q"][0, u, c, 1]):
        return None
    idx = [D["names"].index(m) for m in members]
    e = D["E"][u]
    return {"members": [{"likes_q10": float(D["Q"][m, u, c, 0]), "likes_q50": float(D["Q"][m, u, c, 1]), "likes_q90": float(D["Q"][m, u, c, 2]),
                         "rel_like_q50": float(D["Q"][m, u, c, 4]), "p_own_top10": float(D["Q"][m, u, c, 5]), "likes_hist": D["H"][m, u, c]} for m in idx],
            "features": r, "history": {k: r.get(k) for k in S.HISTORY_KEYS},
            "novelty": {"sim_mean": float(e[c * 3]), "sim_top3": float(e[c * 3 + 1]), "sim_centroid": float(e[c * 3 + 2]), "cohesion": float(e[7])},
            "embedding": D["DV"][u, c] if D["DV"] is not None else None}


def pair_rows(D: dict, pairs: pl.DataFrame, kind: str, ens: str = "ft10") -> pl.DataFrame:
    """pairs: ua, ca, ub, cb (unit index and candidate of each side), author_id, likes_a, likes_b, plus any columns to keep.
    Returns one row per pair with the deferral features for calls made by `ens`, the other ensembles' gaps, and labels."""
    recs = []
    for p in pairs.iter_rows(named=True):
        va = {e: version(D, p["ua"], p["ca"], m) for e, m in ENS.items()}
        vb = {e: version(D, p["ub"], p["cb"], m) for e, m in ENS.items()}
        if va[ens] is None or vb[ens] is None:
            continue
        cos = float(D["E"][p["ua"], 6]) if p["ua"] == p["ub"] else None
        f = S.pair_features(va[ens], vb[ens], kind=kind, cos_ab=cos, other=(va["start10"], vb["start10"]) if ens == "ft10" else (va["ft10"], vb["ft10"]))
        d = float(np.log((p["likes_a"] or 0) + 1) - np.log((p["likes_b"] or 0) + 1))
        s = {e: float(np.mean([x["likes_q50"] for x in va[e]["members"]]) - np.mean([x["likes_q50"] for x in vb[e]["members"]])) for e in ENS}
        keep = {k: v for k, v in p.items() if k not in ("ua", "ca", "ub", "cb")}
        recs.append({**keep, "d": d, **{f"s_{e}": v for e, v in s.items()}, "correct": float(s[ens] * d > 0), **{f"f_{k}": v for k, v in f.items()}})
    return pl.DataFrame(recs, infer_schema_length=None)


def same_topic_pairs(name: str, src: str) -> pl.DataFrame:
    U = pl.read_parquet(f"data/sel/units_{name}.parquet").sort("unit_id")
    P = pl.read_parquet(src)
    U = U.with_columns(pl.col("post_ids").list.get(0).alias("post_id"), pl.col("post_ids").list.get(1).alias("post_id_b")).join(P.drop("author_id"), on=["post_id", "post_id_b"], how="left")
    return U.select(pl.col("unit_id").alias("ua"), pl.lit(0).alias("ca"), pl.col("unit_id").alias("ub"), pl.lit(1).alias("cb"), "author_id", "post_id", "post_id_b",
                    pl.col("likes").alias("likes_a"), pl.col("likes_b"), *[c for c in ("part", "h48", "bench_account") if c in U.columns])


def launch_pairs(name: str, cap: int = 30) -> pl.DataFrame:
    """Same account, both launch-type posts of the set, likes 1.25x apart, at most `cap` per account (shuffle seed 1): scripts/launch/model_check.py."""
    U = pl.read_parquet(f"data/sel/units_{name}.parquet").with_columns(pl.col("post_ids").list.get(0).alias("post_id"))
    T = pl.read_parquet("data/tracks/launch/launch_posts.parquet")
    if name == "launch_test":       # exactly the frame rep_launch.py builds its pairs from, so the 850 pairs come out identical
        P = T.join(pl.read_parquet("data/tracks/launch/launch_scores.parquet").select("post_id", "pred_likes"), on="post_id")
    else:
        P = T.join(U.select("post_id"), on="post_id")
    p = P.join(P, on="author_id", suffix="_b").filter(pl.col("post_id") < pl.col("post_id_b"))
    p = p.with_columns(((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log()).alias("d0")).filter(pl.col("d0").abs() >= np.log(1.25))
    p = p.with_columns(pl.int_range(pl.len()).shuffle(seed=1).over("author_id").alias("r")).filter(pl.col("r") < cap)
    u = U.select("post_id", pl.col("unit_id").alias("ua"))
    p = p.join(u, on="post_id").join(u.rename({"post_id": "post_id_b", "ua": "ub"}), on="post_id_b")
    return p.select("ua", pl.lit(0).alias("ca"), "ub", pl.lit(0).alias("cb"), "author_id", "post_id", "post_id_b", pl.col("likes").alias("likes_a"), pl.col("likes_b"),
                    "akind", pl.col("tier"), "created_at_ms", "created_at_ms_b")


# --- evaluation --------------------------------------------------------------------------------------------

def boot(auth: np.ndarray, cols: list[np.ndarray], stat, n: int = 1000, seed: int = 1) -> list[float]:
    """95% interval from resampling accounts; `cols` are per-pair arrays summed per account, `stat` maps the sums to a number."""
    ua, inv = np.unique(auth, return_inverse=True)
    sums = [np.bincount(inv, weights=c.astype(float), minlength=len(ua)) for c in cols]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ua), (n, len(ua)))
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = stat(*[s[idx].sum(1) for s in sums])
    return [float(x) for x in np.nanpercentile(vals, [2.5, 97.5])]


def top_mask(score: np.ndarray, share: float) -> np.ndarray:
    n = int(round(share * len(score)))
    m = np.zeros(len(score), bool)
    m[np.argsort(-score, kind="stable")[:n]] = True
    return m


def curve(score: np.ndarray, correct: np.ndarray, auth: np.ndarray, d: np.ndarray, covs=(0.2, 0.3, 0.4, 0.5), gaps=(1.25, 2.0)) -> dict:
    """Accuracy of the surest `cov` share of pairs (ranked inside each gap group), with account-resampled intervals."""
    out = {}
    for g in gaps:
        k = np.abs(d) >= np.log(g)
        sc, c, a = score[k], correct[k], auth[k]
        out[f"{g}x"] = {"pairs": int(k.sum()), "accuracy": float(c.mean()), "accuracy_ci": boot(a, [c, np.ones(len(c))], lambda x, y: x / y)}
        for cov in covs:
            m = top_mask(sc, cov)
            out[f"{g}x"][f"cov{int(cov * 100)}"] = {"pairs": int(m.sum()), "accuracy": float(c[m].mean()), "ci": boot(a, [c * m, m], lambda x, y: x / y)}
    return out


def at_threshold(p: np.ndarray, correct: np.ndarray, auth: np.ndarray, d: np.ndarray, thr: float = 0.6, gaps=(1.25, 2.0)) -> dict:
    out = {}
    for g in gaps:
        k = np.abs(d) >= np.log(g)
        m, c, a = p[k] >= thr, correct[k], auth[k]
        out[f"{g}x"] = {"pairs": int(k.sum()), "spoken": int(m.sum()), "share": float(m.mean()), "accuracy": float(c[m].mean()) if m.sum() else None,
                        "share_ci": boot(a, [m, np.ones(len(m))], lambda x, y: x / y), "ci": boot(a, [c * m, m], lambda x, y: x / y) if m.sum() else None}
    return out


def paired_cov(score_a, score_b, correct_a, correct_b, auth, d, cov=0.4, gap=2.0) -> dict:
    k = np.abs(d) >= np.log(gap)
    ma, mb = top_mask(score_a[k], cov), top_mask(score_b[k], cov)
    ca, cb, a = correct_a[k], correct_b[k], auth[k]
    g = float(ca[ma].mean() - cb[mb].mean())
    return {"gap": g, "ci": boot(a, [ca * ma, ma, cb * mb, mb], lambda p, q, r, s: p / q - r / s)}


def aurc(score, correct, d, gap=2.0, covs=(0.2, 0.3, 0.4, 0.5)) -> float:
    k = np.abs(d) >= np.log(gap)
    return float(np.mean([correct[k][top_mask(score[k], c)].mean() for c in covs]))
