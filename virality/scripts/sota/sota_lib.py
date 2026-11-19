"""Track "sota" helpers: per-pair arrays for benchmark, development and launch pairs, with the confidence fitted two ways.

gap    the shipped rule: P(pick is right) = sigmoid(k * |score gap|), k cross-fitted over accounts.
dist   heteroscedastic: sigmoid(k * |score gap| / sqrt(sd_a^2 + sd_b^2)) with each version's predicted spread, k cross-fitted
       the same way; a pair between two posts the model is unsure of needs a wider gap to be called.
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, str(Path(__file__).parent.parent))
import rep_lib as L  # noqa: E402  (tracks/replicate: table by size of real gap, paired differences, account bootstrap)


def crossfit(x: np.ndarray, first_wins: np.ndarray, auth: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    ua = np.unique(auth)
    fold = dict(zip(ua, rng.integers(0, 5, len(ua))))
    fo = np.array([fold[a] for a in auth])
    conf, ks = np.zeros(len(x)), []
    for f in range(5):
        k = L.evaluate.fit_pair_scale(x[fo != f], first_wins[fo != f])
        ks.append(k)
        p = 1 / (1 + np.exp(-k * x[fo == f]))
        conf[fo == f] = np.maximum(p, 1 - p)
    return conf, float(np.mean(ks))


def make(key: pl.DataFrame, d: np.ndarray, s: np.ndarray, auth: np.ndarray, sd_a=None, sd_b=None) -> dict:
    keep = np.abs(d) >= np.log(1.25)
    key, d, s, auth = key.filter(pl.Series(keep)), d[keep], s[keep], auth[keep]
    A = {"key": key, "d": d, "s": s, "auth": auth, "correct": ((s * d) > 0).astype(float)}
    A["conf"], A["scale"] = crossfit(s, d > 0, auth)
    if sd_a is not None:
        A["z"] = s / np.sqrt(sd_a[keep] ** 2 + sd_b[keep] ** 2 + 1e-6)
        A["conf_dist"], A["scale_dist"] = crossfit(A["z"], d > 0, auth)
    return A


def bench(names: list[str], root: str = "data/bench3") -> dict:
    """Score-level ensemble of the named runs on the benchmark pairs (one name: that run)."""
    fr = []
    for n in names:
        R = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{root}/{n}/shard_*.parquet"))]).sort("post_id", "post_id_b")
        fr.append(R)
    R = fr[0]
    for f in fr[1:]:
        assert f.select("post_id", "post_id_b").equals(R.select("post_id", "post_id_b"))
    s = np.mean([f["s_likes"].to_numpy() for f in fr], 0)
    d = ((R["likes"].cast(pl.Float64) + 1).log() - (R["likes_b"].cast(pl.Float64) + 1).log()).to_numpy()
    sd = [np.sqrt(np.mean([f[c].to_numpy() ** 2 for f in fr], 0)) for c in ("sd_a", "sd_b")] if "sd_a" in R.columns else [None, None]
    A = make(R.select("post_id", "post_id_b"), d, s, R["author_id"].to_numpy(), *sd)
    A["likes_min"] = np.minimum(R["likes"].to_numpy(), R["likes_b"].to_numpy())[np.abs(d) >= np.log(1.25)]
    return A


def dev(names: list[str], root: str = "data/sota/dev", table: str = "data/tracks/replicate/dev_table") -> dict:
    fr = pl.read_parquet(f"{table}/pairs.parquet")
    z = [np.load(f"{root}/{n}.npz") for n in names]
    d = np.log((fr["likes"].to_numpy() + 1.0) / (fr["likes_b"].to_numpy() + 1.0))
    s = np.mean([x["s"] for x in z], 0)
    sd = [np.sqrt(np.mean([x[c] ** 2 for x in z], 0)) for c in ("sd_a", "sd_b")]
    return make(fr.select("post_id", "post_id_b"), d, s, fr["author_id"].to_numpy(), *sd)


def launch(names: list[str], root: str = "data/sota/launch") -> dict:
    """The launch track's 850 same-account launch pairs, rebuilt as scripts/launch/model_check.py and rep_launch.py do."""
    T = pl.read_parquet("data/tracks/launch/launch_posts.parquet")
    S = pl.read_parquet("data/tracks/launch/launch_scores.parquet").select("post_id", "pred_likes")
    P = T.join(S, on="post_id")
    p = P.join(P, on="author_id", suffix="_b").filter(pl.col("post_id") < pl.col("post_id_b"))
    p = p.with_columns(((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log()).alias("d")).filter(pl.col("d").abs() >= np.log(1.25))
    p = p.with_columns(pl.int_range(pl.len()).shuffle(seed=1).over("author_id").alias("r")).filter(pl.col("r") < 30).select("post_id", "post_id_b", "author_id", "d")
    mem = []
    for n in names:
        s = (S if n == "launch_track_stored" else pl.read_parquet(f"{root}/{n}.parquet")).with_columns((pl.col("pred_likes") + 1).log10().alias("q")).select("post_id", "q")
        j = p.join(s, on="post_id", how="left").join(s.rename({"post_id": "post_id_b", "q": "q_b"}), on="post_id_b", how="left")
        assert j["q"].null_count() == 0 and j["q_b"].null_count() == 0
        mem.append((j["q"] - j["q_b"]).to_numpy())
    return make(p.select("post_id", "post_id_b"), p["d"].to_numpy(), np.mean(mem, 0), p["author_id"].to_numpy())


def with_scale(A: dict, k: float, key: str = "s") -> np.ndarray:
    return 1 / (1 + np.exp(-k * np.abs(A[key])))


def cov_at(A: dict, gap: float, share: float, key: str = "s") -> tuple[float, list[float], int]:
    """Accuracy on the surest `share` of pairs with a real gap of at least `gap` (no confidence scale needed)."""
    big = np.abs(A["d"]) >= np.log(gap)
    n = int(round(share * big.sum()))
    m = np.zeros(int(big.sum()))
    m[np.argsort(-np.abs(A[key][big]), kind="stable")[:n]] = 1
    c = A["correct"][big]
    return float((c * m).sum() / n), L.boot(A["auth"][big], [c * m, m], lambda x, y: x / y), n
