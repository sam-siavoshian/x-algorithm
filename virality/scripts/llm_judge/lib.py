"""Shared helpers for the llm_judge track: load judge outputs, metrics with account-resampled intervals."""
import hashlib, json, numpy as np, polars as pl

L125, L2 = np.log(1.25), np.log(2)


def half_of(a: int) -> str:
    return "A" if int(hashlib.md5(str(a).encode()).hexdigest(), 16) % 2 == 0 else "B"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_judge(path: str) -> pl.DataFrame:
    """One row per pair: j = order-balanced log-odds that post_id beat post_id_b; pos = mean P(first-shown wins) (position bias)."""
    R = pl.DataFrame([r for r in map(json.loads, open(path)) if r.get("p") is not None]).unique(["post_id", "post_id_b", "order"], keep="last")
    a = R.filter(pl.col("order") == 0).select("post_id", "post_id_b", pl.col("p").alias("p0"))
    b = R.filter(pl.col("order") == 1).select("post_id", "post_id_b", pl.col("p").alias("p1"))
    J = a.join(b, on=["post_id", "post_id_b"])
    l0, l1 = logit(J["p0"].to_numpy()), logit(J["p1"].to_numpy())
    return J.with_columns(pl.Series("j", (l0 - l1) / 2), pl.Series("j0", l0), pl.Series("j1", -l1), pl.Series("pos", (J["p0"].to_numpy() + J["p1"].to_numpy()) / 2))


def pairs(name: str) -> pl.DataFrame:
    P = pl.read_parquet(f"data/llm/pairs_{name}.parquet")
    d = np.log1p(P["likes"].to_numpy().astype(float)) - np.log1p(P["likes_b"].to_numpy().astype(float))
    return P.with_columns(pl.Series("d", d), pl.Series("half", [half_of(a) for a in P["author_id"].to_list()]))


def ci(auth, stat, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    ua, inv = np.unique(auth, return_inverse=True)
    mem = [np.flatnonzero(inv == j) for j in range(len(ua))]
    v = [stat(np.concatenate([mem[j] for j in rng.integers(0, len(ua), len(ua))])) for _ in range(n)]
    return [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]


def acc(s, d, auth, mask=None, with_ci=True):
    m = (s != 0) if mask is None else mask & (s != 0)
    c = ((s * d) > 0).astype(float)
    if m.sum() == 0:
        return {"pairs": 0}
    out = {"pairs": int(m.sum()), "acc": float(c[m].mean())}
    if with_ci:
        out["ci"] = ci(auth, lambda i: c[i][m[i]].mean() if m[i].sum() else np.nan)
    return out


def fit_scale(s, y):
    grid = np.geomspace(0.02, 60, 300)
    def nll(k):
        p = 1 / (1 + np.exp(-k * s))
        return -(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).mean()
    return float(grid[int(np.argmin([nll(k) for k in grid]))])


def crossfit_conf(s, d, auth, seed=0):
    """Stated confidence with the scale cross-fitted over accounts (as compare_runs.py)."""
    rng = np.random.default_rng(seed)
    ua = np.unique(auth)
    fold = dict(zip(ua, rng.integers(0, 5, len(ua))))
    fo = np.array([fold[a] for a in auth])
    conf, ks = np.zeros(len(s)), []
    for k_ in range(5):
        k = fit_scale(s[fo != k_], (d[fo != k_] > 0).astype(float)); ks.append(k)
        p = 1 / (1 + np.exp(-k * s[fo == k_])); conf[fo == k_] = np.maximum(p, 1 - p)
    return conf, float(np.mean(ks))
