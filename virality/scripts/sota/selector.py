#!/usr/bin/env python3
"""Pair-level selector (selective prediction): the probability that the ensemble's pick is right, from the score gap, the
predicted spreads and the predicted like level of the two versions. Fitted on development pairs only (cross-fitted over
accounts there to judge it), then applied unchanged to the benchmark pairs, once.

    python scripts/sota/selector.py --members lvl_ts46_s1,lvl_ts46_s2,lvl_v2full --out data/sota/results/selector.json
"""
import argparse, glob, json, sys
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "scripts/sota"); sys.path.insert(0, "scripts")
import sota_lib as S
L = S.L
ap = argparse.ArgumentParser(); ap.add_argument("--members", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
names = a.members.split(",")

def feats(s, sd_a, sd_b, m_a, m_b):
    g = np.abs(s)
    sd = np.sqrt(sd_a ** 2 + sd_b ** 2)
    return np.stack([g, g / sd, sd, np.minimum(m_a, m_b), g * np.minimum(m_a, m_b)], 1)
FN = ["gap", "gap/spread", "spread", "lower predicted level (log10 likes)", "gap x level"]

def load_dev():
    fr = pl.read_parquet("data/tracks/replicate/dev_table/pairs.parquet")
    z = [np.load(f"data/sota/dev/{n}.npz") for n in names]
    mean = lambda k: np.mean([x[k] for x in z], 0)  # noqa: E731
    d = np.log((fr["likes"].to_numpy() + 1.0) / (fr["likes_b"].to_numpy() + 1.0))
    return d, mean("s"), feats(mean("s"), mean("sd_a"), mean("sd_b"), mean("m_a"), mean("m_b")), fr["author_id"].to_numpy(), np.minimum(fr["likes"].to_numpy(), fr["likes_b"].to_numpy())

def load_bench():
    fr = [pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"data/bench3/{n}/shard_*.parquet"))]).sort("post_id", "post_id_b") for n in names]
    mean = lambda k: np.mean([f[k].to_numpy() for f in fr], 0)  # noqa: E731
    R = fr[0]
    d = ((R["likes"].cast(pl.Float64) + 1).log() - (R["likes_b"].cast(pl.Float64) + 1).log()).to_numpy()
    return d, mean("s_likes"), feats(mean("s_likes"), mean("sd_a"), mean("sd_b"), mean("m_a"), mean("m_b")), R["author_id"].to_numpy(), np.minimum(R["likes"].to_numpy(), R["likes_b"].to_numpy())

def prep(d, s, X, auth, lmin):
    k = np.abs(d) >= np.log(1.25)
    return d[k], s[k], X[k], auth[k], lmin[k], ((s[k] * d[k]) > 0).astype(float)

def table(name, d, auth, c, conf, order, out):
    out[name] = {}
    one = np.ones(len(c))
    for g in (1.25, 2.0, 5.0):
        m = np.abs(d) >= np.log(g)
        k = (conf[m] >= 0.6).astype(float)
        row = {"pairs": int(m.sum()), "accuracy": float(c[m].mean()), "confident_share": float(k.mean()), "confident_accuracy": float((c[m] * k).sum() / max(k.sum(), 1)),
               "confident_accuracy_ci": L.boot(auth[m], [c[m] * k, k], lambda x, y: x / y)}
        for sh in (0.3, 0.4, 0.5):
            n = int(round(sh * m.sum())); mm = np.zeros(int(m.sum())); mm[np.argsort(-order[m], kind="stable")[:n]] = 1
            row[f"surest_{int(sh * 100)}"] = {"accuracy": float((c[m] * mm).sum() / n), "ci": L.boot(auth[m], [c[m] * mm, mm], lambda x, y: x / y), "_m": mm}
        out[name][f"{g}x"] = row
        print(f"  {name:9s} {g}x n={row['pairs']} acc={100 * row['accuracy']:.1f} confident {100 * row['confident_share']:.0f}% at {100 * row['confident_accuracy']:.1f} "
              f"({100 * row['confident_accuracy_ci'][0]:.1f}-{100 * row['confident_accuracy_ci'][1]:.1f}); surest 30/40/50%: " + " / ".join(f"{100 * row[f'surest_{q}']['accuracy']:.1f}" for q in (30, 40, 50)))

def paired(out, d, auth, c, g=2.0, q=40):
    m = np.abs(d) >= np.log(g)
    ma, mb = out["selector"][f"{g}x"][f"surest_{q}"]["_m"], out["gap"][f"{g}x"][f"surest_{q}"]["_m"]
    gap = float((c[m] * ma).sum() / ma.sum() - (c[m] * mb).sum() / mb.sum())
    ci = L.boot(auth[m], [c[m] * ma, ma, c[m] * mb, mb], lambda p, q_, r, s_: p / q_ - r / s_)
    print(f"  selector minus gap, surest {q}% of {g}x pairs: {100 * gap:+.1f} ({100 * ci[0]:+.1f} to {100 * ci[1]:+.1f})")
    return {"gap": gap, "ci": ci}

res = {}
d, s, X, auth, lmin, c = prep(*load_dev())
mu, sg = X.mean(0), X.std(0)
fit = lambda Xt, ct: LogisticRegression(C=1.0, max_iter=1000).fit((Xt - mu) / sg, ct)  # noqa: E731
rng = np.random.default_rng(0); ua = np.unique(auth); fold = dict(zip(ua, rng.integers(0, 5, len(ua)))); fo = np.array([fold[x] for x in auth])
p = np.zeros(len(c))
for f in range(5):
    p[fo == f] = fit(X[fo != f], c[fo != f]).predict_proba((X[fo == f] - mu) / sg)[:, 1]
gconf, _ = S.crossfit(s, d > 0, auth)
print("development pairs (selector cross-fitted over accounts)")
res["dev"] = {}
table("gap", d, auth, c, gconf, np.abs(s), res["dev"]); table("selector", d, auth, c, p, p, res["dev"])
res["dev"]["selector_minus_gap_surest40_2x"] = paired(res["dev"], d, auth, c)
model = fit(X, c)
res["coefficients_standardised"] = dict(zip(FN, model.coef_[0].tolist()))
print("  coefficients (standardised):", {k: round(v, 3) for k, v in res["coefficients_standardised"].items()})
d, s, Xb, auth, lmin, c = prep(*load_bench())
pb = model.predict_proba((Xb - mu) / sg)[:, 1]
gconf, _ = S.crossfit(s, d > 0, auth)
print("benchmark pairs (selector fitted on development pairs, applied once)")
res["bench"] = {}
table("gap", d, auth, c, gconf, np.abs(s), res["bench"]); table("selector", d, auth, c, pb, pb, res["bench"])
res["bench"]["selector_minus_gap_surest40_2x"] = paired(res["bench"], d, auth, c)
res["bench"]["by_min_likes"] = {}
for lo, hi in ((0, 3), (3, 10 ** 9)):
    m = (lmin >= lo) & (lmin < hi) & (np.abs(d) >= np.log(2))
    r = {"pairs": int(m.sum()), "accuracy": float(c[m].mean())}
    for nm, cf in (("gap", gconf), ("selector", pb)):
        k = cf[m] >= 0.6
        r[nm] = {"confident_share": float(k.mean()), "confident_accuracy": float(c[m][k].mean()) if k.any() else None}
    res["bench"]["by_min_likes"][f"{lo}+" if hi > 1e6 else f"{lo}-{hi - 1}"] = r
    print(f"  2x pairs, lower like count in [{lo}, {hi if hi < 1e6 else 'inf'}): n={r['pairs']} acc={100 * r['accuracy']:.1f}; gap rule {100 * r['gap']['confident_share']:.0f}% at {100 * (r['gap']['confident_accuracy'] or 0):.1f}; selector {100 * r['selector']['confident_share']:.0f}% at {100 * (r['selector']['confident_accuracy'] or 0):.1f}")
clean = lambda o: {k: clean(v) for k, v in o.items() if k != "_m"} if isinstance(o, dict) else o  # noqa: E731
json.dump(clean(res), open(a.out, "w"), indent=1, default=float)
