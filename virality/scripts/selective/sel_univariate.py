#!/usr/bin/env python3
"""Selective track: which inputs say anything about "the call is right" beyond the size of the gap? Training pairs only.
For each feature: logistic regression of correct on [gap, feature] (standardised), the feature's coefficient with a 95% interval
from resampling accounts, on all pairs and on pairs 2x apart.   python scripts/selective/sel_univariate.py dev_ft10 launch_pre_start10"""
import json, sys
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
out = {}
for name in sys.argv[1:]:
    F = pl.read_parquet(f"data/sel/feat_{name}.parquet")
    y, d, auth = F["correct"].to_numpy(), F["d"].to_numpy(), F["author_id"].to_numpy()
    ua, inv = np.unique(auth, return_inverse=True); mem = [np.flatnonzero(inv == j) for j in range(len(ua))]
    rng = np.random.default_rng(0); draws = [np.concatenate([mem[j] for j in rng.integers(0, len(ua), len(ua))]) for _ in range(40)]
    g = F["f_gap"].to_numpy(); rows = []
    for c in [c for c in F.columns if c.startswith("f_") and c != "f_gap"]:
        x = F[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
        if np.all(np.isnan(x)) or np.nanstd(x) == 0: continue
        x = np.where(np.isnan(x), np.nanmedian(x), x); x = (x - x.mean()) / x.std()
        for lab, k in (("all", np.ones(len(y), bool)), ("2x", np.abs(d) >= np.log(2))):
            A = np.column_stack([g / g.std(), x])
            fit = lambda i: LogisticRegression(C=100).fit(A[i], y[i]).coef_[0, 1]      # noqa: E731
            idx = np.flatnonzero(k); b = fit(idx)
            bs = [fit(i[k[i]]) for i in draws]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append({"feature": c[2:], "pairs": lab, "coef": float(b), "lo": float(lo), "hi": float(hi)})
    out[name] = rows
    R = pl.DataFrame(rows).filter(pl.col("pairs") == "2x").with_columns((pl.col("coef").abs() / ((pl.col("hi") - pl.col("lo")) / 3.92)).alias("z")).sort("z", descending=True)
    print(name, "| pairs 2x apart | features whose interval excludes zero:", R.filter((pl.col("lo") > 0) | (pl.col("hi") < 0)).height, "of", R.height)
    with pl.Config(tbl_rows=14): print(R.head(14))
json.dump(out, open("data/sel/univariate.json", "w"), indent=1)
