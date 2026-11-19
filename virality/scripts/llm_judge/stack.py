#!/usr/bin/env python3
"""Step 4, light head on the judge's own outputs: average the log-odds of several cheap luna runs (different history / masking).
Chosen on development pairs; the benchmark has three of the runs (n30, n30_m, n0_m)."""
import json, sys, numpy as np, polars as pl
sys.path.insert(0, "scripts/llm_judge")
from lib import *
out = {}
for name, cfgs in (("dev", ["n30", "n30_m", "n0_m", "n0", "n10", "bw30"]), ("bench", ["n30", "n30_m", "n0_m"])):
    P = pairs(name).filter(pl.col("d").abs() >= L125)
    for c in cfgs:
        P = P.join(load_judge(f"data/llm/out/{name}__{c}__gpt-6-luna__logprob.jsonl").select("post_id", "post_id_b", pl.col("j").alias(c)), on=["post_id", "post_id_b"], how="left").with_columns(pl.col(c).fill_null(0.0))
    d, a = P["d"].to_numpy(), P["author_id"].to_numpy(); big = np.abs(d) >= L2
    sets = {"n30": ["n30"], "mean of n30, n30_m, n0_m": ["n30", "n30_m", "n0_m"]} | ({"mean of all six": cfgs} if name == "dev" else {})
    C = {}
    for n, cs in sets.items():
        s = np.mean([P[c].to_numpy() for c in cs], 0); c_ = np.where(s == 0, .5, (s * d) > 0).astype(float); C[n] = c_
        idx = np.flatnonzero(big); top = idx[np.argsort(-np.abs(s[idx]), kind="stable")[: int(.4 * len(idx))]]
        out[f"{name}: {n}"] = {"pairs": int(len(d)), "acc": float(c_.mean()), "ci": ci(a, lambda i: c_[i].mean()), "acc_2x": float(c_[big].mean()), "surest40_2x": float(c_[top].mean())}
        print(name, n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[f"{name}: {n}"].items()})
    for n in list(sets)[1:]:
        dd = C[n] - C["n30"]; out[f"{name}: {n} minus n30"] = [float(dd.mean())] + ci(a, lambda i: dd[i].mean()); print("  ", n, "minus n30", np.round(out[f"{name}: {n} minus n30"], 4))
json.dump(out, open("data/llm/stack.json", "w"), indent=1)
