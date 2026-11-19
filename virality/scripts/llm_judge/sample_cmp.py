#!/usr/bin/env python3
"""Same development sample (half A, every 4th pair, n30 history): luna log-probabilities, luna with reasoning, stronger model."""
import json, sys, numpy as np, polars as pl
sys.path.insert(0, "scripts/llm_judge")
from lib import *
P = pairs("dev")
runs = {"luna logprob (no reasoning)": "dev__n30__gpt-6-luna__logprob", "luna json, reasoning low": "dev__n30__gpt-6-luna__json-low", "luna json, reasoning medium": "dev__n30__gpt-6-luna__json-medium", "gpt-6.1-sol json, reasoning low": "dev__n30__gpt-6.1-sol__json-low"}
key = load_judge("data/llm/out/dev__n30__gpt-6.1-sol__json-low.jsonl").select("post_id", "post_id_b")
out = {}
for n, f in runs.items():
    J = P.join(key, on=["post_id", "post_id_b"]).join(load_judge(f"data/llm/out/{f}.jsonl"), on=["post_id", "post_id_b"]).sort("post_id", "post_id_b")
    d, a, j = J["d"].to_numpy(), J["author_id"].to_numpy(), J["j"].to_numpy()
    c = np.where(j == 0, 0.5, (j * d) > 0)
    big = np.abs(d) >= L2
    out[n] = {"pairs": J.height, "acc": float(c.mean()), "ci": ci(a, lambda i: c[i].mean()), "acc_2x": float(c[big].mean()), "ties": float((j == 0).mean()), "p_first": float(J["pos"].mean()), "surrogate": float(((J["s3"].to_numpy() * d) > 0).mean())}
    out[n]["_c"], out[n]["_a"] = c, a
    print(n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[n].items() if not k.startswith("_")})
b = out["luna logprob (no reasoning)"]
for n in list(runs)[1:]:
    if out[n]["pairs"] == b["pairs"]:
        dd = out[n]["_c"] - b["_c"]
        print(n, "minus luna logprob:", round(100 * dd.mean(), 1), [round(100 * x, 1) for x in ci(b["_a"], lambda i: dd[i].mean())])
        out[n]["minus_luna_logprob"] = [float(dd.mean())] + ci(b["_a"], lambda i: dd[i].mean())
json.dump({k: {a: b for a, b in v.items() if not a.startswith("_")} for k, v in out.items()}, open("data/llm/sample_cmp.json", "w"), indent=1)
