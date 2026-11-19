#!/usr/bin/env python3
"""Compare judge configurations on the development pairs: half A (prompt development) and half B (validation)."""
import glob, json, sys, numpy as np
sys.path.insert(0, "scripts/llm_judge")
from lib import *
P = pairs("dev")
res = {}
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "data/llm/out/dev__*.jsonl")):
    J = P.join(load_judge(f), on=["post_id", "post_id_b"])
    d, a, j, s3 = J["d"].to_numpy(), J["author_id"].to_numpy(), J["j"].to_numpy(), J["s3"].to_numpy()
    r = {"pairs": J.height, "position_bias_p_first": float(J["pos"].mean()), "one_order_acc": float((((J["j0"].to_numpy() * d) > 0).mean() + ((J["j1"].to_numpy() * d) > 0).mean()) / 2)}
    for h in ("A", "B", "all"):
        m = np.ones(len(d), bool) if h == "all" else (J["half"].to_numpy() == h)
        big = np.abs(d) >= L2
        r[h] = {"judge": acc(j, d, a, m), "judge_2x": acc(j, d, a, m & big), "surrogate": acc(s3, d, a, m), "surrogate_2x": acc(s3, d, a, m & big),
                "agree_with_surrogate": float(((j * s3) > 0)[m].mean()), "corr": float(np.corrcoef(j[m], s3[m])[0, 1]),
                "later_posted_preferred": float(((j > 0) == (J["t_a"].to_numpy() > J["t_b"].to_numpy()))[m].mean())}
    res[f.split("/")[-1][:-6]] = r
    g = lambda x: f"{100 * x['acc']:.1f} ({100 * x['ci'][0]:.1f}-{100 * x['ci'][1]:.1f}) n={x['pairs']}"
    print(f.split("/")[-1][5:-6], "| A:", g(r["A"]["judge"]), "2x", g(r["A"]["judge_2x"]), "| B:", g(r["B"]["judge"]), "2x", g(r["B"]["judge_2x"]), "| sur A/B", f"{100 * r['A']['surrogate']['acc']:.1f}/{100 * r['B']['surrogate']['acc']:.1f}",
          "| agree", round(r["all"]["agree_with_surrogate"], 3), "corr", round(r["all"]["corr"], 3), "| P(first)", round(r["position_bias_p_first"], 3), "one-order", round(r["one_order_acc"], 3), "later pref", round(r["all"]["later_posted_preferred"], 3))
json.dump(res, open("data/llm/dev_configs.json", "w"), indent=1)
