#!/usr/bin/env python3
"""Where does the judge's edge sit? Pairs split by how similar the two texts are (difflib ratio): near-identical pairs are
mostly a post deleted and reposted with a correction."""
import difflib, json, sys, numpy as np, polars as pl
sys.path.insert(0, "scripts/llm_judge")
from lib import *
out = {}
for name in ("dev", "bench", "launch"):
    S = pl.read_parquet(f"data/llm/scores_{name}.parquet")
    sim = {(x["post_id"], x["post_id_b"]): difflib.SequenceMatcher(None, x["a"]["text"], x["b"]["text"]).ratio() for x in map(json.loads, open(f"data/llm/items_{name}.jsonl"))}
    r = np.array([sim[(a, b)] for a, b in zip(S["post_id"].to_list(), S["post_id_b"].to_list())])
    S.select("post_id", "post_id_b").with_columns(pl.Series("text_similarity", r)).write_parquet(f"data/llm/textsim_{name}.parquet")
    d, a = S["d"].to_numpy(), S["author_id"].to_numpy(); t0 = np.minimum(S["t_a"].to_numpy(), S["t_b"].to_numpy())
    later_b = S["t_b"].to_numpy() > S["t_a"].to_numpy()
    for lab, m in (("similarity 0.9 and over", r >= .9), ("0.5 to 0.9", (r >= .5) & (r < .9)), ("under 0.5", r < .5), ("under 0.9, after 2026-05-18", (r < .9) & (t0 >= 1779062400000)), ("0.9 and over, after 2026-05-18", (r >= .9) & (t0 >= 1779062400000))):
        if m.sum() < 30:
            continue
        row = {"pairs": int(m.sum()), "accounts": int(len(set(a[m]))), "later_post_won": float(((d > 0) != later_b)[m].mean())}
        for cn in ("s3", "luna", "sol", "blend_s3_luna"):
            if cn not in S.columns:
                continue
            s = S[cn].to_numpy().astype(float); mm = m & ~np.isnan(s)
            if mm.sum() < 30:
                continue
            c = np.where(s == 0, .5, (s * d) > 0).astype(float)
            row[cn] = {"pairs": int(mm.sum()), "acc": float(c[mm].mean()), "ci": ci(a, lambda i: c[i][mm[i]].mean() if mm[i].sum() else np.nan, n=500)}
        out[f"{name}: {lab}"] = row
        print(name, lab, row["pairs"], "later won", round(row["later_post_won"], 3), {k: (round(v["acc"], 3), [round(x, 3) for x in v["ci"]], v["pairs"]) for k, v in row.items() if isinstance(v, dict)})
json.dump(out, open("data/llm/near_dup.json", "w"), indent=1)
