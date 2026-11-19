#!/usr/bin/env python3
"""Tokens and dollars per run, from the per-request usage stored with every judge answer (batch requests at half price)."""
import glob, json, sys
sys.path.insert(0, "scripts/llm_judge")
from judge import PRICE
rows, tot = [], 0.0
for f in sorted(glob.glob("data/llm/out/*.jsonl")):
    model = f.split("__")[2]; pi, pc, po = PRICE[model]
    a = {"run": f.split("/")[-1][:-6], "requests": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "usd": 0.0}
    for r in map(json.loads, open(f)):
        if r.get("in") is None:
            continue
        k = 0.5 if r.get("batch") else 1.0
        a["requests"] += 1; a["input_tokens"] += r["in"]; a["cached_tokens"] += r.get("cached") or 0; a["output_tokens"] += r["out"] or 0
        a["usd"] += k * ((r["in"] - (r.get("cached") or 0)) * pi + (r.get("cached") or 0) * pc + (r["out"] or 0) * po) / 1e6
    rows.append(a); tot += a["usd"]
emb = [json.loads(l) for l in open("data/llm/spend.jsonl") if "embedding" in l]
for e in emb:
    rows.append({"run": e["path"], "requests": 0, "input_tokens": e["in"], "cached_tokens": 0, "output_tokens": 0, "usd": e["usd"]}); tot += e["usd"]
probe = 0.01   # a handful of probe requests during setup (capability checks), not stored per request
rows.append({"run": "setup probes (estimate)", "requests": 5, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "usd": probe}); tot += probe
print("| Run | Requests | Input tokens | Cached | Output tokens | USD |\n|---|---|---|---|---|---|")
for a in rows:
    print(f"| {a['run']} | {a['requests']:,} | {a['input_tokens']:,} | {a['cached_tokens']:,} | {a['output_tokens']:,} | {a['usd']:.2f} |")
print(f"| **Total** | {sum(a['requests'] for a in rows):,} | {sum(a['input_tokens'] for a in rows):,} | {sum(a['cached_tokens'] for a in rows):,} | {sum(a['output_tokens'] for a in rows):,} | **{tot:.2f}** |")
json.dump({"runs": rows, "total_usd": tot}, open("data/llm/spend.json", "w"), indent=1)
