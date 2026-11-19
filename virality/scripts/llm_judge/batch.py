#!/usr/bin/env python3
"""The same judge requests through OpenAI's batch interface (half price, asynchronous).

    python scripts/llm_judge/batch.py submit --set bench --config n30 --model gpt-6.1-sol [--half B] [--every 4] [--clear] [--sample 1900]
    python scripts/llm_judge/batch.py collect            # polls every open batch in data/llm/batches.json, appends finished results

Results land in the same data/llm/out/<set>__<config>__<model>__json-<effort>.jsonl files judge.py writes."""
import argparse, json, os, sys, time, numpy as np, requests
sys.path.insert(0, os.path.dirname(__file__))
from judge import build, half_of, PRICE

H = {"Authorization": "Bearer proxy-injected"}
API = "https://api.openai.com/v1"
REG = "data/llm/batches.json"


def reg():
    return json.load(open(REG)) if os.path.exists(REG) else []


def submit(a):
    items = [json.loads(l) for l in open(f"data/llm/items_{a.set}.jsonl")]
    if a.half:
        items = [x for x in items if half_of(x["author_id"]) == a.half]
    if a.clear:
        items = [x for x in items if abs(np.log1p(x["a"]["likes"]) - np.log1p(x["b"]["likes"])) >= np.log(1.25)]
    items = items[a.offset::a.every]
    if a.sample:
        rng = np.random.default_rng(a.seed)
        items = [items[i] for i in sorted(rng.choice(len(items), min(a.sample, len(items)), replace=False))]
    path = f"data/llm/out/{a.set}__{a.config}__{a.model}__json-{a.effort}.jsonl"
    done = {(r["post_id"], r["post_id_b"], r["order"]) for r in map(json.loads, open(path)) if r.get("p") is not None} if os.path.exists(path) else set()
    pending = {tuple(k) for b in reg() if b["path"] == path and not b.get("collected") for k in b["keys"]}
    todo = [(x, o) for x in items for o in (0, 1) if (x["post_id"], x["post_id_b"], o) not in done and (x["post_id"], x["post_id_b"], o) not in pending]
    schema = {"type": "json_schema", "json_schema": {"name": "judgement", "strict": True, "schema": {"type": "object", "properties": {"p_a": {"type": "number"}}, "required": ["p_a"], "additionalProperties": False}}}
    lines = [json.dumps({"custom_id": f"{x['post_id']}|{x['post_id_b']}|{o}", "method": "POST", "url": "/v1/chat/completions",
                         "body": {"model": a.model, "messages": build(x, a.config, o, "json"), "reasoning_effort": a.effort, "max_completion_tokens": 3000, "response_format": schema}}) for x, o in todo]
    chars = sum(len(l) for l in lines)
    print(f"{len(todo)} requests ({len(todo) // 2} pairs), about {chars / 4.2 / 1e6:.2f}M input tokens, about ${chars / 4.2 / 1e6 * PRICE[a.model][0] / 2 + len(todo) * 45 * PRICE[a.model][2] / 2e6:.2f} at batch price")
    if a.dry or not todo:
        return
    fn = f"data/llm/batch_{int(time.time())}.jsonl"
    open(fn, "w").write("\n".join(lines) + "\n")
    f = requests.post(f"{API}/files", headers=H, files={"file": (os.path.basename(fn), open(fn, "rb"))}, data={"purpose": "batch"}, timeout=600).json()
    b = requests.post(f"{API}/batches", headers=H, json={"input_file_id": f["id"], "endpoint": "/v1/chat/completions", "completion_window": "24h"}, timeout=120).json()
    print(json.dumps(b)[:400])
    r = reg(); r.append({"id": b["id"], "path": path, "model": a.model, "keys": [[x["post_id"], x["post_id_b"], o] for x, o in todo], "submitted": time.time()})
    json.dump(r, open(REG, "w"))


def collect(a):
    r = reg()
    for b in r:
        if b.get("collected"):
            continue
        s = requests.get(f"{API}/batches/{b['id']}", headers=H, timeout=120).json()
        print(b["id"], b["path"].split("/")[-1], s.get("status"), s.get("request_counts"), f"{(time.time() - b['submitted']) / 60:.0f} min")
        if s.get("status") not in ("completed", "expired", "cancelled", "failed"):
            continue
        tot = {"in": 0, "cached": 0, "out": 0, "n": 0, "fail": 0}
        if s.get("output_file_id"):
            txt = requests.get(f"{API}/files/{s['output_file_id']}/content", headers=H, timeout=600).text
            with open(b["path"], "a") as out:
                for l in txt.splitlines():
                    j = json.loads(l); pid, pidb, o = j["custom_id"].split("|"); body = (j.get("response") or {}).get("body") or {}; u = body.get("usage") or {}
                    try:
                        p = min(1.0, max(0.0, float(json.loads(body["choices"][0]["message"]["content"])["p_a"])))
                    except Exception:  # noqa: BLE001
                        p = None
                    tot["n"] += 1; tot["fail"] += p is None; tot["in"] += u.get("prompt_tokens", 0); tot["out"] += u.get("completion_tokens", 0); tot["cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                    out.write(json.dumps({"post_id": int(pid), "post_id_b": int(pidb), "order": int(o), "p": p, "in": u.get("prompt_tokens"), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "out": u.get("completion_tokens"), "batch": True}) + "\n")
        pi, pc, po = PRICE[b["model"]]
        usd = ((tot["in"] - tot["cached"]) * pi + tot["cached"] * pc + tot["out"] * po) / 2e6
        b["collected"] = True; b["status"] = s.get("status"); b["usage"] = tot; b["usd"] = usd
        print("  collected", tot, f"${usd:.2f}")
        open("data/llm/spend.jsonl", "a").write(json.dumps({"path": b["path"], **tot, "usd": usd, "model": b["model"], "batch": True, "time": time.time()}) + "\n")
    json.dump(r, open(REG, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--set"); ap.add_argument("--config", default="n30"); ap.add_argument("--model", default="gpt-6.1-sol"); ap.add_argument("--effort", default="low")
    ap.add_argument("--half"); ap.add_argument("--every", type=int, default=1); ap.add_argument("--offset", type=int, default=0); ap.add_argument("--sample", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clear", action="store_true"); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    (submit if a.cmd == "submit" else collect)(a)
