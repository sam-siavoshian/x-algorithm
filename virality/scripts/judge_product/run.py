#!/usr/bin/env python3
"""Run the product's judge (xviral/judge.py) over a set of pairs, or submit the same requests to OpenAI's batch interface.

    python scripts/judge_product/run.py sync   --set dev --config n30 --model openai/gpt-oss-120b [--half A] [--every 4] [--limit 50] [--keys file.json]
    python scripts/judge_product/run.py submit --set launchpre --config n60 --model gpt-6.1-sol --keys data/llm/sample_launchpre.json
    python scripts/judge_product/run.py collect

Configs: n0, n10, n30, n60, n100 (posts of history), suffix _ev (the measured launch evidence stated in the prompt).
Results append to data/llm/out/<set>__<config>__<model>__<mode>.jsonl, one line per request (resumable), in the format
of scripts/llm_judge/judge.py, so scripts/llm_judge/lib.load_judge reads both."""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_judge"))
from lib import half_of
from xviral import judge as J

O = "data/llm/out/"
REG = "data/llm/batches.json"
H = {"Authorization": "Bearer proxy-injected"}
API = "https://api.openai.com/v1"


def messages(item: dict, config: str, order: int, mode: str) -> list[dict]:
    parts = config.split("_")
    n = int(parts[0][1:])
    a, b = (item["a"], item["b"]) if order == 0 else (item["b"], item["a"])
    acct = {"name": item.get("name"), "handle": item.get("handle"), "followers": item.get("followers")}
    return J.build_messages(acct, item["history"][-n:] if n else [], a, b, mode, mask="m" in parts[1:], note=J.evidence_note() if "ev" in parts[1:] else None)


def out_path(a, jd) -> str:
    return f"{O}{a.set}__{a.config}__{a.model.replace('/', '--')}__{'logprob' if jd.mode == 'logprob' else 'json-' + str(jd.reasoning_effort or 'default')}.jsonl"


def select(a) -> list[dict]:
    items = [json.loads(l) for l in open(f"data/llm/items_{a.set}.jsonl")]
    if a.keys:
        keep = {tuple(k) for k in json.load(open(a.keys))}
        items = [x for x in items if (x["post_id"], x["post_id_b"]) in keep]
    if a.half:
        items = [x for x in items if half_of(x["author_id"]) == a.half]
    items = items[a.offset::a.every]
    return items[:a.limit] if a.limit else items


def todo_of(a, path, items):
    done = {(r["post_id"], r["post_id_b"], r["order"]) for r in map(json.loads, open(path)) if r.get("p") is not None} if os.path.exists(path) else set()
    pend = {tuple(k) for b in (json.load(open(REG)) if os.path.exists(REG) else []) if b["path"] == path and not b.get("collected") for k in b["keys"]}
    return [(x, o) for x in items for o in (0, 1) if (x["post_id"], x["post_id_b"], o) not in done and (x["post_id"], x["post_id_b"], o) not in pend]


def sync(a):
    jd = J.Judge(a.model, endpoint=a.endpoint, mode=a.mode, reasoning_effort=a.effort if a.effort else "default", timeout=a.timeout, retries=4, concurrency=a.concurrency)
    path = out_path(a, jd)
    todo = todo_of(a, path, select(a))
    chars = sum(len(json.dumps(messages(x, a.config, o, jd.mode))) for x, o in todo)
    print(f"{len(todo)} requests, about {chars / 4.2 / 1e6:.2f}M input tokens, about ${chars / 4.2 / 1e6 * jd.price[0]:.2f} for input", flush=True)
    if a.dry or not todo:
        return
    os.makedirs(O, exist_ok=True)
    out, t0, n = open(path, "a"), time.time(), 0

    gap, nxt, lk = (60.0 / a.rpm if a.rpm else 0.0), [time.time()], __import__("threading").Lock()

    def one(xo):
        x, o = xo
        if gap:                              # a key limited to a number of requests per minute: space the calls
            with lk:
                w = nxt[0] - time.time(); nxt[0] = max(nxt[0], time.time()) + gap
            if w > 0:
                time.sleep(w)
        r = jd.ask(messages(x, a.config, o, jd.mode))
        return x, o, r
    from concurrent.futures import as_completed
    with ThreadPoolExecutor(a.concurrency) as ex:
        for fut in as_completed([ex.submit(one, t) for t in todo]):          # written as each call returns: one slow call holds nothing back
            x, o, r = fut.result()
            out.write(json.dumps({"post_id": x["post_id"], "post_id_b": x["post_id_b"], "order": o, "p": r["p"], "in": r["tokens_in"], "cached": r["tokens_cached"], "out": r["tokens_out"],
                                  "err": r["error"], "s": round(r["seconds"], 2)}) + "\n"); out.flush()
            n += 1
            if n % 500 == 0:
                print(n, jd.spend, f"{time.time() - t0:.0f}s", flush=True)
    rec = {"path": path, "n": jd.spend["requests"], "fail": jd.spend["errors"], "in": jd.spend["tokens_in"], "cached": jd.spend["tokens_cached"], "out": jd.spend["tokens_out"], "usd": jd.spend["usd"],
           "model": a.model, "endpoint": jd.endpoint, "time": time.time(), "seconds": round(time.time() - t0)}
    print(json.dumps(rec))
    open("data/llm/spend.jsonl", "a").write(json.dumps(rec) + "\n")


def submit(a):
    jd = J.Judge(a.model, endpoint="openai", mode=a.mode, reasoning_effort=a.effort if a.effort else "default")
    path = out_path(a, jd)
    todo = todo_of(a, path, select(a))
    lines = [json.dumps({"custom_id": f"{x['post_id']}|{x['post_id_b']}|{o}", "method": "POST", "url": "/v1/chat/completions", "body": jd.body(messages(x, a.config, o, jd.mode))}) for x, o in todo]
    chars = sum(len(l) for l in lines)
    print(f"{len(todo)} requests ({len(todo) // 2} pairs), about {chars / 4.2 / 1e6:.2f}M input tokens, about ${chars / 4.2 / 1e6 * jd.price[0] / 2 + len(todo) * 45 * jd.price[2] / 2e6:.2f} at batch price")
    if a.dry or not todo:
        return
    fn = f"data/llm/batch_{int(time.time())}.jsonl"
    open(fn, "w").write("\n".join(lines) + "\n")
    f = requests.post(f"{API}/files", headers=H, files={"file": (os.path.basename(fn), open(fn, "rb"))}, data={"purpose": "batch"}, timeout=600).json()
    b = requests.post(f"{API}/batches", headers=H, json={"input_file_id": f["id"], "endpoint": "/v1/chat/completions", "completion_window": "24h"}, timeout=120).json()
    print(json.dumps(b)[:200])
    r = json.load(open(REG)) if os.path.exists(REG) else []
    r.append({"id": b["id"], "path": path, "model": a.model, "keys": [[x["post_id"], x["post_id_b"], o] for x, o in todo], "submitted": time.time()})
    json.dump(r, open(REG, "w"))


def collect(a):
    r = json.load(open(REG))
    for b in r:
        if b.get("collected"):
            continue
        s = requests.get(f"{API}/batches/{b['id']}", headers=H, timeout=120).json()
        print(b["id"][-8:], b["path"].split("/")[-1], s.get("status"), s.get("request_counts"), f"{(time.time() - b['submitted']) / 60:.0f} min", flush=True)
        if a.cancel_after and s.get("status") == "in_progress" and (time.time() - b["submitted"]) / 60 > a.cancel_after and (s.get("request_counts") or {}).get("completed", 0) >= a.cancel_share * (s.get("request_counts") or {}).get("total", 1):
            print("  cancelling:", requests.post(f"{API}/batches/{b['id']}/cancel", headers=H, timeout=120).json().get("status"))
            continue
        if s.get("status") not in ("completed", "expired", "cancelled", "failed"):
            continue
        tot = {"in": 0, "cached": 0, "out": 0, "n": 0, "fail": 0}
        if s.get("output_file_id"):
            txt = requests.get(f"{API}/files/{s['output_file_id']}/content", headers=H, timeout=600).text
            with open(b["path"], "a") as out:
                for l in txt.splitlines():
                    j = json.loads(l); pid, pidb, o = j["custom_id"].split("|"); body = (j.get("response") or {}).get("body") or {}; u = body.get("usage") or {}
                    p = J.parse_json(body) if body.get("choices") else None
                    c = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                    tot["n"] += 1; tot["fail"] += p is None; tot["in"] += u.get("prompt_tokens", 0); tot["out"] += u.get("completion_tokens", 0); tot["cached"] += c
                    out.write(json.dumps({"post_id": int(pid), "post_id_b": int(pidb), "order": int(o), "p": p, "in": u.get("prompt_tokens"), "cached": c, "out": u.get("completion_tokens"), "batch": True}) + "\n")
        pi, pc, po = J.MODELS[b["model"]]["price"]
        usd = ((tot["in"] - tot["cached"]) * pi + tot["cached"] * pc + tot["out"] * po) / 2e6
        b.update(collected=True, status=s.get("status"), usage=tot, usd=usd)
        print("  collected", tot, f"${usd:.2f}")
        open("data/llm/spend.jsonl", "a").write(json.dumps({"path": b["path"], **tot, "usd": usd, "model": b["model"], "endpoint": "openai", "batch": True, "time": time.time()}) + "\n")
    json.dump(r, open(REG, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sync", "submit", "collect"])
    ap.add_argument("--set"); ap.add_argument("--config", default="n30"); ap.add_argument("--model", default="gpt-6.1-sol"); ap.add_argument("--endpoint"); ap.add_argument("--mode"); ap.add_argument("--effort")
    ap.add_argument("--half"); ap.add_argument("--every", type=int, default=1); ap.add_argument("--offset", type=int, default=0); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--keys")
    ap.add_argument("--concurrency", type=int, default=24); ap.add_argument("--timeout", type=float, default=180); ap.add_argument("--dry", action="store_true"); ap.add_argument("--rpm", type=float, default=0, help="sync: at most this many requests a minute")
    ap.add_argument("--cancel-after", type=float, default=0, help="collect: cancel a batch older than this many minutes once --cancel-share of it is done (finished requests are kept)")
    ap.add_argument("--cancel-share", type=float, default=0.9)
    a = ap.parse_args()
    {"sync": sync, "submit": submit, "collect": collect}[a.cmd](a)
