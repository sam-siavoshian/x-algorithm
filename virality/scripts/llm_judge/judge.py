#!/usr/bin/env python3
"""Language-model judge for pairs of posts by one account: probability that A got more likes than B.

    python scripts/llm_judge/judge.py --set dev --config n30 [--model gpt-6-luna] [--limit 50] [--half A|B] [--mode logprob|json]

Configs: n0 (no history), n10 / n30 (most recent earlier posts with likes), bw30 (10 best, 10 worst, 10 most recent of the
last 200), each with suffix _m for masked handles and names (no account line, @handles replaced). Every pair is asked in
both orders; the history is the shared prefix of both requests so the second is served from the prompt cache.
Mode logprob: reasoning off, one output token, P(A) from the token log-probabilities. Mode json: reasoning effort low,
structured output {"p_a": number}. Results append to data/llm/out/<set>__<config>__<model>__<mode>.jsonl (resumable);
token usage is logged per request."""
import argparse, asyncio, hashlib, json, math, os, re, time
import aiohttp

URL = "https://api.openai.com/v1/chat/completions"
PRICE = {"gpt-6-luna": (0.10, 0.01, 0.50), "gpt-6-sol": (2.0, 0.20, 10.0), "gpt-6.1-sol": (2.0, 0.10, 10.0), "gpt-6.1-terra": (1.0, 0.05, 5.0)}   # USD per 1M: input, cached input, output
HANDLE = re.compile(r"@\w+")

SYSTEM = ("You judge posts on X (Twitter). You are shown two posts, A and B, written by the same account. "
          "One of them received clearly more likes than the other. Decide which one.\n"
          "When earlier posts by this account are listed with the likes each received, use them: they show what this "
          "account's audience rewards and what it ignores. Judge the two posts as that audience would meet them in the feed: "
          "the hook of the first line, how concrete and clear the claim is, whether it stands alone without the link, format and media.\n"
          "{answer}")
ANS_LP = "Answer with a single letter, A or B, and nothing else."
ANS_JSON = 'Return JSON {{"p_a": x}} where x is your probability from 0 to 1 that post A received more likes than post B.'


def half_of(author_id: int) -> str:
    return "A" if int(hashlib.md5(str(author_id).encode()).hexdigest(), 16) % 2 == 0 else "B"


def fmt(p: dict, mask: bool, cap: int = 0) -> str:
    t = p["text"]
    if cap and len(t) > cap:
        t = t[:cap] + "…"
    if mask:
        t = HANDLE.sub("@user", t)
    tags = []
    if p["media"]:
        tags.append(f"[{p['media']} attached]")
    if p["links"]:
        tags.append("[link: " + ", ".join(p["links"]) + "]")
    if p["quoted"]:
        q = p["quoted"][:300].replace("\n", " ")
        tags.append("[quotes another post: \"" + (HANDLE.sub("@user", q) if mask else q) + "\"]")
    return (t + ("\n" if tags else "") + " ".join(tags)).strip()


def history(item: dict, config: str) -> list[dict]:
    h = item["history"]
    base = config.split("_")[0]
    if base == "n0":
        return []
    if base.startswith("n"):
        return h[-int(base[1:]):]
    if base == "bw30":
        if len(h) <= 30:
            return h
        by = sorted(h, key=lambda x: x["likes"] or 0)
        ids, pick = set(), []
        for x in by[-10:] + by[:10] + h[::-1]:
            if x["id"] not in ids and len(pick) < 30:
                ids.add(x["id"]); pick.append(x)
        return sorted(pick, key=lambda x: x["t"])
    raise ValueError(config)


def build(item: dict, config: str, order: int, mode: str) -> list[dict]:
    mask = config.endswith("_m")
    parts = []
    if not mask:
        f = item.get("followers")
        parts.append(f"Account: {item.get('name') or ''} (@{item.get('handle') or ''})" + (f", about {f:,} followers" if f else ""))
    h = history(item, config)
    if h:
        base = config.split("_")[0]
        parts.append(("Earlier posts by this account (its highest-liked, its lowest-liked and its most recent, in order of posting), with likes received:" if base == "bw30" else
                      "Earlier posts by this account, oldest first, with likes received:") + "\n" +
                     "\n".join(f"- ({x['likes']} likes) " + fmt(x, mask, 400).replace("\n", " / ") for x in h))
    a, b = (item["a"], item["b"]) if order == 0 else (item["b"], item["a"])
    parts.append(f"Post A:\n{fmt(a, mask)}\n\nPost B:\n{fmt(b, mask)}\n\nWhich post received more likes?")
    return [{"role": "system", "content": SYSTEM.format(answer=ANS_LP if mode == "logprob" else ANS_JSON)}, {"role": "user", "content": "\n\n".join(parts)}]


def parse_lp(js: dict):
    c = js["choices"][0]
    top = c["logprobs"]["content"][0]["top_logprobs"]
    pa = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().upper() == "A")
    pb = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().upper() == "B")
    if pa + pb < 0.5:
        return None
    return max(pa, 1e-7) / (max(pa, 1e-7) + max(pb, 1e-7))


async def ask(sess, sem, model, messages, mode, effort):
    body = {"model": model, "messages": messages}
    if mode == "logprob":
        body |= {"reasoning_effort": "none", "logprobs": True, "top_logprobs": 5, "max_completion_tokens": 4}
    else:
        body |= {"reasoning_effort": effort, "max_completion_tokens": 3000,
                 "response_format": {"type": "json_schema", "json_schema": {"name": "judgement", "strict": True, "schema": {"type": "object", "properties": {"p_a": {"type": "number"}}, "required": ["p_a"], "additionalProperties": False}}}}
    for attempt in range(6):
        try:
            async with sem:
                async with sess.post(URL, json=body, headers={"Authorization": "Bearer proxy-injected"}, timeout=aiohttp.ClientTimeout(total=180)) as r:
                    js = await r.json()
                    if r.status == 200:
                        p = parse_lp(js) if mode == "logprob" else min(1.0, max(0.0, float(json.loads(js["choices"][0]["message"]["content"])["p_a"])))
                        return p, js.get("usage", {})
                    if r.status not in (429, 500, 502, 503, 504):
                        return None, {"error": json.dumps(js)[:300]}
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:200]
        await asyncio.sleep(2 ** attempt)
    return None, {"error": "retries"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--mode", default="logprob")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--half", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="take every k-th pair (after the half filter)")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    items = [json.loads(l) for l in open(f"data/llm/items_{args.set}.jsonl")]
    if args.half:
        items = [x for x in items if half_of(x["author_id"]) == args.half]
    items = items[::args.every]
    if args.limit:
        items = items[:args.limit]
    os.makedirs("data/llm/out", exist_ok=True)
    path = f"data/llm/out/{args.set}__{args.config}__{args.model}__{args.mode}{'-' + args.effort if args.mode == 'json' else ''}.jsonl"
    done = set()
    if os.path.exists(path):
        done = {(r["post_id"], r["post_id_b"], r["order"]) for r in map(json.loads, open(path)) if r.get("p") is not None}
    todo = [(x, o) for x in items for o in (0, 1) if (x["post_id"], x["post_id_b"], o) not in done]
    if args.dry:
        n = sum(len(json.dumps(build(x, args.config, o, args.mode))) for x, o in todo)
        print(f"{len(todo)} requests, about {n / 4 / 1e6:.2f}M input tokens (chars/4)"); print(build(todo[0][0], args.config, todo[0][1], args.mode)[1]["content"][-1500:]); return
    sem, lock, t0 = asyncio.Semaphore(args.concurrency), asyncio.Lock(), time.time()
    tot = {"in": 0, "cached": 0, "out": 0, "n": 0, "fail": 0}
    out = open(path, "a")

    async def one(sess, x, o, first: asyncio.Event | None, wait: asyncio.Event | None):
        if wait is not None:
            await wait.wait()                      # second order waits for the first, so the shared prefix is cached
        p, u = await ask(sess, sem, args.model, build(x, args.config, o, args.mode), args.mode, args.effort)
        if first is not None:
            first.set()
        async with lock:
            tot["n"] += 1; tot["fail"] += p is None
            tot["in"] += u.get("prompt_tokens", 0); tot["cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0); tot["out"] += u.get("completion_tokens", 0)
            out.write(json.dumps({"post_id": x["post_id"], "post_id_b": x["post_id_b"], "order": o, "p": p, "in": u.get("prompt_tokens"), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                                  "out": u.get("completion_tokens"), "err": u.get("error")}) + "\n"); out.flush()
            if tot["n"] % 500 == 0:
                print(tot, f"{time.time() - t0:.0f}s", flush=True)
    async with aiohttp.ClientSession(trust_env=True, connector=aiohttp.TCPConnector(limit=args.concurrency + 4)) as sess:
        tasks, by = [], {}
        for x, o in todo:
            by.setdefault((x["post_id"], x["post_id_b"]), []).append((x, o))
        for lst in by.values():
            ev = asyncio.Event() if len(lst) == 2 else None
            for i, (x, o) in enumerate(lst):
                tasks.append(one(sess, x, o, ev if i == 0 else None, ev if i == 1 else None))
        await asyncio.gather(*tasks)
    pi, pc, po = PRICE.get(args.model, (0, 0, 0))
    cost = ((tot["in"] - tot["cached"]) * pi + tot["cached"] * pc + tot["out"] * po) / 1e6
    print(json.dumps({"path": path, **tot, "usd": round(cost, 4), "seconds": round(time.time() - t0)}))
    with open("data/llm/spend.jsonl", "a") as f:
        f.write(json.dumps({"path": path, **tot, "usd": cost, "model": args.model, "time": time.time()}) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
