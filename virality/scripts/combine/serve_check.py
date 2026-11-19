#!/usr/bin/env python3
"""A real run of the production bundle through the HTTP layer: latency and cost of a first and a repeat request.

    python scripts/combine/serve_check.py --bundle data/combine/bundle [--judge gpt-6.1-sol] [--seed 11] --out data/combine/serve_check

Three accounts drawn with a fixed seed from the launch test pairs (real gap 2x or more, 30 or more earlier posts, one pair
per account). For each: the account's public posts from before the earlier of two real launch posts are the history
(`past_posts` with `share_past_posts`: they are public corpus posts), the two launch posts are versions 1 and 2.
  first    the account has never been seen by this process: history and versions are embedded
  repeat   the same request again: nothing is embedded
  edited   the same account with one sentence added to version 2: one text is embedded
  six      six versions for the first account
  no_perm  the same request without share_past_posts: answered by the fallback models, nothing is sent
Everything sent is a public post already in the corpus. Three accounts show that the path runs and what it costs; they
measure no accuracy.
"""
import argparse, json, os, time
import numpy as np, polars as pl
from fastapi.testclient import TestClient

from xviral.app import server, service

ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True); ap.add_argument("--judge", default=None); ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--out", default="data/combine/serve_check"); ap.add_argument("--threads", type=int, default=4); a = ap.parse_args()
import torch
torch.set_num_threads(a.threads)
iso = lambda ms: service._iso(int(ms))      # noqa: E731


def past(rows):
    out = []
    for r in rows.iter_rows(named=True):
        k = {m.get("type") for m in (r["media"] or [])}
        out.append({"text": r["text"], "posted_at": iso(r["created_at_ms"]), "likes": r["likes"], "views": r["views"], "replies": r["replies"], "reposts": r["reposts"],
                    "image": "photo" in k, "video": bool(k & {"video", "animated_gif"})})
    return out


def version(r):
    k = {m.get("type") for m in (r["media"] or [])}
    return {"text": r["text"], "image": "photo" in k and not k & {"video", "animated_gif"}, "video": bool(k & {"video", "animated_gif"})}


P = pl.read_parquet("data/tracks/judge_product/jp_pairs_launch.parquet").filter((pl.col("d").abs() >= np.log(2)) & (pl.col("n_hist") >= 30) & pl.col("followers").is_not_null())
P = P.sample(fraction=1.0, shuffle=True, seed=a.seed).unique("author_id", keep="first", maintain_order=True).head(3)
posts = pl.scan_parquet("data/work/filtered/posts.parquet").filter(pl.col("author_id").is_in(P["author_id"].implode())).collect().filter(~pl.col("is_reply").fill_null(False) & pl.col("text").is_not_null())
judge = None
if a.judge:
    from xviral.judge import Judge
    judge = Judge(a.judge, timeout=45, retries=1)
t0 = time.time()
svc = service.load_bundle(a.bundle, judge=judge)
load_s = time.time() - t0
client = TestClient(server.create_app(svc, settings=server.Settings(rate_per_minute=0)))
os.makedirs(a.out, exist_ok=True)
rows, info = [], svc.embedding_info()
print(f"bundle loaded in {load_s:.1f}s: {svc.model_set}, encoder {info}, fallback {svc.fallback.model_set if svc.fallback else None}, judge {a.judge}", flush=True)


def call(body, tag):
    t = time.perf_counter(); r = client.post("/api/analyze", json=body); dt = time.perf_counter() - t
    assert r.status_code == 200, r.text[:400]
    j = r.json(); e = j["embedding"]
    rec = {"request": tag, "seconds": round(dt, 3), "model_set": j["model_set"], "embedding": {k: e[k] for k in ("texts", "texts_sent", "texts_from_cache", "calls", "tokens", "cost_usd", "ms", "fallback_used", "fallback_reason")},
           "cost_usd": j["cost_usd"], "timing_ms": j["timing_ms"], "judge": {k: j["judge"][k] for k in ("used", "pairs", "calls", "cost_usd", "added_ms")},
           "pick": j["pick"], "p_v2_better": j["comparison"]["pairs"][0]["p_b_better"] if j["comparison"]["pairs"] else None, "status": j["comparison"]["status"]}
    return rec, j


for k, pr in enumerate(P.iter_rows(named=True)):
    h = posts.filter(pl.col("author_id") == pr["author_id"]).sort("created_at_ms")
    A, B = (h.filter(pl.col("post_id") == pr[c]).row(0, named=True) for c in ("post_id", "post_id_b"))
    tmin = min(A["created_at_ms"], B["created_at_ms"])
    hist = h.filter(pl.col("created_at_ms") < tmin).tail(300)
    body = {"handle": A["author_handle"], "followers": int(pr["followers"]), "versions": [version(A), version(B)], "past_posts": past(hist), "when": iso(tmin), "share_past_posts": True}
    recs = {}
    recs["first"], full = call(body, "first request for the account")
    recs["repeat"], _ = call(body, "the same request again")
    edited = {**version(B), "text": B["text"].rstrip() + "\n\nMore details in the thread below."}      # a version this process has not seen
    recs["edited"], _ = call({**body, "versions": [version(A), edited]}, "same account, one new version")
    recs["no_permission"], _ = call({**body, "share_past_posts": False}, "past posts may not be sent: fallback models")
    if judge is not None:
        recs["no_judge"], _ = call({**body, "judge": False}, "the same request, judge off (all vectors cached)")
    if k == 0:
        six = h.filter(pl.col("created_at_ms") >= tmin).head(6)
        if six.height >= 3:
            recs["six_versions"], _ = call({**body, "versions": [version(r) for r in six.iter_rows(named=True)]}, f"{six.height} versions")
        json.dump({"request": {**body, "past_posts": f"{len(body['past_posts'])} public corpus posts from before {iso(tmin)}"}, "response": full}, open(f"{a.out}/first_response.json", "w"), indent=1)
    row = {"account": k + 1, "handle": A["author_handle"], "followers": int(pr["followers"]), "history_posts": len(body["past_posts"]), "likes": [A["likes"], B["likes"]], "requests": recs}
    rows.append(row)
    print(json.dumps(row, indent=1), flush=True)
enc = svc.scorer.encoder
json.dump({"bundle": a.bundle, "model_set": svc.model_set, "encoder": info, "judge": a.judge, "load_seconds": round(load_s, 1), "threads": a.threads, "accounts": rows,
           "embedding_total": getattr(getattr(enc, "usage", None), "total", None), "judge_spend": judge.spend if judge else None}, open(f"{a.out}/summary.json", "w"), indent=1)
print("embedding total", getattr(getattr(enc, "usage", None), "total", None), "judge spend", judge.spend if judge else None)
