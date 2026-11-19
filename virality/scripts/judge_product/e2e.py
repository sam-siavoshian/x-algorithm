#!/usr/bin/env python3
"""A real end-to-end run of the product with the judge: three accounts from the corpus, through the HTTP layer.

    python scripts/judge_product/e2e.py [--model gpt-6.1-sol] [--seed 11]

For each of three accounts (drawn with a fixed seed from the launch test pairs: real gap 2x or more, 30 or more earlier
posts, one pair per account) the request carries two real launch posts of the account as versions 1 and 2, the account's
posts from before the earlier of the two as `past_posts` (with share_past_posts: these are public posts from the corpus),
and the earlier post's time as `when`. The same request is sent with `judge: false` for the report without a second
opinion and the added latency. A third request sends six of the account's launch posts as six versions (15 pairs, 30
calls) to time the largest request. Everything sent is a public post already in the corpus. Three accounts are an
illustration that the pipeline runs, not a measurement; the outcomes are printed next to the verdicts all the same."""
import argparse, json, os, sys, time
import numpy as np, polars as pl
from fastapi.testclient import TestClient

from xviral import judge as J
from xviral.app import server, service

D = "data"
CK = [f"{D}/runs/ts46_s1/best.pt", f"{D}/runs/ts46_s2/best.pt", f"{D}/work/model/compare_v2/full/best.pt"]


def iso(ms):
    return service._iso(int(ms))


def past(rows: pl.DataFrame) -> list[dict]:
    out = []
    for r in rows.iter_rows(named=True):
        kinds = {m.get("type") for m in (r["media"] or [])}
        out.append({"text": r["text"], "posted_at": iso(r["created_at_ms"]), "likes": r["likes"], "views": r["views"], "replies": r["replies"], "reposts": r["reposts"],
                    "image": "photo" in kinds, "video": bool(kinds & {"video", "animated_gif"})})
    return out


def version(r: dict) -> dict:
    kinds = {m.get("type") for m in (r["media"] or [])}
    return {"text": r["text"], "image": "photo" in kinds and not kinds & {"video", "animated_gif"}, "video": bool(kinds & {"video", "animated_gif"})}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gpt-6.1-sol"); ap.add_argument("--seed", type=int, default=11); ap.add_argument("--out", default="data/llm/e2e"); a = ap.parse_args()
    P = pl.read_parquet(f"{D}/llm/jp_pairs_launch.parquet").filter((pl.col("d").abs() >= np.log(2)) & (pl.col("n_hist") >= 30) & pl.col("followers").is_not_null())
    P = P.sample(fraction=1.0, shuffle=True, seed=a.seed).unique("author_id", keep="first", maintain_order=True).head(3)
    posts = pl.scan_parquet(f"{D}/work/filtered/posts.parquet").filter(pl.col("author_id").is_in(P["author_id"].implode())).collect().filter(~pl.col("is_reply").fill_null(False) & pl.col("text").is_not_null())
    L = pl.read_parquet(f"{D}/tracks/launch/launch_posts.parquet").filter(pl.col("created_at_ms") >= 1776902400000)
    judge = J.Judge(a.model, timeout=45, retries=1)
    t0 = time.time()
    svc = service.load(CK, "research/15_pair_confidence.json", posts=None, sid_codebook=f"{D}/work/emb/sid_codebook.npz", type_correction="research/tracks/posttype/type_correction.json", judge=judge)
    print(f"models loaded in {time.time() - t0:.1f}s; judge {judge.describe()}", flush=True)
    client = TestClient(server.create_app(svc, settings=server.Settings(rate_per_minute=0)))
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for k, pr in enumerate(P.iter_rows(named=True)):
        h = posts.filter(pl.col("author_id") == pr["author_id"]).sort("created_at_ms")
        A, B = (h.filter(pl.col("post_id") == pr[c]).row(0, named=True) for c in ("post_id", "post_id_b"))
        tmin = min(A["created_at_ms"], B["created_at_ms"])
        hist = h.filter(pl.col("created_at_ms") < tmin).tail(300)
        body = {"handle": A["author_handle"], "followers": int(pr["followers"]), "versions": [version(A), version(B)], "past_posts": past(hist), "when": iso(tmin), "share_past_posts": True}
        res = {}
        for tag, extra in (("warm-up, no judge", {"judge": False}), ("forecast model alone", {"judge": False}), ("with judge", {})):
            t = time.perf_counter(); r = client.post("/api/analyze", json={**body, **extra}); dt = time.perf_counter() - t
            assert r.status_code == 200, r.text[:500]
            res[tag] = (r.json(), dt)
        (rj, tj), (r0, t0_) = res["with judge"], res["forecast model alone"]
        json.dump({"request": {**body, "past_posts": f"{len(body['past_posts'])} posts from before {iso(tmin)} (public, from the corpus)"}, "response_with_judge": rj, "response_judge_false": r0},
                  open(f"{a.out}/account_{k + 1}.json", "w"), indent=1)
        actual = 2 if B["likes"] > A["likes"] else 1
        pj, p0 = rj["comparison"]["pairs"][0], r0["comparison"]["pairs"][0]
        row = {"account": k + 1, "handle": A["author_handle"], "followers": int(pr["followers"]), "history_posts": len(body["past_posts"]), "likes": [A["likes"], B["likes"]], "actual_better": actual,
               "texts": [A["text"][:160], B["text"][:160]],
               "forecast_alone": {"p_v2_better": p0["p_b_better"], "status": p0["status"], "pick": r0["pick"]["version"], "statement": r0["pick"]["statement"], "seconds": round(t0_, 2)},
               "with_judge": {"p_v2_better": pj["p_b_better"], "status": pj["status"], "pick": rj["pick"]["version"], "statement": rj["pick"]["statement"], "opinions": pj.get("opinions"), "kind": pj.get("kind"),
                              "seconds": round(tj, 2)},
               "judge": rj["judge"], "judge_validation": rj["method"]["judge_validation"], "timing_ms": rj["timing_ms"], "added_seconds": round(tj - t0_, 2)}
        rows.append(row)
        print(json.dumps(row, indent=1), flush=True)
    # the largest request: six versions, 15 pairs, 30 calls
    pr = P.row(0, named=True)
    h = posts.filter(pl.col("author_id") == pr["author_id"]).sort("created_at_ms")
    six = h.filter(pl.col("post_id").is_in(L.filter(pl.col("author_id") == pr["author_id"])["post_id"].implode())).head(6)
    if six.height >= 3:
        tmin = int(six["created_at_ms"].min())
        body = {"handle": six["author_handle"][0], "followers": int(pr["followers"]), "versions": [version(r) for r in six.iter_rows(named=True)], "past_posts": past(h.filter(pl.col("created_at_ms") < tmin).tail(300)),
                "when": iso(tmin), "share_past_posts": True}
        t = time.perf_counter(); r0 = client.post("/api/analyze", json={**body, "judge": False}); t_0 = time.perf_counter() - t
        t = time.perf_counter(); r = client.post("/api/analyze", json=body); t_1 = time.perf_counter() - t
        rj = r.json()
        big = {"versions": six.height, "pairs": rj["judge"]["pairs"], "calls": rj["judge"]["calls"], "pairs_answered": rj["judge"]["pairs_answered"], "cost_usd": rj["judge"]["cost_usd"], "judge_wait_ms": rj["timing_ms"]["judge_wait"],
               "seconds_with_judge": round(t_1, 2), "seconds_without": round(t_0, 2), "comparison_status": rj["comparison"]["status"], "statement": rj["comparison"]["statement"], "opinions": rj["comparison"].get("opinions"),
               "likes": six["likes"].to_list(), "winner": rj["comparison"]["winner"]}
        json.dump({"response_with_judge": rj}, open(f"{a.out}/six_versions.json", "w"), indent=1)
        print(json.dumps(big, indent=1))
    else:
        big = None
    json.dump({"model": a.model, "provider": judge.provider, "accounts": rows, "six_versions": big, "judge_spend": judge.spend}, open(f"{a.out}/summary.json", "w"), indent=1)
    print("judge spend", judge.spend)
    open("data/llm/spend.jsonl", "a").write(json.dumps({"path": f"e2e {a.model}", "n": judge.spend["requests"], "in": judge.spend["tokens_in"], "cached": judge.spend["tokens_cached"], "out": judge.spend["tokens_out"],
                                                         "usd": judge.spend["usd"], "model": a.model, "endpoint": judge.endpoint, "time": time.time()}) + "\n")


if __name__ == "__main__":
    main()
