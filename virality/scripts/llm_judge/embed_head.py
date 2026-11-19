#!/usr/bin/env python3
"""Step 4, light supervised head: text-embedding-3-large of each post of a pair (text as the judge sees it), a logistic head
on the difference of the two embeddings (PCA to k dims, both orders, no intercept), trained on development pairs only.
Out-of-fold over accounts on the development pairs; one fit on all development pairs is applied to benchmark and launch pairs.
Writes data/llm/emb_head_{set}.parquet (post_id, post_id_b, e)."""
import json, os, sys, numpy as np, polars as pl, requests
sys.path.insert(0, "scripts/llm_judge")
from judge import fmt
from lib import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def embed(texts):
    out, tok = [], 0
    for i in range(0, len(texts), 256):
        r = requests.post("https://api.openai.com/v1/embeddings", headers={"Authorization": "Bearer proxy-injected"}, json={"model": "text-embedding-3-large", "input": [t[:6000] or " " for t in texts[i:i + 256]], "dimensions": 1024}, timeout=300).json()
        out += [x["embedding"] for x in r["data"]]; tok += r["usage"]["total_tokens"]
    return np.asarray(out, np.float32), tok

E, tot = {}, 0
for s in ("dev", "bench", "launch"):
    fn = f"data/llm/emb_{s}.npz"
    if not os.path.exists(fn):
        items = [json.loads(l) for l in open(f"data/llm/items_{s}.jsonl")]
        A, t1 = embed([fmt(x["a"], True) for x in items]); B, t2 = embed([fmt(x["b"], True) for x in items]); tot += t1 + t2
        np.savez(fn, a=A, b=B, post_id=[x["post_id"] for x in items], post_id_b=[x["post_id_b"] for x in items])
    E[s] = np.load(fn)
if tot:
    open("data/llm/spend.jsonl", "a").write(json.dumps({"path": "embeddings text-embedding-3-large", "in": tot, "cached": 0, "out": 0, "n": 0, "fail": 0, "usd": tot * 0.13e-6, "model": "text-embedding-3-large"}) + "\n"); print("embedding tokens", tot, "usd", tot * 0.13e-6)
P = pairs("dev")
assert (P["post_id"].to_numpy() == E["dev"]["post_id"]).all()
X = E["dev"]["a"] - E["dev"]["b"]; d = P["d"].to_numpy(); a = P["author_id"].to_numpy(); y = d > 0
rng = np.random.default_rng(0); ua = np.unique(a); fold = dict(zip(ua, rng.integers(0, 5, len(ua)))); fo = np.array([fold[x] for x in a])
res = {}
def fit(Xt, yt, k, C):
    p = PCA(k, random_state=0).fit(np.vstack([Xt, -Xt])); Z = p.transform(Xt)
    m = LogisticRegression(C=C, fit_intercept=False, max_iter=3000).fit(np.vstack([Z, -Z]), np.concatenate([yt, ~yt]).astype(int))
    return lambda Q: p.transform(Q) @ m.coef_[0]
best = None
for k in (16, 64, 256):
    for C in (0.01, 0.1, 1.0):
        s = np.zeros(len(d))
        for f in range(5):
            s[fo == f] = fit(X[fo != f], y[fo != f], k, C)(X[fo == f])
        r = acc(s, d, a); res[f"k{k}_C{C}"] = r
        tr = float((((fit(X, y, k, C)(X)) * d) > 0).mean())
        print(k, C, "out-of-fold", round(r["acc"], 4), [round(x, 3) for x in r["ci"]], "train", round(tr, 3))
        if best is None or r["acc"] > best[0]:
            best = (r["acc"], k, C, s)
_, k, C, s = best
print("chosen", k, C)
P.select("post_id", "post_id_b").with_columns(pl.Series("e", s)).write_parquet("data/llm/emb_head_dev.parquet")
f = fit(X, y, k, C)
for sname in ("bench", "launch"):
    pl.DataFrame({"post_id": E[sname]["post_id"], "post_id_b": E[sname]["post_id_b"], "e": f(E[sname]["a"] - E[sname]["b"])}).write_parquet(f"data/llm/emb_head_{sname}.parquet")
json.dump({"grid": res, "chosen": {"k": k, "C": C}, "note": "chosen setting's out-of-fold accuracy is the maximum of nine, so slightly optimistic"}, open("data/llm/emb_head.json", "w"), indent=1)
