#!/usr/bin/env python3
"""Is there any within-account signal in the poster frame alone? Ridge on the lift (likes over the account's trailing
median), features and outcome demeaned per account, trained on video posts before 2026-04-23, tested on same-account
pairs of video posts from 2026-04-23 on (lifts differing 1.25x or more, at most 50 pairs an account).
Feature sets: 32 principal components of the SigLIP-2 embedding; the labelled attributes; both. Writes data/video/thumb_probe.json."""
import json, numpy as np, polars as pl
rng = np.random.default_rng(0); CUT = 1776902400000
ATTR = ["c_ui", "c_face", "c_title", "c_graphic", "c_real", "empty_first", "dark_first", "has_face", "big_face", "text_on_frame", "text_heavy", "big_headline", "portrait", "square", "hd1080", "sub720", "is_gif", "studio"]
F = pl.read_parquet("data/video/thumb_feats.parquet").with_row_index("i").select("post_id", "i"); E = np.load("data/video/thumb_emb.npy").astype(np.float32)
L = pl.read_parquet("data/video/labels.parquet").with_columns([(pl.col("cls") == k).alias("c_" + k) for k in ["ui", "face", "title", "graphic", "real"]] + [(pl.col("mtype") == "gif").alias("is_gif"), (pl.col("route") == "amplify").alias("studio")])
T = pl.read_parquet("data/launch/table.parquet").filter((pl.col("source") == "community_archive") & pl.col("lift_likes").is_not_null()).select("post_id", "author_id", "created_at_ms", "lift_likes")
D = T.join(L.select(["post_id"] + ATTR), on="post_id").join(F, on="post_id").sort("author_id", "created_at_ms")
Em = E[D["i"].to_numpy()]; tr = (D["created_at_ms"] < CUT).to_numpy(); a = D["author_id"].to_numpy(); y = D["lift_likes"].to_numpy()
mu = Em[tr].mean(0); _, _, Vt = np.linalg.svd(Em[tr] - mu, full_matrices=False); P = (Em - mu) @ Vt[:32].T; P /= P[tr].std(0)
A = D.select(ATTR).to_numpy().astype(np.float32); A = (A - A[tr].mean(0)) / np.maximum(A[tr].std(0), 1e-3)
def demean(M, g):
    u, inv = np.unique(g, return_inverse=True); s = np.zeros((len(u),) + M.shape[1:]); np.add.at(s, inv, M); c = np.bincount(inv).astype(float)
    return M - (s / c.reshape(-1, *([1] * (M.ndim - 1))))[inv]
te = D.filter(~pl.Series(tr)).with_row_index("j")
pr = te.join(te, on="author_id", suffix="_b").filter((pl.col("post_id") < pl.col("post_id_b")) & ((pl.col("lift_likes") - pl.col("lift_likes_b")).abs() >= np.log(1.25)))
pr = pr.with_columns(pl.int_range(pl.len()).shuffle(seed=1).over("author_id").alias("r")).filter(pl.col("r") < 50)
ja, jb = pr["j"].to_numpy(), pr["j_b"].to_numpy(); d = (pr["lift_likes"] - pr["lift_likes_b"]).to_numpy(); au = pr["author_id"].to_numpy()
out = {"train_posts": int(tr.sum()), "train_accounts": int(len(np.unique(a[tr]))), "test_pairs": int(pr.height), "test_accounts": int(len(np.unique(au))), "sets": {}}
for name, X in (("embedding_32pc", P), ("attributes", A), ("both", np.concatenate([P, A], 1))):
    Xd, yd = demean(X[tr], a[tr]), demean(y[tr], a[tr]); best = None
    for lam in (10.0, 100.0, 1000.0):     # chosen on the last 20% of training accounts' posts by time is overkill here; report all three
        w = np.linalg.solve(Xd.T @ Xd + lam * np.eye(X.shape[1]), Xd.T @ yd); s = X[~tr] @ w; c = (((s[ja] - s[jb]) * d) > 0).astype(float)
        u, inv = np.unique(au, return_inverse=True); S = np.zeros(len(u)); C = np.zeros(len(u)); np.add.at(S, inv, c); np.add.at(C, inv, 1)
        idx = rng.integers(0, len(u), (2000, len(u))); b = S[idx].sum(1) / C[idx].sum(1)
        out["sets"].setdefault(name, {})[f"ridge_{int(lam)}"] = {"accuracy": round(float(c.mean()), 4), "lo": round(float(np.quantile(b, .025)), 4), "hi": round(float(np.quantile(b, .975)), 4)}
        print(name, lam, out["sets"][name][f"ridge_{int(lam)}"])
print({k: v for k, v in out.items() if k != "sets"}); json.dump(out, open("data/video/thumb_probe.json", "w"), indent=1)
