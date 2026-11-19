#!/usr/bin/env python3
"""Step 3: a small projection network on frozen text embeddings, trained so that distance reflects
"does better for this account" rather than topic.

    python research/tracks/embed_api/adapter.py --table data/work/model/table --recipe data/api/recipes/bge_te3l1024.json \
        --raw te3l=data/api/te3l_table_full.npy --out data/api/adapters/bge_te3l1024 [--d-out 128 --hidden 512 --dropout 0.3 ...]

Input: the recipe's parts (unit blocks, concatenated). Network: Linear -> GELU -> dropout -> Linear -> z (d-out, used
unit length) and a linear scalar head on z. Each batch holds A accounts x M posts of the training split. Two losses,
both strictly within account:

  rank   logistic loss on head(z_i) - head(z_j) for two posts of one account whose likes differ by 25% or more
         ("which of this account's two posts did better")
  knn    the trunk's own use of the embedding (model.HistoryKNN): a post's outcome relative to the account's level
         is predicted as the softmax(cosine / temperature)-weighted mean of the account's other posts in the batch;
         squared error. This is the contrastive part: it pulls together posts of one account that performed alike
         and pushes apart those that did not, and nothing is compared across accounts.

Outcome: log10(likes + 1) minus the account's mean over its posts in the batch (the account's level is removed, so
audience size carries no signal).

Validation, to guard against both kinds of overfitting (a rule that holds on held-out accounts may fail later in time):
  fit        training-split accounts, posts before --holdout-days before the cutoff
  val_acct   validation-split accounts (never seen), same period
  val_time   training-split accounts, the last --holdout-days before the cutoff
  val_both   validation-split accounts, the last --holdout-days before the cutoff     <- model selection
  test       test-split rows (after the cutoff): reported once, never used for a choice
Metrics on each: pair accuracy of the scalar head on same-account pairs at most --pair-days apart with a 25%+ like
gap, and the same for the leave-one-out knn prediction; each with the raw (unadapted) embedding's numbers beside it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import recipe as R  # noqa: E402

CUTOFF_S = 1776902400            # 2026-04-23 00:00 UTC, the benchmark's --after
LOG125 = float(np.log10(1.25))


class Adapter(nn.Module):
    def __init__(self, d_in: int, hidden: int, d_out: int, dropout: float, in_dropout: float = 0.0):
        super().__init__()
        self.cfg = dict(d_in=d_in, hidden=hidden, d_out=d_out, dropout=dropout, in_dropout=in_dropout)
        self.in_drop = nn.Dropout(in_dropout)
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_out)) if hidden else nn.Linear(d_in, d_out)
        self.head = nn.Linear(d_out, 1, bias=False)
        self.scale = (d_in ** 0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = F.normalize(self.net(self.in_drop(x * self.scale)), dim=-1)
        return z, self.head(z).squeeze(-1)

    @torch.no_grad()
    def embed(self, X: np.ndarray, batch: int = 65536) -> np.ndarray:
        self.eval()
        return np.concatenate([self(torch.from_numpy(np.asarray(X[i:i + batch], dtype=np.float32)))[0].numpy() for i in range(0, len(X), batch)])

    @torch.no_grad()
    def score(self, X: np.ndarray, batch: int = 65536) -> np.ndarray:
        self.eval()
        return np.concatenate([self(torch.from_numpy(np.asarray(X[i:i + batch], dtype=np.float32)))[1].numpy() for i in range(0, len(X), batch)])


def load_adapter(path: str) -> Adapter:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Adapter(**ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.eval()


# ---------------------------------------------------------------------------------------------------------------------
def make_pairs(author: np.ndarray, t: np.ndarray, y: np.ndarray, rows: np.ndarray, pair_days: float, per_account: int, seed: int):
    """Same-account pairs among `rows`, at most pair_days apart, 25%+ like gap; at most per_account per account."""
    rng = np.random.default_rng(seed)
    order = rows[np.lexsort((t[rows], author[rows]))]
    a = author[order]
    starts = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
    ends = np.r_[starts[1:], len(order)]
    I, J = [], []
    for s, e in zip(starts, ends):
        n = e - s
        if n < 2:
            continue
        k = min(per_account * 3, n * (n - 1) // 2)
        i = rng.integers(0, n, k)
        j = rng.integers(0, n, k)
        ok = (i != j)
        gi, gj = order[s + i[ok]], order[s + j[ok]]
        ok2 = (np.abs(t[gi] - t[gj]) <= pair_days * 86400) & (np.abs(y[gi] - y[gj]) >= LOG125)
        gi, gj = gi[ok2][:per_account], gj[ok2][:per_account]
        I.append(gi)
        J.append(gj)
    return np.concatenate(I), np.concatenate(J)


def cluster_ci(auth, correct, n=300, seed=1):
    rng = np.random.default_rng(seed)
    ua, inv = np.unique(auth, return_inverse=True)
    num = np.bincount(inv, weights=correct.astype(float), minlength=len(ua))
    den = np.bincount(inv, minlength=len(ua)).astype(float)
    v = []
    for _ in range(n):
        s = rng.integers(0, len(ua), len(ua))
        v.append(num[s].sum() / den[s].sum())
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def knn_scores(Z: np.ndarray, author: np.ndarray, t: np.ndarray, y: np.ndarray, rows: np.ndarray, temp: float, hist: int = 48) -> np.ndarray:
    """For each row: the similarity-weighted mean outcome (relative to the history mean) of the account's previous `hist` posts
    among `rows`: the content kernel of the trunk's HistoryKNN. NaN when there is no earlier post."""
    out = np.full(len(author), np.nan, dtype=np.float32)
    order = rows[np.lexsort((t[rows], author[rows]))]
    a = author[order]
    starts = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        z = Z[idx].astype(np.float32)
        z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
        sim = z @ z.T
        yy = y[idx]
        for k in range(1, len(idx)):
            lo = max(0, k - hist)
            w = sim[k, lo:k] / temp
            w = np.exp(w - w.max())
            h = yy[lo:k]
            out[idx[k]] = float((w * (h - h.mean())).sum() / w.sum())
    return out


def pair_acc(s: np.ndarray, y: np.ndarray, I: np.ndarray, J: np.ndarray, author: np.ndarray) -> dict:
    d = s[I] - s[J]
    ok = np.isfinite(d) & (d != 0)
    c = (d[ok] * (y[I][ok] - y[J][ok])) > 0
    if not len(c):
        return {"pairs": 0, "accuracy": None}
    return {"pairs": int(len(c)), "accuracy": float(c.mean()), "ci": cluster_ci(author[I][ok], c)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--recipe", required=True, help="the adapter's input (parts only)")
    ap.add_argument("--raw", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--d-out", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--in-dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--accounts", type=int, default=48)
    ap.add_argument("--posts", type=int, default=24)
    ap.add_argument("--knn-weight", type=float, default=1.0)
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--holdout-days", type=float, default=60)
    ap.add_argument("--pair-days", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="also score the test split (after the cutoff); run once, for the chosen configuration")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    src, out = Path(args.table), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = R.load(args.recipe)
    assert not rec.get("adapter")
    raw = {"bge": np.load(src / "emb.npy", mmap_mode="r")} | {s.split("=")[0]: np.load(s.split("=")[1], mmap_mode="r") for s in args.raw}
    X = R.apply(rec, {k: raw[k] for k in R.sources(rec)})
    X *= np.float32(np.sqrt(len(rec["parts"])))                                               # unit blocks side by side
    split, author, t = np.load(src / "split.npy"), np.load(src / "author_idx.npy"), np.load(src / "created_s.npy")
    y = np.load(src / "y_likes.npy").astype(np.float32)                                       # log10(likes + 1)
    T1 = CUTOFF_S - args.holdout_days * 86400
    sets = {"fit": np.flatnonzero((split == 0) & (t < T1)), "val_acct": np.flatnonzero((split == 1) & (t < T1)),
            "val_time": np.flatnonzero((split == 0) & (t >= T1) & (t < CUTOFF_S)), "val_both": np.flatnonzero((split == 1) & (t >= T1) & (t < CUTOFF_S))}
    if args.final:
        sets["test"] = np.flatnonzero(split == 2)
    pairs = {k: make_pairs(author, t, y, r, args.pair_days, 40 if k == "fit" else 200, 0) for k, r in sets.items()}
    print({k: (len(r), len(pairs[k][0]), len(np.unique(author[r]))) for k, r in sets.items()}, "(rows, pairs, accounts)", flush=True)

    # fit rows grouped by account
    fr = sets["fit"]
    order = fr[np.argsort(author[fr], kind="stable")]
    a = author[order]
    starts = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
    ends = np.r_[starts[1:], len(order)]
    keep = (ends - starts) >= 4
    starts, ends = starts[keep], ends[keep]
    p_acc = np.sqrt(ends - starts).astype(float)
    p_acc /= p_acc.sum()                                   # large accounts more often, but far less than in proportion to their posts

    model = Adapter(X.shape[1], args.hidden, args.d_out, args.dropout, args.in_dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)

    def evaluate() -> dict:
        s = model.score(X)
        Z = model.embed(X)
        res = {}
        for k, r in sets.items():
            I, J = pairs[k]
            sub = r if k != "fit" else r[rng.random(len(r)) < 0.15]
            kn = knn_scores(Z, author, t, y, sub, args.temp)
            res[k] = {"head": pair_acc(s, y, I, J, author), "knn": pair_acc(kn, y, I, J, author)}
        return res

    hist, best, t0 = [], None, time.time()
    for step in range(1, args.steps + 1):
        model.train()
        acc = rng.choice(len(starts), args.accounts, replace=False, p=p_acc)
        idx = np.stack([order[rng.integers(starts[k], ends[k], args.posts)] for k in acc])            # [A, M], with replacement inside an account
        xb, yb = Xt[idx.reshape(-1)], yt[idx]
        z, s = model(xb)
        z, s = z.view(args.accounts, args.posts, -1), s.view(args.accounts, args.posts)
        yc = yb - yb.mean(1, keepdim=True)
        same = torch.from_numpy(idx[:, :, None] == idx[:, None, :])
        dy = yb[:, :, None] - yb[:, None, :]
        m = (dy.abs() >= LOG125) & ~same
        l_rank = F.binary_cross_entropy_with_logits((s[:, :, None] - s[:, None, :])[m], (dy[m] > 0).float())
        sim = (z @ z.transpose(1, 2) / args.temp).masked_fill(same, float("-inf"))
        pred = (sim.softmax(-1) * yc[:, None, :]).sum(-1)
        l_knn = F.mse_loss(pred, yc)
        loss = args.rank_weight * l_rank + args.knn_weight * l_knn
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate()
            sel = np.mean([ev["val_both"]["head"]["accuracy"], ev["val_both"]["knn"]["accuracy"]])
            hist.append({"step": step, "loss_rank": float(l_rank), "loss_knn": float(l_knn), "select": float(sel), **ev})
            print(f"step {step} rank {float(l_rank):.4f} knn {float(l_knn):.4f} | " + " | ".join(
                f"{k} head {100 * v['head']['accuracy']:.1f} knn {100 * v['knn']['accuracy']:.1f}" for k, v in ev.items()) + f" | {time.time() - t0:.0f}s", flush=True)
            if best is None or sel > best["select"]:
                best = hist[-1]
                torch.save({"cfg": model.cfg, "state": model.state_dict(), "args": vars(args), "recipe": rec, "step": step}, out / "adapter.pt")

    # the unadapted embedding under the same metrics: knn on the raw vectors, and a linear head on them trained on the fit pairs
    model = load_adapter(str(out / "adapter.pt"))
    base = {}
    from sklearn.linear_model import LogisticRegression
    I, J = pairs["fit"]
    keep_ = np.random.default_rng(0).permutation(len(I))[:60_000]
    I, J = I[keep_], J[keep_]
    D = (X[I] - X[J]).astype(np.float32)
    D = np.vstack([D, -D])
    lab = np.r_[y[I] > y[J], y[I] < y[J]].astype(int)
    w = LogisticRegression(C=0.01, fit_intercept=False, max_iter=300).fit(D * np.sqrt(X.shape[1]), lab).coef_[0]
    s_lin = X @ w
    for k, r in sets.items():
        I, J = pairs[k]
        sub = r if k != "fit" else r[rng.random(len(r)) < 0.15]
        base[k] = {"linear_head_on_raw": pair_acc(s_lin, y, I, J, author), "knn_on_raw": pair_acc(knn_scores(X, author, t, y, sub, args.temp), y, I, J, author)}
    res = {"args": vars(args), "recipe": rec, "rows_pairs_accounts": {k: [len(r), len(pairs[k][0]), int(len(np.unique(author[r])))] for k, r in sets.items()},
           "best": best, "raw": base, "history": hist}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print("BEST step", best["step"])
    for k in sets:
        print(f"{k:<9} adapted: head {100 * best[k]['head']['accuracy']:.1f} knn {100 * best[k]['knn']['accuracy']:.1f} | raw: linear head "
              f"{100 * base[k]['linear_head_on_raw']['accuracy']:.1f} knn {100 * base[k]['knn_on_raw']['accuracy']:.1f}  ({best[k]['head']['pairs']} pairs)")


if __name__ == "__main__":
    main()
