#!/usr/bin/env python3
"""Development pairs (tracks/replicate, 2,805 pre-cutoff same-topic pairs) scored as alternative drafts by checkpoints:
per pair the difference of predicted median log10 likes (A minus B, as the benchmark scores) and each side's spread.

    python scripts/sota/dev_dist.py --out data/sota/dev name=ckpt.pt ...      -> <out>/<name>.npz (s, sd_a, sd_b, m_a, m_b)
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, "scripts")
from train_one import FEATURE_SETS
from xviral.model import episodes, pairs, train
from xviral.model.model import hist_quantile

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--dev-table", default="data/tracks/replicate/dev_table")
ap.add_argument("ckpt", nargs="+"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
ptable, frame = pairs.load(a.dev_table, FEATURE_SETS["ts46"])
ps = pairs.pair_set(ptable, frame)
for spec in a.ckpt:
    name, path = spec.split("=", 1)
    model, tcfg, ck = train.load(path, "cpu")
    T = (ck.get("temperature") or {}).get("likes", 1.0)
    spec_l = model.cfg.hist_heads["likes"]
    S, A, B, MA, MB = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(ps), 256):
            idx = torch.arange(i, min(i + 256, len(ps)))
            ep = episodes.build(ps.table, ps.anchor[idx], model.cfg, tcfg, train=False, n_candidates=2)
            lg = model(ps.table, ep)["hist"]["likes"].float() / T
            q = {p: hist_quantile(lg, spec_l, p) for p in (0.1, 0.5, 0.9)}
            first = ps.a_is_first[idx]
            ds = q[0.5][:, 0] - q[0.5][:, 1]
            sd = (q[0.9] - q[0.1]) / 2.563
            S.append(torch.where(first, ds, -ds)); A.append(torch.where(first, sd[:, 0], sd[:, 1])); B.append(torch.where(first, sd[:, 1], sd[:, 0]))
            MA.append(torch.where(first, q[0.5][:, 0], q[0.5][:, 1])); MB.append(torch.where(first, q[0.5][:, 1], q[0.5][:, 0]))
    np.savez(f"{a.out}/{name}.npz", s=torch.cat(S).numpy(), sd_a=torch.cat(A).numpy(), sd_b=torch.cat(B).numpy(), m_a=torch.cat(MA).numpy(), m_b=torch.cat(MB).numpy())
    print(name, "done", flush=True)
