#!/usr/bin/env python3
"""Does the poster frame help the model on video posts? Same-rows test, table path.

    python scripts/video/train_video.py --arm control|thumb|baseline --seed 1 --steps 600 --out data/video/runs/<name>

Both arms warm-start from the same baseline checkpoint (runs/ts46_s1) and fine-tune the `small` model with anchors
restricted to the SAME training rows: train-split posts that have poster-frame features (history rows are unrestricted).
  control  the 128 baseline inputs only
  thumb    plus 36 inputs: has-features flag, 16 principal components of the SigLIP-2 poster-frame embedding (fitted on
           train rows only), first-frame class one-hots and the attributes of label_thumbs.py
  baseline no training: the checkpoint as it is, scored on the same rows
Model selection on val-split rows with features. Predictions are written for every test-split row with features and for
every post on either side of the two benchmarks that is in the table: <out>/pred.parquet (post_id, likes_mean, likes_q50).
"""
import argparse, logging, os, sys
import numpy as np, polars as pl, torch
sys.path.insert(0, ".")
from xviral.model import evaluate, train
from xviral.model.config import ModelCfg, TrainCfg
from xviral.model.features import SPLITS, FeatureTable
ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["control", "thumb", "baseline"]); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--out", required=True); ap.add_argument("--init", default="data/runs/ts46_s1/best.pt"); ap.add_argument("--table", default="data/work/model/table")
args = ap.parse_args(); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s"); torch.set_num_threads(4)
ATTR = ["c_ui", "c_face", "c_title", "c_graphic", "c_real", "c_chart", "empty_first", "dark_first", "has_face", "big_face", "text_on_frame", "text_heavy", "big_headline", "portrait", "square", "hd1080", "sub720", "is_gif", "studio"]


class VTable(FeatureTable):
    def rows_of_split(self, name, audience_only=False):
        keep = (self.t["split"] == SPLITS[name]) & self.t["vt_has"]
        return torch.nonzero(keep).squeeze(1)


base = FeatureTable.load(args.table)
pid = base.t["post_id"].numpy(); split = base.t["split"].numpy()
F = pl.read_parquet("data/video/thumb_feats.parquet").with_row_index("i"); E = np.load("data/video/thumb_emb.npy").astype(np.float32)
L = pl.read_parquet("data/video/labels.parquet").with_columns([(pl.col("cls") == k).alias("c_" + k) for k in ["ui", "face", "title", "graphic", "real", "chart"]] + [(pl.col("mtype") == "gif").alias("is_gif"), (pl.col("route") == "amplify").alias("studio")])
L = L.join(F.select("post_id", "i"), on="post_id").unique("post_id")
row_of = {int(p): r for r, p in enumerate(pid)}
rows = np.array([row_of.get(int(p), -1) for p in L["post_id"].to_list()]); ok = rows >= 0
rows, Lk = rows[ok], L.filter(pl.Series(ok)); Ek = E[Lk["i"].to_numpy()]
has = np.zeros(len(pid), bool); has[rows] = True
tr = split[rows] == SPLITS["train"]
mu = Ek[tr].mean(0); _, _, Vt = np.linalg.svd(Ek[tr] - mu, full_matrices=False); P = (Ek - mu) @ Vt[:16].T; P = P / P[tr].std(0)
A = Lk.select(ATTR).to_numpy().astype(np.float32); A = (A - A[tr].mean(0)) / np.maximum(A[tr].std(0), 1e-3)
X = np.zeros((len(pid), 1 + 16 + len(ATTR)), np.float32); X[rows, 0] = 1.0; X[rows, 1:17] = P; X[rows, 17:] = A
names = ["vt_has"] + [f"vt_pc{i}" for i in range(16)] + ["vt_" + a for a in ATTR]
print("rows with features in table:", int(has.sum()), {k: int((has & (split == v)).sum()) for k, v in SPLITS.items()}, flush=True)
t = dict(base.t); t["vt_has"] = torch.from_numpy(has); meta = dict(base.meta)
if args.arm == "thumb":
    # already standardised; not part of num_base, so the stored serving transforms are untouched
    t["num"] = torch.cat([base.t["num"], torch.from_numpy(X)], 1); meta["num_names"] = list(meta["num_names"]) + names
table = VTable(meta=meta, t=t, texts=None)
ck = torch.load(args.init, map_location="cpu", weights_only=False)
os.makedirs(args.out, exist_ok=True)
init = args.init
if args.arm == "thumb":
    # new inputs start at zero weight: at step 0 the model equals the baseline
    w = ck["state"]["post.num.0.weight"]; ck["state"]["post.num.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], len(names))], 1)
    init = os.path.join(args.out, "init.pt"); torch.save(ck, init)
mcfg = ModelCfg(**{**ck["model_cfg"], "n_num": len(meta["num_names"])})
tcfg = TrainCfg(**{**ck["train_cfg"], "steps": args.steps, "lr": args.lr, "warmup_steps": 50, "eval_every": max(args.steps // 6, 1), "seed": args.seed, "train_audience_only": False})
if args.arm == "baseline":
    model, tcfg, _ = train.load(args.init)
else:
    train.train(table, args.out, mcfg, tcfg, init_from=init)
    model, tcfg, ck2 = train.load(os.path.join(args.out, "best.pt")); print("best", ck2["metrics"].get("step"), ck2["metrics"].get("val_nll"), flush=True)
B = pl.read_parquet("data/tracks/pairs/benchmark_pairs.parquet")
want = set(B["post_id"].to_list()) | set(B["post_id_b"].to_list()) | set(pl.read_parquet("data/tracks/launch/launch_scores.parquet")["post_id"].to_list())
sel = has & (split == SPLITS["test"]); sel |= np.isin(pid, np.array(sorted(want)))
r = torch.nonzero(torch.from_numpy(sel)).squeeze(1)
with torch.no_grad():
    pred = evaluate.predict_rows(model, table, r, tcfg, temperature=ck.get("temperature") or {}, with_nll=False)
out = pl.DataFrame({"post_id": pid[sel], "likes_mean": np.asarray(pred["likes_mean"], dtype=np.float32), "likes_q50": np.asarray(pred["likes_q50"], dtype=np.float32), "y_likes": np.asarray(pred["y_likes"], dtype=np.float32)})
out.write_parquet(os.path.join(args.out, "pred.parquet")); print("PRED_DONE", out.height, flush=True)
