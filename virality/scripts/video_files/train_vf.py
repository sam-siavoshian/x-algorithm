#!/usr/bin/env python3
"""Do whole-video and audio inputs help the model on video posts? Same-rows test, table path (harness of
scripts/video/train_video.py).

    python scripts/video_files/train_vf.py --arm control|vf|baseline --seed 1 --steps 600 --out data/vf/runs/<name>

Both arms warm-start from the same baseline checkpoint (runs/ts46_s1) and fine-tune the `small` model with anchors
restricted to the SAME training rows: train-split posts with video-file features (history rows are unrestricted).
  control   the baseline inputs only
  vf        plus a has-features flag, 16 principal components of the pooled SigLIP-2 embedding of 5 frames spread over
            the video (fitted on train rows only), and the attributes in ATTR (standardised on train rows; missing = 0)
  baseline  no training: the checkpoint as it is
Model selection on val-split rows with features. Predictions are written for EVERY row with features (train, val and
test split) so that the three can be reported side by side, and for up to 60 other test-split posts of each account
with a test-split video (video-against-other pairs): <out>/pred.parquet (post_id, split, likes_mean, y_likes).
"""
import argparse, logging, os, sys
import numpy as np, polars as pl, torch
sys.path.insert(0, ".")
from xviral.model import evaluate, train
from xviral.model.config import ModelCfg, TrainCfg
from xviral.model.features import SPLITS, FeatureTable
ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["control", "vf", "baseline"]); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--out", required=True); ap.add_argument("--init", default="data/runs/ts46_s1/best.pt"); ap.add_argument("--table", default="data/work/model/table")
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args(); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s"); torch.set_num_threads(args.threads)
c = pl.col


class VTable(FeatureTable):
    def rows_of_split(self, name, audience_only=False):
        keep = (self.t["split"] == SPLITS[name]) & self.t["vf_has"]
        return torch.nonzero(keep).squeeze(1)


def attributes(V):
    """Numeric attribute matrix (NaN where unknown) and names. Chosen before the model ran: every attribute that
    survived or leaned in the effects, plus the plain descriptors of length, kind and sound."""
    f = lambda e: e.cast(pl.Float64)
    cols = {"log_dur": c("duration").clip(0.5, 3600).log(), "dur_u15": f(c("dur_band") == "u15"), "dur_15_30": f(c("dur_band") == "15_30"), "dur_30_60": f(c("dur_band") == "30_60"), "dur_60_120": f(c("dur_band") == "60_120"),
            "kind_screen": f(c("kind") == "screen"), "kind_person": f(c("kind") == "person"), "kind_film": f(c("kind") == "film"), "kind_mixed": f(c("kind") == "mixed"),
            "ui_share": c("ui_share"), "face_share": c("face_share"), "log_cuts": (c("cuts_per_min") + 1).log(), "any_cut": f(c("cuts") > 0), "cut_first3s": f(c("cut_first3s")), "captions": f(c("captions")),
            "product_first3s": f(c("product_first3s")), "person_first3s": f(c("person_first3s")), "title_first3s": f(c("title_first3s")), "is_gif": f(c("is_gif")), "portrait": f(c("portrait")), "sub720": f(c("sub720")),
            "has_audio": f(c("has_audio")), "audible": f(c("audible")), "speech": f(c("sound_class").is_in(["speech_only", "speech_and_music"])), "music": f(c("sound_class").is_in(["music_only", "speech_and_music"])),
            "lufs": c("lufs").clip(-60, 0), "speech_first_2s": f(c("speech_first_2s")), "speech_share": c("speech_share"), "music_share": c("music_share")}
    X = V.select([e.alias(k) for k, e in cols.items()]).to_numpy().astype(np.float32)
    return X, list(cols)


base = FeatureTable.load(args.table)
pid = base.t["post_id"].numpy(); split = base.t["split"].numpy()
V = pl.read_parquet("data/vf/video_features.parquet").with_row_index("i"); E = np.load("data/vf/video_emb.npy").astype(np.float32)
row_of = {int(p): r for r, p in enumerate(pid)}
rows = np.array([row_of.get(int(p), -1) for p in V["post_id"].to_list()]); ok = rows >= 0
rows, Vk = rows[ok], V.filter(pl.Series(ok)); Ek = E[Vk["i"].to_numpy()]
has = np.zeros(len(pid), bool); has[rows] = True
tr = split[rows] == SPLITS["train"]
mu = Ek[tr].mean(0); _, _, Vt = np.linalg.svd(Ek[tr] - mu, full_matrices=False); P = (Ek - mu) @ Vt[:16].T; P = P / P[tr].std(0)
A, anames = attributes(Vk) if args.arm == "vf" else (np.zeros((Vk.height, 1), np.float32), ["none"]); m = np.nanmean(A[tr], 0); s = np.maximum(np.nanstd(A[tr], 0), 1e-3); A = np.nan_to_num((A - m) / s, nan=0.0)
X = np.zeros((len(pid), 1 + 16 + A.shape[1]), np.float32); X[rows, 0] = 1.0; X[rows, 1:17] = P; X[rows, 17:] = A
names = ["vf_has"] + [f"vf_pc{i}" for i in range(16)] + ["vf_" + a for a in anames]
print("rows with features in table:", int(has.sum()), {k: int((has & (split == v)).sum()) for k, v in SPLITS.items()}, "new inputs:", len(names), flush=True)
t = dict(base.t); t["vf_has"] = torch.from_numpy(has); meta = dict(base.meta)
if args.arm == "vf":
    t["num"] = torch.cat([base.t["num"], torch.from_numpy(X)], 1); meta["num_names"] = list(meta["num_names"]) + names
table = VTable(meta=meta, t=t, texts=None)
ck = torch.load(args.init, map_location="cpu", weights_only=False)
os.makedirs(args.out, exist_ok=True)
init = args.init
if args.arm == "vf":
    w = ck["state"]["post.num.0.weight"]; ck["state"]["post.num.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], len(names))], 1)   # new inputs start at zero: step 0 equals the baseline
    init = os.path.join(args.out, "init.pt"); torch.save(ck, init)
mcfg = ModelCfg(**{**ck["model_cfg"], "n_num": len(meta["num_names"])})
tcfg = TrainCfg(**{**ck["train_cfg"], "steps": args.steps, "lr": args.lr, "warmup_steps": 50, "eval_every": max(args.steps // 6, 1), "seed": args.seed, "train_audience_only": False})
if args.arm == "baseline":
    model, tcfg, _ = train.load(args.init)
else:
    train.train(table, args.out, mcfg, tcfg, init_from=init)
    model, tcfg, ck2 = train.load(os.path.join(args.out, "best.pt")); print("best", ck2["metrics"].get("step"), ck2["metrics"].get("val_nll"), flush=True)
# also score up to 60 other test-split posts of each account that has a test-split video, for video-against-other pairs
au = base.t["author_idx"].numpy(); rs = np.random.default_rng(0)
tv = np.unique(au[has & (split == SPLITS["test"])]); cand = np.nonzero((split == SPLITS["test"]) & ~has & np.isin(au, tv))[0]; rs.shuffle(cand)
_, first = np.unique(au[cand], return_index=True); order = np.argsort(au[cand], kind="stable"); rank = np.empty(len(cand), int)
sa = au[cand][order]; start = np.r_[0, np.nonzero(np.diff(sa))[0] + 1]; rank[order] = np.arange(len(cand)) - np.repeat(start, np.diff(np.r_[start, len(cand)]))
sel = has.copy(); sel[cand[rank < 60]] = True
r = torch.nonzero(torch.from_numpy(sel)).squeeze(1)
with torch.no_grad():
    pred = evaluate.predict_rows(model, table, r, tcfg, temperature=ck.get("temperature") or {}, with_nll=False)
inv = {v: k for k, v in SPLITS.items()}
out = pl.DataFrame({"post_id": pid[sel], "author_idx": au[sel], "has_vf": has[sel], "split": [inv[int(x)] for x in split[sel]], "likes_mean": np.asarray(pred["likes_mean"], dtype=np.float32), "likes_q50": np.asarray(pred["likes_q50"], dtype=np.float32), "y_likes": np.asarray(pred["y_likes"], dtype=np.float32)})
out.write_parquet(os.path.join(args.out, "pred.parquet")); print("PRED_DONE", out.height, flush=True)
