#!/usr/bin/env python3
"""Train one model on the general table with train, validation and test curves at every evaluation point.

    python scripts/general/train_general.py --table data/general/table --out data/general/runs/<name> --seed 1 \
        --roles orig,expand,train --steps 6000 [--drop-cat src_name,src_regime] [--init ckpt --lr 2e-4 --select val_aud]

--roles   which rows are trained on (row_info g_role): orig (the 465,675 existing rows), expand (the expand track's added rows),
          train (the general track's training accounts). Rows of other roles stay in the table as history and are never anchors.
Curves, at --evals points, on fixed account samples: train (rows trained on), val_aud (audience validation accounts of the existing
table), val_broad (the general track's validation accounts), and for reporting only test_aud (existing audience test rows) and bed
(test-bed accounts). The checkpoint is chosen on validation alone: the mean NLL of --select (default val_aud,val_broad).
The loop is xviral.model.train.train with more evaluation sets; optimiser, schedule, EMA and loss are its own.
"""
from __future__ import annotations
import argparse, copy, json, logging, time
from pathlib import Path
import numpy as np, polars as pl, torch
from xviral.model import episodes, evaluate, train as tr
from xviral.model.config import TrainCfg
from xviral.model.features import FeatureTable
from xviral.model.model import Surrogate

log = logging.getLogger("gen")


def whole_accounts(mask: np.ndarray, author: np.ndarray, budget: int, rng) -> np.ndarray:
    """Rows of randomly chosen whole accounts under `mask`, up to about `budget` rows (pairs need an account's rows together)."""
    ua = rng.permutation(np.unique(author[mask]))
    cnt = np.bincount(author[mask], minlength=author.max() + 1)
    keep = ua[np.cumsum(cnt[ua]) <= budget]
    return np.flatnonzero(mask & np.isin(author, keep))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/general/table"); ap.add_argument("--out", required=True)
    ap.add_argument("--roles", default="orig,expand,train"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--evals", type=int, default=8); ap.add_argument("--drop-cat", default=""); ap.add_argument("--init", default=None)
    ap.add_argument("--select", default="val_aud,val_broad"); ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--features", default="ts46"); ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    table = FeatureTable.load(args.table, "cpu")
    if args.drop_cat:
        table = table.without_cat([c for c in args.drop_cat.split(",") if c])
    info = pl.read_parquet(Path(args.table) / "row_info.parquet")
    role, split, aud, author = info["g_role"].to_numpy(), table.t["split"].numpy(), table.t["audience"].numpy(), table.t["author_idx"].numpy()
    bed_eval = info["bed_eval"].to_numpy()
    roles = set(args.roles.split(","))
    trainable = (split == 0) & np.isin(role, list(roles))
    rng = np.random.default_rng(0)             # the same evaluation accounts in every run
    sets = {"val_aud": whole_accounts((split == 1) & aud & (role == "orig"), author, 2 * args.budget, rng),
            "val_broad": whole_accounts((split == 1) & (role == "val"), author, 2 * args.budget, rng),
            "test_aud": whole_accounts((split == 2) & aud & (role == "orig"), author, args.budget, rng),
            "bed": whole_accounts(bed_eval, author, args.budget, rng),
            "train": whole_accounts(trainable, author, args.budget, np.random.default_rng(1))}
    sets = {k: torch.from_numpy(v) for k, v in sets.items()}
    # Rows of other roles keep their place in the table (history, neighbours) but can be neither anchors nor candidates.
    table.t["split"] = torch.where(torch.from_numpy((split == 0) & ~trainable), torch.full_like(table.t["split"], 3), table.t["split"])
    train_rows = torch.from_numpy(np.flatnonzero(trainable))
    log.info("training rows %d of %d accounts (roles %s); evaluation rows %s", len(train_rows), len(np.unique(author[trainable])), sorted(roles), {k: len(v) for k, v in sets.items()})
    mcfg = tr.model_cfg_for(table, "small")
    tcfg = TrainCfg(steps=args.steps, batch_size=64, lr=args.lr, warmup_steps=args.warmup, seed=args.seed)
    torch.manual_seed(args.seed); gen = torch.Generator().manual_seed(args.seed)
    model = Surrogate(mcfg, table.meta["action_index"])
    if args.init:
        missing, unexpected = model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["state"], strict=False)
        log.info("warm start from %s (missing %d, unexpected %d)", args.init, len(missing), len(unexpected))
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(tr.param_groups(model, tcfg.weight_decay, 1.0), lr=tcfg.lr, betas=(0.9, 0.95))
    points = sorted({int(round(args.steps * (i + 1) / args.evals)) for i in range(args.evals)})
    sel = args.select.split(",")
    curves, best = [], {"sel": float("inf")}
    perm = torch.randperm(len(train_rows), generator=gen); cursor = 0; t0 = time.time(); running: dict = {}

    def evaluate_all(step: int) -> dict:
        rec = {"step": step, "train_loss_running": dict(running)}
        for name, rows in sets.items():
            m = tr.validate(ema, table, rows, tcfg)
            rec[name] = {k: m.get(k) for k in ("val_nll", "nll_likes", "nll_views", "pairwise_likes", "pairs_likes", "pairwise_views", "pairs_views", "within_rho_likes", "n")}
        rec["sel"] = float(np.mean([rec[s]["val_nll"] for s in sel]))
        return rec

    if args.init:
        curves.append(evaluate_all(0)); log.info("step 0: %s", {k: (round(v["nll_likes"], 4), round(v["pairwise_likes"], 4)) for k, v in curves[-1].items() if isinstance(v, dict) and "n" in v})
    for step in range(args.steps):
        if cursor + 64 > len(perm):
            perm = torch.randperm(len(train_rows), generator=gen); cursor = 0
        anchors = train_rows[perm[cursor:cursor + 64]]; cursor += 64
        model.train()
        for g in opt.param_groups:
            g["lr"] = tr.lr_at(step, tcfg) * g["ratio"]
        ep = episodes.build(table, anchors, mcfg, tcfg, train=True, generator=gen)
        loss, parts = model.loss(ep, model(table, ep), tcfg)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], tcfg.grad_clip); opt.step()
        with torch.no_grad():
            d = min(tcfg.ema_decay, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        for k, v in parts.items():
            running[k] = 0.98 * running.get(k, v) + 0.02 * v
        if (step + 1) % 200 == 0:
            log.info("step %d loss %.4f %.2f it/s", step + 1, float(loss), (step + 1) / (time.time() - t0))
        if step + 1 in points:
            rec = evaluate_all(step + 1); curves.append(rec)
            log.info("eval @%d sel %.4f: %s", step + 1, rec["sel"], {k: (round(v["nll_likes"], 4), round(v["pairwise_likes"], 4)) for k, v in rec.items() if isinstance(v, dict) and "n" in v})
            if rec["sel"] < best["sel"]:
                best = rec; tr.save(out / "best.pt", ema, mcfg, tcfg, table.meta, {"step": step + 1, "sel": rec["sel"]})
            (out / "curves.json").write_text(json.dumps({"args": vars(args), "train_rows": int(len(train_rows)), "curves": curves, "best_step": best.get("step")}, indent=1, default=float))
    model, tcfg2, ck = tr.load(out / "best.pt", "cpu")
    temps = evaluate.fit_temperature(model, table, sets["val_aud"], tcfg2)
    tr.save(out / "best.pt", model, model.cfg, tcfg2, ck["table_meta"], ck["metrics"], temps)
    print("TRAIN_DONE", args.out, "best step", best.get("step"), "minutes", round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
