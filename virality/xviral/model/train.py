"""Training loop: AdamW, warmup + cosine, EMA weights, periodic validation, best checkpoint.

Model selection uses author-held-out validation rows: the mean of the validation
NLLs of the distributional heads, with same-author pairwise accuracy logged
next to it. The checkpoint stores the EMA weights, both configs and the feature
table's metadata, so serving applies the same feature transforms.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from xviral.model import episodes, evaluate
from xviral.model.config import PRESETS, ModelCfg, TrainCfg
from xviral.model.features import FeatureTable
from xviral.model.model import Surrogate

log = logging.getLogger(__name__)


def model_cfg_for(table: FeatureTable, preset: str = "small", **overrides) -> ModelCfg:
    m = table.meta
    kw = dict(PRESETS[preset])
    kw.update(emb_dim=m["emb_dim"], n_num=len(m["num_names"]), cat_cards=m["cat_cards"], aux_dims=m["aux_dims"],
              sid_levels=m["sid_levels"])
    kw.update(overrides)
    if "nbr_idx" not in table.t:
        kw["max_neighbors"] = 0
    return ModelCfg(**kw)


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim >= 2 and "emb" not in name.split(".")[-2:] else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step: int, tcfg: TrainCfg) -> float:
    if step < tcfg.warmup_steps:
        return tcfg.lr * (step + 1) / tcfg.warmup_steps
    p = (step - tcfg.warmup_steps) / max(1, tcfg.steps - tcfg.warmup_steps)
    return tcfg.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def validate(model: Surrogate, table: FeatureTable, rows: torch.Tensor, tcfg: TrainCfg) -> dict:
    pred = evaluate.predict_rows(model, table, rows, tcfg)
    m = evaluate.metrics(pred, table, tcfg)
    nlls = [v for k, v in m.items() if k.startswith("nll_") and not np.isnan(v)]
    m["val_nll"] = float(np.mean(nlls)) if nlls else float("nan")
    return m


def train(table: FeatureTable, out_dir: str | Path, mcfg: ModelCfg, tcfg: TrainCfg, log_every: int = 100) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = table.device
    torch.manual_seed(tcfg.seed)
    gen = torch.Generator(device=dev).manual_seed(tcfg.seed)
    model = Surrogate(mcfg, table.meta["action_index"]).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(param_groups(model, tcfg.weight_decay), lr=tcfg.lr, betas=(0.9, 0.95))
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model: %.2fM parameters, d=%d, layers=%d", n_params / 1e6, mcfg.d_model, mcfg.n_layers)

    train_rows = table.rows_of_split("train")
    val_rows = table.rows_of_split("val")
    if len(val_rows) > tcfg.eval_max_rows:
        val_rows = val_rows[torch.randperm(len(val_rows), device=dev, generator=gen)[:tcfg.eval_max_rows]]
    use_amp = tcfg.amp and dev.type == "cuda"
    history, best = [], {"val_nll": float("inf"), "step": -1}
    perm = torch.randperm(len(train_rows), device=dev, generator=gen)
    cursor = 0
    t0 = time.time()
    running: dict[str, float] = {}
    for step in range(tcfg.steps):
        if cursor + tcfg.batch_size > len(perm):
            perm = torch.randperm(len(train_rows), device=dev, generator=gen)
            cursor = 0
        anchors = train_rows[perm[cursor:cursor + tcfg.batch_size]]
        cursor += tcfg.batch_size
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg)
        ep = episodes.build(table, anchors, mcfg, tcfg, train=True, generator=gen)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            o = model(table, ep)
        loss, parts = model.loss(ep, o, tcfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        with torch.no_grad():
            d = min(tcfg.ema_decay, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        for k, v in parts.items():
            running[k] = 0.98 * running.get(k, v) + 0.02 * v
        if (step + 1) % log_every == 0:
            log.info("step %d  loss %.4f  %s  %.1f it/s", step + 1, float(loss), " ".join(f"{k}={v:.3f}" for k, v in running.items()),
                     (step + 1) / (time.time() - t0))
        if (step + 1) % tcfg.eval_every == 0 or step + 1 == tcfg.steps:
            m = validate(ema, table, val_rows, tcfg) if len(val_rows) else {"val_nll": float("nan")}
            m["step"] = step + 1
            history.append(m)
            log.info("val @%d: nll %.4f  pairwise_like %s  pairwise_reach %s", step + 1, m["val_nll"],
                     m.get("pairwise_like"), m.get("pairwise_reach"))
            if not (m["val_nll"] >= best["val_nll"]):
                best = m
                save(out / "best.pt", ema, mcfg, tcfg, table.meta, m)
    save(out / "last.pt", ema, mcfg, tcfg, table.meta, history[-1] if history else {})
    (out / "history.json").write_text(json.dumps(history, indent=2))
    return {"best": best, "history": history, "params": n_params}


def save(path: Path, model: Surrogate, mcfg: ModelCfg, tcfg: TrainCfg, meta: dict, metrics: dict, temperature: dict | None = None) -> None:
    torch.save({"state": model.state_dict(), "model_cfg": mcfg.model_dump(), "train_cfg": tcfg.model_dump(),
                "table_meta": meta, "metrics": metrics, "temperature": temperature or {}}, path)


def load(path: str | Path, device: str = "cpu") -> tuple[Surrogate, TrainCfg, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    mcfg, tcfg = ModelCfg(**ck["model_cfg"]), TrainCfg(**ck["train_cfg"])
    model = Surrogate(mcfg, ck["table_meta"]["action_index"]).to(device)
    model.load_state_dict(ck["state"])
    return model.eval(), tcfg, ck
