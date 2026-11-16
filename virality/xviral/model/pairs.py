"""Controlled pairs as a training signal: two versions of one message by one account.

A pair table (scripts/build_pair_table.py) holds, per pair, the account's history followed by the two posts as
drafts at the time of the earlier one, built through the serving path. An episode on it is the serving episode:
shared history, two isolated candidates, no outcome of either visible. The loss asks the like head to order the
two by their real like counts. This is the product's question, trained on directly; the generic same-author pair
loss (model.pair_loss) compares consecutive posts on any topic, each with its own features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from xviral.model import episodes
from xviral.model.config import TrainCfg
from xviral.model.features import FeatureTable
from xviral.model.model import Surrogate, hist_mean


@dataclass
class PairSet:
    table: FeatureTable
    anchor: torch.Tensor      # [P] row of the first draft
    a_is_first: torch.Tensor  # [P] bool: version A (the earlier real post) is candidate 0
    d: torch.Tensor           # [P] ln((likes_a + 1) / (likes_b + 1))
    frame: pl.DataFrame

    def __len__(self) -> int:
        return len(self.anchor)

    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randint(0, len(self), (n,), generator=generator, device=self.anchor.device)


def load(directory: str | Path, drop_num=None, device: str = "cpu") -> tuple[FeatureTable, pl.DataFrame]:
    table = FeatureTable.load(directory, device)
    if drop_num is not None:
        table = table.without_num(drop_num)
    return table, pl.read_parquet(Path(directory) / "pairs.parquet")


def pair_set(table: FeatureTable, frame: pl.DataFrame) -> PairSet:
    dev = table.device
    d = np.log((frame["likes"].to_numpy() + 1.0) / (frame["likes_b"].to_numpy() + 1.0))
    return PairSet(table, torch.from_numpy(frame["anchor"].to_numpy()).to(dev), torch.from_numpy((frame["row_a"] == frame["anchor"]).to_numpy()).to(dev),
                   torch.from_numpy(d.astype(np.float32)).to(dev), frame)


def score_diff(model: Surrogate, ps: PairSet, idx: torch.Tensor, tcfg: TrainCfg) -> torch.Tensor:
    """Predicted mean log10 likes of version A minus version B."""
    ep = episodes.build(ps.table, ps.anchor[idx], model.cfg, tcfg, train=False, n_candidates=2)
    out = model(ps.table, ep)
    s = hist_mean(out["hist"]["likes"].float(), model.cfg.hist_heads["likes"])
    ds = s[:, 0] - s[:, 1]
    return torch.where(ps.a_is_first[idx], ds, -ds)


def loss(model: Surrogate, ps: PairSet, idx: torch.Tensor, tcfg: TrainCfg) -> torch.Tensor:
    d = ps.d[idx]
    ds = score_diff(model, ps, idx, tcfg)
    gap = d.abs()
    use = gap >= math.log(tcfg.ctl_min_ratio)
    if not use.any():
        return ds.sum() * 0.0
    w = (gap.clamp(max=math.log(5.0)) / math.log(2.0) if tcfg.ctl_weight == "gap" else torch.ones_like(gap)) * use
    if tcfg.ctl_loss == "margin":
        # Hinge in log10 likes: the predicted gap should have the right sign and at least half the real gap, up to 0.3.
        margin = (gap / math.log(10.0) * 0.5).clamp(max=0.3)
        per = F.relu(margin - torch.sign(d) * ds) * 4.0
    else:
        per = F.binary_cross_entropy_with_logits(ds * model.pair_log_scale.exp(), (d > 0).float(), reduction="none")
    return (per * w).sum() / w.sum().clamp_min(1e-8)


@torch.no_grad()
def scores(model: Surrogate, ps: PairSet, tcfg: TrainCfg, batch: int = 256) -> np.ndarray:
    model.eval()
    idx = torch.arange(len(ps), device=ps.anchor.device)
    return np.concatenate([score_diff(model, ps, idx[i:i + batch], tcfg).float().cpu().numpy() for i in range(0, len(ps), batch)])


def accuracy(model: Surrogate, ps: PairSet, tcfg: TrainCfg) -> dict:
    s, d = scores(model, ps, tcfg), ps.d.cpu().numpy()
    out = {}
    for g in (1.25, 2.0, 3.0):
        m = np.abs(d) >= math.log(g)
        out[f"ctl_acc_{g}x"] = float(((s * d) > 0)[m].mean()) if m.any() else float("nan")
    if "tier" in ps.frame.columns:
        for tier in ("tac", "strict", "loose"):
            m = (np.abs(d) >= math.log(1.25)) & (ps.frame["tier"] == tier).to_numpy()
            out[f"ctl_acc_{tier}"] = float(((s * d) > 0)[m].mean()) if m.any() else float("nan")
            out[f"ctl_n_{tier}"] = int(m.sum())
    out["ctl_n"] = int((np.abs(d) >= math.log(1.25)).sum())
    return out
