"""Turn model outputs into the report the product shows (ARCHITECTURE.md section 10).

Works on rows of a feature table: held-out posts for backtests, or drafts once
the serving path appends them to the table. All quantities are read off the
calibrated distributions; X's score is computed by the ported value model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from xviral.model import episodes
from xviral.model import train as tr
from xviral.model.features import FeatureTable
from xviral.xalgo import value_model as vm


class Predictor:
    def __init__(self, model, table: FeatureTable, tcfg, temperature: dict | None = None, weights: vm.Weights | None = None):
        self.model, self.table, self.tcfg = model.eval(), table, tcfg
        self.temperature = temperature or {}
        self.weights = weights or vm.Weights.from_repo()
        # A head is trained only if the table carries labels for it (a like-only corpus trains none).
        if "labeled_actions" in table.meta:      # recorded when the training table was built
            self.trained_heads = set(table.meta["labeled_actions"])
        else:
            has_label = (~torch.isnan(table.t["rates"])).any(0).cpu().numpy()
            self.trained_heads = {idx for idx, ok in zip(table.meta["action_index"], has_label) if ok}

    @classmethod
    def load(cls, checkpoint: str | Path, table_dir: str | Path, device: str = "cpu") -> "Predictor":
        model, tcfg, ck = tr.load(checkpoint, device)
        return cls(model, FeatureTable.load(table_dir, device), tcfg, ck.get("temperature"))

    @torch.no_grad()
    def report(self, rows: torch.Tensor, mutual_share: float = 0.0) -> list[dict]:
        """One report per row, each scored alone with its own history (cold start: no counts observed)."""
        ep = episodes.build(self.table, rows, self.model.cfg, self.tcfg, train=False)
        return self._reports(ep, self.model(self.table, ep), rows, 0, mutual_share)

    @torch.no_grad()
    def report_joint(self, first_row: int, n: int, mutual_share: float = 0.0) -> list[dict]:
        """`n` consecutive rows of one account scored as alternatives: one episode, shared history (the posts
        before `first_row`), candidates isolated from each other."""
        anchor = torch.tensor([first_row], device=self.table.device)
        ep = episodes.build(self.table, anchor, self.model.cfg, self.tcfg, train=False, n_candidates=n)
        out = self.model(self.table, ep)
        return [r for c in range(n) for r in self._reports(ep, out, anchor + c, c, mutual_share)]

    def _reports(self, ep, out, rows: torch.Tensor, cand: int, mutual_share: float) -> list[dict]:
        t = self.table.t
        s = {k: v[:, cand].float().cpu().numpy() for k, v in self.model.summarize(ep, out, self.temperature).items()}
        followers = t["followers"][rows].cpu().numpy()
        med_like, med_view = ep.ctx[:, 3].cpu().numpy(), ep.ctx[:, 5].cpu().numpy()
        has_like, has_view = ep.ctx[:, 7].cpu().numpy() > 0, ep.ctx[:, 8].cpu().numpy() > 0
        w_in = vm.head_weight_vector(self.weights, mutual_share=mutual_share, in_network=True)
        w_oon = vm.head_weight_vector(self.weights, in_network=False)
        observed = np.zeros(64, dtype=bool)
        observed[list(self.trained_heads)] = True
        unobserved = float(np.abs(w_in[~observed]).sum() / np.abs(w_in).sum())
        reports = []
        for i in range(len(rows)):
            f = float(followers[i])
            p = s["actions"][i].astype(np.float64) * observed          # heads without labels contribute nothing
            to_count = lambda q: float(10 ** q - 1)                    # noqa: E731  absolute heads, log10(count + 1)
            rep = {
                "post_id": int(t["post_id"][rows[i]]),
                "followers": f,
                "views_24h": {k: max(to_count(s[f"views_{k}"][i]), 0.0) for k in ("q10", "q50", "q90")},
                "likes": {k: max(to_count(s[f"likes_{k}"][i]), 0.0) for k in ("q10", "q50", "q90")},
                "p_reach_1x": float(s["p_reach_1x"][i]), "p_reach_10x": float(s["p_reach_10x"][i]),
                "p_reach_100x": float(s["p_reach_100x"][i]),
                "lift_vs_own_median": {
                    "likes": float(10 ** s["rel_like_q50"][i]) if has_like[i] else None,
                    "views": float(10 ** s["rel_view_q50"][i]) if has_view[i] else None,
                },
                "p_own_top_10pct": {
                    "likes": float(s["p_own_p90_rel_like"][i]) if has_like[i] else None,
                    "views": float(s["p_own_p90_rel_view"][i]) if has_view[i] else None,
                },
                "own_median": {"likes": float(10 ** med_like[i] - 1) if has_like[i] else None,
                               "views": float(10 ** med_view[i] - 1) if has_view[i] else None},
                "action_rates": {name: float(s["actions"][i][idx]) for name, idx in vm.PHOENIX_INDEX.items() if idx in self.trained_heads},
                "x_value_score": {
                    "in_network": float(p @ w_in), "out_of_network": float(vm.offset_score(float(p @ w_oon), self.weights) * self.weights.oon_weight_factor),
                    "breakdown": {name: float(p[idx] * w_in[idx]) for name, idx in vm.PHOENIX_INDEX.items() if idx in self.trained_heads},
                    "unobserved_weight_share": unobserved,
                },
            }
            reports.append(rep)
        return reports
