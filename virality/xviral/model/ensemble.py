"""Stage C: several trained surrogates scored as one model.

Members differ by seed or size and are trained on the same feature table. Each member's
distributions are calibrated with its own temperatures, then the probabilities are averaged:
the mixture of the members' categorical distributions for every distributional head, and the
mean probability for every action head. The mixture's log-likelihood on any post is at least the
members' average, and where members disagree the mixture is wider, which is the behaviour the
product's intervals need.

The ensemble returns the log of the averaged probabilities in the place of logits, so everything
written for one model (evaluate.predict_rows, predict.Predictor, serve.DraftScorer) runs on it
unchanged, with no further temperature.
"""

from __future__ import annotations

from pathlib import Path

import torch

from xviral.model import train as tr

EPISODE_FIELDS = ("max_history", "max_candidates", "max_neighbors", "n_ctx", "use_history_outcomes", "use_neighbors")


class Ensemble(torch.nn.Module):
    def __init__(self, models: list, temperatures: list[dict] | None = None):
        super().__init__()
        assert models, "an ensemble needs at least one member"
        self.models = torch.nn.ModuleList(models)
        self.temperatures = temperatures or [{} for _ in models]
        self.cfg = models[0].cfg
        self.action_index = models[0].action_index
        for m in models[1:]:
            # Members read the same episode, so everything that shapes an episode must agree.
            diff = [f for f in EPISODE_FIELDS if getattr(m.cfg, f) != getattr(self.cfg, f)]
            assert not diff and set(m.cfg.hist_heads) == set(self.cfg.hist_heads), f"members disagree on {diff or 'heads'}"
            assert all(m.cfg.hist_heads[k] == self.cfg.hist_heads[k] for k in self.cfg.hist_heads), "members disagree on head bins"

    def forward(self, table, ep) -> dict:
        outs = [m(table, ep) for m in self.models]
        hist = {}
        for name in self.cfg.hist_heads:
            p = torch.stack([torch.softmax(o["hist"][name].float() / T.get(name, 1.0), -1) for o, T in zip(outs, self.temperatures)]).mean(0)
            hist[name] = p.clamp_min(1e-12).log()
        p_act = torch.stack([o["actions"].float().sigmoid() for o in outs]).mean(0)
        return {"actions": torch.logit(p_act.clamp(1e-7, 1 - 1e-7)), "hist": hist}

    def summarize(self, ep, out: dict, temperature: dict | None = None) -> dict:
        return self.models[0].summarize(ep, out, None)        # members' temperatures are already applied


def load(checkpoints: list[str | Path], device: str = "cpu") -> tuple[Ensemble, object, dict]:
    """Returns (ensemble, train config of the first member, checkpoint dict of the first member)."""
    loaded = [tr.load(p, device) for p in checkpoints]
    ens = Ensemble([m for m, _, _ in loaded], [ck.get("temperature") or {} for _, _, ck in loaded]).eval()
    return ens, loaded[0][1], loaded[0][2]
