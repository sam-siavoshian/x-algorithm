"""Model and training configuration. Stored inside every checkpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HistHead(BaseModel):
    """A distributional head: a categorical distribution over `bins` equal bins on [lo, hi] (log10 units)."""

    lo: float
    hi: float
    bins: int
    weight: float = 1.0


def default_hist_heads() -> dict[str, HistHead]:
    return {
        # log10((views + 1) / (followers + 1)): reach relative to followers, X's exploration scale.
        "reach": HistHead(lo=-4.0, hi=4.0, bins=80, weight=1.0),
        # log10((likes + 1) / (followers + 1)): defined for every source, the bridge between corpora.
        "like": HistHead(lo=-6.0, hi=2.0, bins=80, weight=1.0),
        # log10 of the outcome over the author's own median in the visible history: "x times your usual".
        "rel_view": HistHead(lo=-3.0, hi=3.0, bins=60, weight=0.5),
        "rel_like": HistHead(lo=-3.0, hi=3.0, bins=60, weight=0.5),
    }


class ModelCfg(BaseModel):
    # Trunk: Phoenix's block (pre-norm RMSNorm, grouped-query attention with qk-norm and RoPE,
    # non-gated GELU FFN with widening 2; phoenix/xrex/configs/xrecsys.py:1041-1077) at reduced width.
    d_model: int = 256
    n_layers: int = 4
    n_q_heads: int = 8
    n_kv_heads: int = 2
    ffn_mult: float = 2.0
    dropout: float = 0.1
    # Episode shape.
    max_history: int = 64
    max_candidates: int = 8
    max_neighbors: int = 16
    # Inputs (filled from the feature table).
    emb_dim: int = 1024
    n_num: int = 0
    cat_cards: list[int] = Field(default_factory=list)
    aux_dims: dict[str, int] = Field(default_factory=dict)
    sid_levels: int = 6
    sid_vocab: int = 257
    n_ctx: int = 12
    # Outputs.
    n_actions: int = 64
    hist_heads: dict[str, HistHead] = Field(default_factory=default_hist_heads)
    hl_gauss_sigma_bins: float = 0.75
    # Ablation switches (used by evaluation, never by the product).
    use_history_outcomes: bool = True
    use_content: bool = True
    use_neighbors: bool = True
    use_hist_knn: bool = True


PRESETS: dict[str, dict] = {
    "nano": dict(d_model=64, n_layers=2, n_q_heads=4, n_kv_heads=2, max_history=16, max_candidates=4, max_neighbors=4),
    "small": dict(d_model=192, n_layers=3, n_q_heads=6, n_kv_heads=2, max_history=48, max_candidates=6, max_neighbors=8),
    "base": dict(d_model=512, n_layers=8, n_q_heads=8, n_kv_heads=4, max_history=96, max_candidates=8, max_neighbors=16),
    "large": dict(d_model=1024, n_layers=12, n_q_heads=16, n_kv_heads=4, max_history=128, max_candidates=8, max_neighbors=16),
}


class TrainCfg(BaseModel):
    steps: int = 20_000
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    eval_every: int = 1000
    eval_max_rows: int = 20_000
    seed: int = 0
    # Episode construction.
    pair_window_days: float = 7.0
    outcome_mask_hours: float = 24.0
    """Outcomes of history posts younger than this at the cutoff are hidden: at serving time
    their counts are still immature."""
    min_history_for_rel: int = 5
    pair_margin: float = 0.1
    """Minimum |difference| in log10 outcome for a same-author pair to count (ties are noise)."""
    # Loss weights.
    w_rates: float = 1.0
    w_pair: float = 1.0
    # Live-forecast conditioning (off by default; see MODEL.md section 6).
    p_live: float = 0.0
    live_half_life_hours: tuple[float, float] = (0.5, 8.0)
    amp: bool = True
