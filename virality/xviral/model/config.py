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
        # log10(views + 1) and log10(likes + 1): absolute outcomes, defined whenever the count exists
        # (some sources have no follower counts). Same-author ranking is trained and scored on these.
        "views": HistHead(lo=0.0, hi=9.0, bins=90, weight=1.0),
        "likes": HistHead(lo=0.0, hi=7.0, bins=70, weight=1.0),
        # log10((views + 1) / (followers + 1)): reach relative to followers, X's exploration scale.
        "reach": HistHead(lo=-4.0, hi=4.0, bins=80, weight=0.5),
        # log10((likes + 1) / (followers + 1)): defined for every source, the bridge between corpora.
        "like": HistHead(lo=-6.0, hi=2.0, bins=80, weight=0.5),
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
    use_hist_ridge: bool = False
    """Per-account adaptation layer: a ridge regression from a learned low-dimensional projection of each history post to
    its outcome (relative to the account's median), solved in closed form inside the forward pass on the account's visible
    history and applied to the candidates (meta-learning with a differentiable closed-form solver, Bertinetto et al. 2019)."""
    ridge_dim: int = 16
    # Stage B: a LoRA-tuned content encoder on candidates. None = stage A (frozen embeddings only).
    live: dict | None = None
    """Keyword arguments for xviral.model.live.LiveEncoder, e.g. {"model_id": "Qwen/Qwen3-VL-Embedding-8B", "r": 16}."""


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
    train_audience_only: bool = False
    """Train on audience accounts only (ablation: what the general accounts add). Default trains on all rows."""
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
    pair_loss: str = "ranknet"
    """Same-author objective over an episode's candidates (research/tracks/sota):
    ranknet    logistic loss on the difference of predicted means, hard labels, pairs under pair_margin dropped (the baseline);
    soft       the same logit against a soft label Phi(dy / pair_noise_sd): the probability that the observed order is the
               real one given the repost noise, so near-ties count as near-coin-flips instead of being dropped or trusted;
    thurstone  heteroscedastic: P(i beats j) = Phi(k * (mu_i - mu_j) / sqrt(var_i + var_j)) with the variances of the two
               predicted like distributions, so pairs the model itself calls noisy pull less on the means;
    listnet    listwise: cross-entropy between softmax(y / list_temp) and softmax(k * mu) over the episode's candidates."""
    pair_noise_sd: float = 0.34
    """SD (log10) of the difference between two outcomes of the same content by one account: an account reposting identical
    text lands within 2x 62% of the time (tracks/pairs), which is |N(0, sd)| < 0.301 at sd = 0.34."""
    list_temp: float = 0.3
    # Regularisation of the in-context pathway (training only).
    hist_outcome_drop: float = 0.0
    """Each visible history outcome is hidden with this probability."""
    hist_drop_all: float = 0.0
    """The whole history of an episode is removed with this probability (the account is scored cold)."""
    # Masked-history auxiliary task: hidden history outcomes are predicted at their own positions.
    mask_hist_p: float = 0.0
    w_mask: float = 0.5
    # Snapshot ensembling / stochastic weight averaging: the cosine schedule restarts lr_cycles times; a snapshot is saved at the end of each.
    lr_cycles: int = 1
    save_snapshots: bool = False
    # Controlled pairs (two versions of one message by one account, scored as alternative drafts; model/pairs.py).
    w_ctl: float = 0.0
    """Weight of the loss on mined controlled pairs; 0 trains without them."""
    ctl_batch: int = 64
    ctl_loss: str = "ranknet"
    """ranknet: logistic loss on the difference of predicted mean log likes. margin: hinge with the real gap as margin."""
    ctl_weight: str = "gap"
    """gap: a pair's weight grows with its real like ratio (capped at 5x). flat: every pair above ctl_min_ratio counts once."""
    ctl_min_ratio: float = 1.25
    # Live-forecast conditioning (off by default; see MODEL.md section 6).
    live_lr: float = 1e-4
    """Learning rate of the stage-B encoder adapters."""
    p_live: float = 0.0
    live_half_life_hours: tuple[float, float] = (0.5, 8.0)
    amp: bool = True
