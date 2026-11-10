"""The trunk: Phoenix's transformer block in PyTorch.

Mirrors phoenix/xrex/models/transformer.py and layers.py as configured for
`home_direct_packed` (phoenix/xrex/configs/xrecsys.py:1041-1077):
  pre-norm RMSNorm (eps 1e-5), grouped-query attention (more query heads than
  key/value heads), RMS norm on queries and keys across all heads (qk_norm),
  1-d rotary positions with base 10000, non-gated FFN with widening 2 and
  tanh-approximate GELU, no biases, and a final RMSNorm.
Candidate isolation (phoenix/README.md "Candidate Isolation in Ranking"):
  context and history attend to each other; each candidate attends to context,
  history and itself, never to another candidate. Scores therefore do not depend
  on which other drafts are in the batch.
Positions are right-anchored (recsys_model.py::right_anchored_rope_positions):
  the newest history post sits next to the candidates, and all candidates share
  one position.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(positions: torch.Tensor, dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin of shape [S, dim/2] for integer positions [S]."""
    inv = base ** (-torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim)
    ang = positions.float()[:, None] * inv[None]
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, heads, S, dk]. Rotates pairs (even, odd)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, d: int, n_q: int, n_kv: int, dropout: float):
        super().__init__()
        assert d % n_q == 0 and n_q % n_kv == 0 and (d // n_q) % 2 == 0
        self.n_q, self.n_kv, self.dk = n_q, n_kv, d // n_q
        self.q = nn.Linear(d, n_q * self.dk, bias=False)
        self.k = nn.Linear(d, n_kv * self.dk, bias=False)
        self.v = nn.Linear(d, n_kv * self.dk, bias=False)
        self.out = nn.Linear(n_q * self.dk, d, bias=False)
        self.q_norm = RMSNorm(n_q * self.dk)   # all-heads qk norm (normalization.py::all_heads_qk_norm)
        self.k_norm = RMSNorm(n_kv * self.dk)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, allowed: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_norm(self.q(x)).view(B, S, self.n_q, self.dk).transpose(1, 2)
        k = self.k_norm(self.k(x)).view(B, S, self.n_kv, self.dk).transpose(1, 2)
        v = self.v(x).view(B, S, self.n_kv, self.dk).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.n_q // self.n_kv
        if rep > 1:
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None], dropout_p=self.dropout if self.training else 0.0)
        return self.out(o.transpose(1, 2).reshape(B, S, self.n_q * self.dk))


class Block(nn.Module):
    def __init__(self, d: int, n_q: int, n_kv: int, ffn_mult: float, dropout: float):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, n_q, n_kv, dropout)
        h = int(d * ffn_mult)
        self.up, self.down = nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, allowed, cos, sin):
        x = x + self.drop(self.attn(self.norm1(x), allowed, cos, sin))
        return x + self.drop(self.down(F.gelu(self.up(self.norm2(x)), approximate="tanh")))


def isolation_mask(valid: torch.Tensor, n_shared: int) -> torch.Tensor:
    """allowed[b, q, k]: key k is visible to query q.

    valid: [B, S] token validity. The first `n_shared` positions are context and history, the rest
    are candidates. A key must be valid and be either shared or the query itself.
    """
    B, S = valid.shape
    shared = torch.arange(S, device=valid.device) < n_shared
    eye = torch.eye(S, dtype=torch.bool, device=valid.device)
    return valid[:, None, :] & (shared[None, None, :] | eye[None])


def right_anchored_positions(n_ctx: int, n_hist: int, n_cand: int, device) -> torch.Tensor:
    """Context at 0.., history at n_ctx.., every candidate at n_ctx + n_hist.

    History is stored right-aligned (padding on the left), so its newest post is always at the
    position just before the candidates, as in Phoenix.
    """
    pos = torch.arange(n_ctx + n_hist + n_cand, device=device)
    pos[n_ctx + n_hist:] = n_ctx + n_hist
    return pos


class Trunk(nn.Module):
    def __init__(self, d: int, n_layers: int, n_q: int, n_kv: int, ffn_mult: float, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, n_q, n_kv, ffn_mult, dropout) for _ in range(n_layers))
        self.final_norm = RMSNorm(d)
        self.dk = d // n_q

    def forward(self, x: torch.Tensor, valid: torch.Tensor, n_ctx: int, n_hist: int) -> torch.Tensor:
        S = x.shape[1]
        allowed = isolation_mask(valid, n_ctx + n_hist)
        cos, sin = rope_tables(right_anchored_positions(n_ctx, n_hist, S - n_ctx - n_hist, x.device), self.dk)
        for blk in self.blocks:
            x = blk(x, allowed, cos, sin)
        return self.final_norm(x)
