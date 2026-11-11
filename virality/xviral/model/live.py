"""Stage B: the content encoder trained through the model's own loss (MODEL.md section 8).

The frozen post embedding stays the shared space for history, neighbours and similarity. On top of
it, each candidate gets a second content vector from the same encoder with LoRA adapters, trained
end to end by the pairwise and distributional losses, so it learns what the frozen embedding does
not encode: which phrasing, framing and media win for this audience. History keeps the frozen
vectors, so no cache has to be refreshed while the adapters move.

The encoder is loaded through sentence-transformers and fed exactly what
phoenix/reference/mm_encoder.py feeds it: the rendered post, with images as conversation content
for the multimodal model. Text-only encoders take the rendered string.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value")


class LiveEncoder(nn.Module):
    def __init__(self, model_id: str, r: int = 16, alpha: int = 32, dropout: float = 0.05, max_tokens: int = 256,
                 prompt: str | None = None, grad_checkpointing: bool = False, device: str = "cpu", dtype: str | None = None):
        super().__init__()
        try:
            from peft import LoraConfig, get_peft_model
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("stage B needs the 'encoders' extra plus peft") from e
        kw = {"model_kwargs": {"dtype": dtype}} if dtype else {}
        self.st = SentenceTransformer(model_id, device=device, **kw)
        self.st.max_seq_length = max_tokens
        self.prompt = prompt
        base = self.st[0].auto_model
        for p in base.parameters():
            p.requires_grad_(False)
        targets = sorted({n.split(".")[-1] for n, m in base.named_modules() if isinstance(m, nn.Linear) and n.split(".")[-1] in LORA_TARGETS})
        if not targets:
            raise ValueError(f"no LoRA target modules found in {model_id}")
        self.st[0].auto_model = get_peft_model(base, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets))
        if grad_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()
        self.dim = self.st.get_embedding_dimension() if hasattr(self.st, "get_embedding_dimension") else self.st.get_sentence_embedding_dimension()
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info("live encoder %s: LoRA r=%d on %s, %.2fM trainable, output %d", model_id, r, targets, n_train / 1e6, self.dim)

    def forward(self, inputs: list) -> torch.Tensor:
        """inputs: rendered strings, or conversations (lists of role/content dicts) for multimodal posts."""
        if self.prompt:
            inputs = [self.prompt + x if isinstance(x, str) else x for x in inputs]
        dev = next(self.parameters()).device
        feats = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in self.st.tokenize(inputs).items()}
        return self.st(feats)["sentence_embedding"].float()

    def encode_chunked(self, inputs: list, chunk: int = 64) -> torch.Tensor:
        return torch.cat([self(inputs[i:i + chunk]) for i in range(0, len(inputs), chunk)]) if inputs else torch.zeros(0, self.dim)


def trainable_state(model: nn.Module) -> dict:
    """State dict without the frozen encoder weights: adapters and everything outside the encoder."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad or not n.startswith("live_enc.")}
    return {k: v for k, v in model.state_dict().items() if k in keep or (not k.startswith("live_enc."))}
