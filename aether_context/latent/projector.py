"""Cross-attention pooler: ``slots`` learned queries read a sender's hidden states.

Output slots are rescaled to the receiver's mean input-embedding RMS so the
injected prefix sits at the magnitude the frozen receiver was trained on.
"""

from __future__ import annotations

import torch
from torch import nn


class SoftPrefixProjector(nn.Module):
    def __init__(self, d_model: int, slots: int, d_attn: int = 512, *, embed_rms: float) -> None:
        super().__init__()
        if d_model < 1 or slots < 1 or d_attn < 1 or not embed_rms > 0:
            raise ValueError("projector sizes and embed_rms must be positive")
        self.slots = slots
        self.d_attn = d_attn
        self.in_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.queries = nn.Parameter(torch.randn(slots, d_attn) / d_attn ** 0.5)
        self.key = nn.Linear(d_model, d_attn)
        self.value = nn.Linear(d_model, d_attn)
        self.out_norm = nn.LayerNorm(d_attn)
        self.out = nn.Linear(d_attn, d_model)
        self.register_buffer("embed_rms", torch.tensor(float(embed_rms)))

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``states`` (B, T, d_model), ``mask`` (B, T) bool -> prefix (B, slots, d_model)."""
        x = self.in_norm(states)
        scores = torch.einsum("sa,bta->bst", self.queries, self.key(x)) / self.d_attn ** 0.5
        scores = scores.masked_fill(~mask[:, None, :], torch.finfo(scores.dtype).min)
        pooled = torch.softmax(scores, dim=-1) @ self.value(x)
        y = self.out(self.out_norm(pooled))
        return y / y.pow(2).mean(dim=-1, keepdim=True).add(1e-12).sqrt() * self.embed_rms


__all__ = ["SoftPrefixProjector"]
