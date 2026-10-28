"""Train only the soft-prefix projector; the base model must be bit-identical afterwards."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from .hf_soft_prefix import HFSoftPrefixBackend

SENDER_BATCH = 16
GRAD_CLIP = 1.0


@dataclass(frozen=True)
class TrainExample:
    document: str
    question: str
    answer: str


@dataclass
class TrainStats:
    steps: int
    examples_seen: int
    batch_size: int
    lr: float
    seed: int
    wall_s: float
    sender_wall_s: float
    sender_tokens: int
    receiver_tokens: int
    head_tokens: int
    trainable_params: int
    base_digest_before: str
    base_digest_after: str
    grad_clip: float
    adapter_id: str
    codebook_digest: str
    sender_layer: int
    slots: int
    losses: list[float] = field(default_factory=list)


def train_projector(backend: HFSoftPrefixBackend, examples: Sequence[TrainExample], *, steps: int,
                    batch_size: int, lr: float, seed: int, log: Optional[Callable[[int, float], None]] = None,
                    log_every: int = 25) -> TrainStats:
    if not examples or steps < 1 or batch_size < 1:
        raise ValueError("training needs examples, steps >= 1 and batch_size >= 1")
    before = backend.base_digest()
    t0 = time.perf_counter()
    docs = sorted({e.document for e in examples})
    cache: dict[str, torch.Tensor] = {}
    sender_tokens = 0
    for i in range(0, len(docs), SENDER_BATCH):
        chunk = docs[i: i + SENDER_BATCH]
        for doc, states in zip(chunk, backend.sender_states(chunk)):
            cache[doc] = states
            sender_tokens += int(states.shape[0])
    sender_wall = time.perf_counter() - t0

    params = list(backend.projector.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    receiver_tokens = head_tokens = 0
    backend.projector.train()
    for step in range(steps):
        batch = [examples[i] for i in torch.randint(len(examples), (batch_size,), generator=gen).tolist()]
        prefixes = backend.project([cache[e.document] for e in batch])
        loss, counts = backend.answer_loss(prefixes, [e.question for e in batch], [e.answer for e in batch])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        losses.append(float(loss.detach()))
        receiver_tokens += counts["tokens"]
        head_tokens += counts["head_tokens"]
        if log is not None and (step + 1) % log_every == 0:
            log(step + 1, sum(losses[-log_every:]) / log_every)
    backend.projector.eval()
    wall = time.perf_counter() - t0
    after = backend.base_digest()
    if after != before:
        raise RuntimeError("base model weights changed during projector training")
    return TrainStats(steps=steps, examples_seen=steps * batch_size, batch_size=batch_size, lr=lr, seed=seed,
                      wall_s=wall, sender_wall_s=sender_wall, sender_tokens=sender_tokens,
                      receiver_tokens=receiver_tokens, head_tokens=head_tokens,
                      trainable_params=sum(p.numel() for p in params), base_digest_before=before,
                      base_digest_after=after, grad_clip=GRAD_CLIP, adapter_id=backend.adapter_id,
                      codebook_digest=backend.codebook_digest(), sender_layer=backend.sender_layer,
                      slots=backend.slots, losses=losses)


__all__ = ["GRAD_CLIP", "TrainExample", "TrainStats", "train_projector"]
