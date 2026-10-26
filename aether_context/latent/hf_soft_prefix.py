"""HF soft-prefix capsule backend: one frozen causal LM as a same-model sender/receiver pair.

Requires the ``[latent]`` extra. Every base-model parameter has
``requires_grad=False``; only the projector trains. The receiver stays
differentiable because the prefix enters through ``inputs_embeds``.

Sender: mid-layer hidden states of the source text -> projector -> ``slots``
prefix vectors, stored as float16 bytes referenced by ``LatentCapsuleV1``.
Receiver: admits the capsule (grant + digest + shape), splices the slots into the
chat template right before the user's question and records ``LatentUseReceiptV1``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from ..contracts import NamespaceV1, digest
from ..contracts.latent_v1 import ContextHandoffGrantV1, LatentCapsuleV1, LatentUseReceiptV1
from . import handoff
from .projector import SoftPrefixProjector
from .tensor_codec import load_tensor, tensor_bytes

CODEC_ID = "hf-soft-prefix"
CODEC_VERSION = 1
_MARK = "\u2063"  # INVISIBLE SEPARATOR: splits the rendered template, never tokenized


@dataclass(frozen=True)
class Layout:
    before: list[int]
    after: list[int]

    def boundary_digest(self) -> str:
        return digest({"after": self.after, "before": self.before})


@dataclass(frozen=True)
class Admission:
    prefix: Optional[torch.Tensor]
    receipt: LatentUseReceiptV1


class HFSoftPrefixBackend:
    """Frozen same-model sender/receiver pair with a trainable soft-prefix projector.

    The backend owns ``tokenizer``: it sets right padding (and a pad token when missing).
    The adapter identity binds the projector weights and :meth:`codec_config`, so a capsule
    made under another layer, slot count, system prompt, storage dtype or model is refused.
    """

    injection_point = "soft_prefix"

    def __init__(self, model, tokenizer, *, model_id: str, slots: int = 8, sender_layer: Optional[int] = None,
                 d_attn: int = 512, system: Optional[str] = None, storage_dtype: str = "float16",
                 seed: int = 0) -> None:
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.model_id = model_id
        self.slots = slots
        self.d_attn = d_attn
        self.system = system
        self.storage_dtype = storage_dtype
        layers = int(model.config.num_hidden_layers)
        self.sender_layer = (2 * layers) // 3 if sender_layer is None else sender_layer
        if not 1 <= self.sender_layer <= layers:
            raise ValueError("sender_layer must index a transformer block output")
        self._base = getattr(model, model.base_model_prefix)
        self._embed = model.get_input_embeddings()
        self._head = model.get_output_embeddings()
        self.width = int(self._embed.weight.shape[1])
        embed_rms = float(self._embed.weight.detach().float().pow(2).mean(-1).sqrt().mean())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.projector = SoftPrefixProjector(self.width, slots, d_attn, embed_rms=embed_rms)

    @classmethod
    def from_pretrained(cls, model_ref: str, *, revision: Optional[str] = None, **kw) -> "HFSoftPrefixBackend":
        """Load a pinned model; ``revision`` (a commit sha) is required so ``model_id`` names exact weights."""
        if not revision:
            raise ValueError("from_pretrained needs a pinned revision")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_ref, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(model_ref, revision=revision, dtype=torch.float32)
        return cls(model, tok, model_id=f"{model_ref}@{revision}", **kw)

    # -- identity --------------------------------------------------------------------------
    def codec_config(self) -> dict:
        """Everything besides the projector weights that shapes what a capsule means."""
        return {"codec_id": CODEC_ID, "codec_version": CODEC_VERSION, "model_id": self.model_id,
                "sender_layer": self.sender_layer, "slots": self.slots, "d_attn": self.d_attn,
                "system": self.system, "storage_dtype": self.storage_dtype}

    def codebook_digest(self) -> str:
        h = hashlib.sha256(digest(self.codec_config()).encode())
        for name, t in sorted(self.projector.state_dict().items()):
            arr = t.detach().cpu().float().contiguous().numpy()
            h.update(name.encode() + b"\0" + repr(tuple(arr.shape)).encode() + b"\0" + arr.tobytes())
        return h.hexdigest()

    @property
    def adapter_id(self) -> str:
        return f"{CODEC_ID}@{self.codebook_digest()[:16]}"

    def base_digest(self) -> str:
        h = hashlib.sha256()
        for name, t in self.model.state_dict().items():
            h.update(name.encode() + b"\0")
            h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy())  # buffer, no copy
        return h.hexdigest()

    def flop_params(self) -> tuple[int, int]:
        """(transformer-block params excluding input embeddings, LM-head matmul params)."""
        blocks = sum(p.numel() for p in self._base.parameters()) - self._embed.weight.numel()
        return int(blocks), int(self._head.weight.numel())

    def save_projector(self, path: "str | Path") -> None:
        torch.save({"config": self.codec_config(), "state": self.projector.state_dict()}, Path(path))

    def load_projector(self, path: "str | Path") -> None:
        saved = torch.load(Path(path), weights_only=True)
        if saved.get("config") != self.codec_config():
            raise ValueError("projector was trained under a different codec config")
        self.projector.load_state_dict(saved["state"])
        self.projector.eval()

    # -- sender ----------------------------------------------------------------------------
    @torch.no_grad()
    def sender_states(self, texts: Sequence[str]) -> list[torch.Tensor]:
        enc = self.tokenizer(list(texts), add_special_tokens=False, padding=True, return_tensors="pt")
        out = self._base(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True,
                         use_cache=False)
        hidden = out.hidden_states[self.sender_layer].float()
        return [hidden[i, :n].clone() for i, n in enumerate(enc.attention_mask.sum(-1).tolist())]

    def project(self, states: Sequence[torch.Tensor]) -> torch.Tensor:
        longest = max(s.shape[0] for s in states)
        batch = torch.zeros(len(states), longest, self.width)
        mask = torch.zeros(len(states), longest, dtype=torch.bool)
        for i, s in enumerate(states):
            batch[i, : s.shape[0]] = s
            mask[i, : s.shape[0]] = True
        return self.projector(batch, mask)

    @torch.no_grad()
    def encode_prefixes(self, texts: Sequence[str]) -> np.ndarray:
        return self.project(self.sender_states(texts)).float().numpy()

    def make_capsule(self, prefix: np.ndarray, *, namespace: NamespaceV1, capsule_id: str, anchor_digest: str,
                     base_revision: str, created_at: int, codebook_digest: Optional[str] = None,
                     ) -> tuple[LatentCapsuleV1, bytes]:
        ref, raw = tensor_bytes(prefix, self.storage_dtype)
        book = codebook_digest or self.codebook_digest()
        capsule = LatentCapsuleV1(
            capsule_id=capsule_id, namespace=namespace, source_model_id=self.model_id,
            source_adapter_id=f"{CODEC_ID}@{book[:16]}", codec_id=CODEC_ID, codec_version=CODEC_VERSION,
            codebook_digest=book, slots=int(prefix.shape[0]), tensor=ref, anchor_digest=anchor_digest,
            base_revision=base_revision, created_at=created_at,
        )
        return capsule, raw

    def encode_capsule(self, text: str, *, namespace: NamespaceV1, capsule_id: str, anchor_digest: str,
                       base_revision: str, created_at: int) -> tuple[LatentCapsuleV1, bytes]:
        return self.make_capsule(self.encode_prefixes([text])[0], namespace=namespace, capsule_id=capsule_id,
                                 anchor_digest=anchor_digest, base_revision=base_revision, created_at=created_at)

    # -- receiver --------------------------------------------------------------------------
    def layout(self, question: str) -> Layout:
        messages = ([{"role": "system", "content": self.system}] if self.system else []) + [
            {"role": "user", "content": _MARK + question}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if rendered.count(_MARK) != 1:
            raise ValueError("chat template must render the user content exactly once")
        before, after = rendered.split(_MARK)
        return Layout(before=self._ids(before), after=self._ids(after))

    def _ids(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def _sequence(self, prefix: Optional[torch.Tensor], question: str, context: Optional[str] = None,
                  answer: Optional[list[int]] = None) -> tuple[torch.Tensor, Layout]:
        lay = self.layout(question)
        emb = lambda ids: self._embed(torch.tensor(ids, dtype=torch.long))  # noqa: E731
        parts = [emb(lay.before)]
        if prefix is not None:
            parts.append(prefix.to(parts[0].dtype))
        if context is not None:
            parts.append(emb(self._ids(context + "\n\n")))
        parts.append(emb(lay.after + (answer or [])))
        return torch.cat(parts, dim=0), lay

    @staticmethod
    def _pad(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        longest = max(s.shape[0] for s in seqs)
        batch = torch.zeros(len(seqs), longest, seqs[0].shape[1], dtype=seqs[0].dtype)
        mask = torch.zeros(len(seqs), longest, dtype=torch.long)
        for i, s in enumerate(seqs):
            batch[i, : s.shape[0]] = s
            mask[i, : s.shape[0]] = 1
        return batch, mask

    def admit(self, capsule: LatentCapsuleV1, grant: ContextHandoffGrantV1, tensor: bytes, question: str, *,
              receipt_id: str, now: int, principal_id: Optional[str] = None,
              current_fence: Optional[int] = None, adapter_id: Optional[str] = None) -> Admission:
        """Admit ``capsule`` for one question, or refuse it; always returns a receipt.

        ``adapter_id`` lets a caller pass a cached :attr:`adapter_id` (it must be this backend's
        current id; it is trusted, not recomputed). The receipt describes the unpadded receiver
        sequence for ``question``; score it with the returned prefix and the same question.
        """
        adapter = adapter_id or self.adapter_id
        code = handoff.admission_reason(capsule, grant, tensor, model_id=self.model_id, adapter_id=adapter,
                                        width=self.width, now=now, principal_id=principal_id,
                                        current_fence=current_fence)
        if code is not None:
            return Admission(None, handoff.refused_receipt(grant, receipt_id=receipt_id, model_id=self.model_id,
                                                           adapter_id=adapter, reason_code=code, used_at=now))
        prefix = torch.from_numpy(load_tensor(capsule.tensor, tensor)[: grant.max_slots].astype(np.float32))
        lay = self.layout(question)
        start = len(lay.before)
        length = start + int(prefix.shape[0]) + len(lay.after)
        receipt = handoff.injected_receipt(
            grant, receipt_id=receipt_id, model_id=self.model_id, adapter_id=adapter,
            positions=list(range(start, start + prefix.shape[0])), template_boundary_digest=lay.boundary_digest(),
            mask_digest=digest({"mask": [1] * length}), used_at=now)
        return Admission(prefix, receipt)

    @torch.no_grad()
    def next_token_logprobs(self, prefixes: Sequence[Optional[torch.Tensor]], questions: Sequence[str],
                            contexts: Optional[Sequence[Optional[str]]] = None) -> torch.Tensor:
        """Log-probabilities of the first answer token, one row per item."""
        ctx = list(contexts) if contexts is not None else [None] * len(questions)
        seqs = [self._sequence(p, q, c)[0] for p, q, c in _strict_zip(prefixes, questions, ctx)]
        batch, mask = self._pad(seqs)
        hidden = self._base(inputs_embeds=batch, attention_mask=mask, use_cache=False).last_hidden_state
        last = hidden[torch.arange(len(seqs)), mask.sum(-1) - 1]
        return torch.log_softmax(self._head(last).float(), dim=-1)

    def answer_loss(self, prefixes: torch.Tensor, questions: Sequence[str], answers: Sequence[str],
                    ) -> tuple[torch.Tensor, dict]:
        """Mean token cross-entropy of ``answers`` given prefix + question (grad flows to ``prefixes``)."""
        seqs, targets, starts = [], [], []
        for prefix, q, a in _strict_zip(prefixes, questions, answers):
            ids = self._ids(a)
            if not ids:
                raise ValueError(f"answer {a!r} tokenizes to nothing")
            seq, _ = self._sequence(prefix, q, answer=ids)
            seqs.append(seq)
            targets.append(ids)
            starts.append(seq.shape[0] - len(ids))
        batch, mask = self._pad(seqs)
        hidden = self._base(inputs_embeds=batch, attention_mask=mask, use_cache=False).last_hidden_state
        rows = [hidden[i, s - 1: s - 1 + len(t)] for i, (s, t) in enumerate(zip(starts, targets))]
        logits = self._head(torch.cat(rows, dim=0)).float()
        gold = torch.tensor([tok for t in targets for tok in t], dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(logits, gold)
        return loss, {"tokens": int(mask.sum()), "head_tokens": int(gold.numel())}

    @torch.no_grad()
    def generate(self, prefix: Optional[torch.Tensor], question: str, *, max_tokens: int = 32) -> str:
        seq, _ = self._sequence(prefix, question)
        out = self.model.generate(inputs_embeds=seq[None], attention_mask=torch.ones(1, seq.shape[0], dtype=torch.long),
                                  max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.0,
                                  temperature=None, top_p=None, top_k=None,
                                  pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def receive(self, capsule: LatentCapsuleV1, grant: ContextHandoffGrantV1, tensor: bytes, prompt: str, *,
                receipt_id: str, now: int, max_tokens: int = 32, principal_id: Optional[str] = None,
                current_fence: Optional[int] = None) -> tuple[Optional[str], LatentUseReceiptV1]:
        adm = self.admit(capsule, grant, tensor, prompt, receipt_id=receipt_id, now=now,
                         principal_id=principal_id, current_fence=current_fence)
        if adm.prefix is None:
            return None, adm.receipt
        return self.generate(adm.prefix, prompt, max_tokens=max_tokens), adm.receipt


def _strict_zip(*seqs: Sequence) -> zip:
    """``zip`` that refuses unequal lengths with ValueError (items must never silently drop)."""
    if len({len(seq) for seq in seqs}) != 1:
        raise ValueError("batched inputs must have equal lengths")
    return zip(*seqs)


__all__ = ["CODEC_ID", "CODEC_VERSION", "Layout", "Admission", "HFSoftPrefixBackend"]
