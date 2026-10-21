"""Additive latent-handoff manifests (W7 seam; lane L1 delivers the contracts only).

Manifests hold integers, strings and hashes only. Tensors are never inlined:
``TensorRefV1`` references them by sha256 + dtype + shape + byte_count, so
``canonical``/``digest`` from the hosted V1 module apply unchanged. Nothing in
the hosted V1 contract set is modified, re-exported or re-versioned by this
module, and the NumPy-only core never imports it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from . import Closed, Digest, Ident, NamespaceV1

DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1, "uint8": 1}
MAX_GRANT_SECONDS = 7 * 86400


class TensorRefV1(Closed):
    """A hashed byte object: identity, element type and shape, never the bytes."""

    schema_version: Literal["LatentTensorRefV1"] = "LatentTensorRefV1"
    digest: Digest
    dtype: Literal["float16", "bfloat16", "float32", "int8", "uint8"]
    shape: list[int] = Field(min_length=1, max_length=8)
    byte_count: int = Field(ge=1)

    @model_validator(mode="after")
    def byte_count_matches(self) -> "TensorRefV1":
        if any(dim < 1 for dim in self.shape):
            raise ValueError("tensor shape dimensions must be positive")
        elements = 1
        for dim in self.shape:
            elements *= dim
        if elements * DTYPE_BYTES[self.dtype] != self.byte_count:
            raise ValueError("tensor byte_count does not match dtype and shape")
        return self


class LatentCapsuleV1(Closed):
    """A learned, source-linked capsule. Not lossless; the exact anchor is the IR."""

    schema_version: Literal["LatentCapsuleV1"] = "LatentCapsuleV1"
    capsule_id: Ident
    namespace: NamespaceV1
    source_model_id: str = Field(min_length=1, max_length=200)
    source_adapter_id: Ident
    codec_id: Ident
    codec_version: int = Field(ge=1)
    codebook_digest: Digest | None = None
    slots: int = Field(ge=1, le=4096)
    tensor: TensorRefV1
    anchor_digest: Digest
    base_revision: Digest
    created_at: int = Field(ge=0)
    lossless: Literal[False] = False

    @model_validator(mode="after")
    def slots_lead_tensor(self) -> "LatentCapsuleV1":
        if self.tensor.shape[0] != self.slots:
            raise ValueError("capsule slots must equal the tensor's leading dimension")
        return self


class ContextHandoffGrantV1(Closed):
    """Short-lived authority for one recipient to inject one capsule."""

    schema_version: Literal["ContextHandoffGrantV1"] = "ContextHandoffGrantV1"
    grant_id: Ident
    namespace: NamespaceV1
    capsule_digest: Digest
    anchor_digest: Digest
    sender_principal_id: Ident
    recipient_principal_id: Ident
    recipient_model_id: str = Field(min_length=1, max_length=200)
    recipient_adapter_id: Ident
    max_slots: int = Field(ge=1, le=4096)
    fence: int = Field(ge=1)
    policy_digest: Digest
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def bounded_grant(self) -> "ContextHandoffGrantV1":
        if self.sender_principal_id == self.recipient_principal_id:
            raise ValueError("handoff sender and recipient must differ")
        if self.expires_at <= self.issued_at or self.expires_at - self.issued_at > MAX_GRANT_SECONDS:
            raise ValueError("handoff grant must be short-lived")
        return self


class LatentUseReceiptV1(Closed):
    """What the receiver actually injected, or why it did not."""

    schema_version: Literal["LatentUseReceiptV1"] = "LatentUseReceiptV1"
    receipt_id: Ident
    namespace: NamespaceV1
    grant_digest: Digest
    capsule_digest: Digest
    recipient_model_id: str = Field(min_length=1, max_length=200)
    recipient_adapter_id: Ident
    injection_point: Literal["soft_prefix", "embedding_slots", "kv_update"]
    slots_injected: int = Field(ge=0, le=4096)
    positions: list[int] = Field(default_factory=list, max_length=4096)
    template_boundary_digest: Digest
    mask_digest: Digest
    outcome: Literal["INJECTED", "REFUSED", "NOT_RUN"]
    reason_code: str | None = Field(default=None, min_length=1, max_length=100)
    fence: int = Field(ge=1)
    used_at: int = Field(ge=0)

    @model_validator(mode="after")
    def outcome_evidence(self) -> "LatentUseReceiptV1":
        if len(self.positions) != self.slots_injected:
            raise ValueError("positions must list exactly the injected slots")
        if any(p < 0 for p in self.positions) or len(set(self.positions)) != len(self.positions):
            raise ValueError("positions must be unique and non-negative")
        if self.outcome == "INJECTED" and (self.slots_injected < 1 or self.reason_code is not None):
            raise ValueError("INJECTED needs injected slots and no reason code")
        if self.outcome != "INJECTED" and (self.slots_injected != 0 or self.reason_code is None):
            raise ValueError("REFUSED/NOT_RUN carry a reason code and inject nothing")
        return self


__all__ = ["TensorRefV1", "LatentCapsuleV1", "ContextHandoffGrantV1", "LatentUseReceiptV1"]
