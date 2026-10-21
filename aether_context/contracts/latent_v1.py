"""Additive latent-handoff manifests (W7 seam; lane L1 delivers the contracts only).

Manifests hold integers, strings and hashes only. Tensors are never inlined:
``TensorRefV1`` references them by sha256 + dtype + shape + byte_count, so
``canonical``/``digest`` from the hosted V1 module apply unchanged. Nothing in
the hosted V1 contract set is modified, re-exported or re-versioned by this
module, and the NumPy-only core never imports it.

Requires the optional hosted extra (pydantic).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import Field, model_validator

from . import Closed, Digest, Ident, NamespaceV1, digest

Dtype = Literal["float16", "bfloat16", "float32", "int8", "uint8"]
DTYPE_BYTES = MappingProxyType({"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1, "uint8": 1})
# Largest integer that survives a JSON/IEEE-754 round trip (2**53 - 1), so digests stay portable.
SAFE_INT = 9007199254740991
# Hosted short-lived authority cap, same as RetentionCommandV1.
MAX_GRANT_SECONDS = 900


class TensorRefV1(Closed):
    """A hashed byte object: identity, element type and shape, never the bytes."""

    schema_version: Literal["LatentTensorRefV1"] = "LatentTensorRefV1"
    digest: Digest
    dtype: Dtype
    shape: list[Annotated[int, Field(ge=1, le=SAFE_INT)]] = Field(min_length=1, max_length=8)
    byte_count: int = Field(ge=1, le=SAFE_INT)

    @model_validator(mode="after")
    def byte_count_matches(self) -> "TensorRefV1":
        elements = 1
        for dim in self.shape:
            elements *= dim
        if elements * DTYPE_BYTES[self.dtype] > SAFE_INT:
            raise ValueError("tensor is too large to reference")
        if elements * DTYPE_BYTES[self.dtype] != self.byte_count:
            raise ValueError("tensor byte_count does not match dtype and shape")
        return self


class LatentCapsuleV1(Closed):
    """A learned, source-linked capsule. Not lossless; the exact anchor is the IR."""

    schema_version: Literal["LatentCapsuleV1"] = "LatentCapsuleV1"
    capsule_id: Ident
    namespace: NamespaceV1
    source_model_id: Ident
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
    """Short-lived authority for one recipient to inject one capsule.

    ``fence`` is the lane fence for ``(namespace.context_cycle_id, lane_id)``,
    matching ``CapabilityV1``.
    """

    schema_version: Literal["ContextHandoffGrantV1"] = "ContextHandoffGrantV1"
    grant_id: Ident
    namespace: NamespaceV1
    lane_id: Ident
    capsule_digest: Digest
    anchor_digest: Digest
    sender_principal_id: Ident
    recipient_principal_id: Ident
    recipient_model_id: Ident
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
            raise ValueError("handoff grant must be short-lived (at most 900 seconds)")
        return self


class LatentUseReceiptV1(Closed):
    """What the receiver actually injected, or why it did not.

    ``positions[i]`` is the injected position of capsule slot ``i``, listed in
    increasing order. The receipt carries ``used_at`` only; its lifetime is
    bounded by the grant's window via ``verify_latent_use``. Its ``fence`` must
    equal the grant's (enforced by ``verify_latent_use``).
    """

    schema_version: Literal["LatentUseReceiptV1"] = "LatentUseReceiptV1"
    receipt_id: Ident
    namespace: NamespaceV1
    lane_id: Ident
    grant_digest: Digest
    capsule_digest: Digest
    recipient_model_id: Ident
    recipient_adapter_id: Ident
    injection_point: Literal["soft_prefix", "embedding_slots", "kv_update"]
    slots_injected: int = Field(ge=0, le=4096)
    positions: list[Annotated[int, Field(ge=0, le=SAFE_INT)]] = Field(default_factory=list, max_length=4096)
    template_boundary_digest: Digest | None = None
    mask_digest: Digest | None = None
    outcome: Literal["INJECTED", "REFUSED", "NOT_RUN"]
    reason_code: Ident | None = None
    fence: int = Field(ge=1)
    used_at: int = Field(ge=0)

    @model_validator(mode="after")
    def outcome_evidence(self) -> "LatentUseReceiptV1":
        if len(self.positions) != self.slots_injected:
            raise ValueError("positions must list exactly the injected slots")
        if any(b <= a for a, b in zip(self.positions, self.positions[1:])):
            raise ValueError("positions must be strictly increasing")
        if self.outcome == "INJECTED" and (self.slots_injected < 1 or self.reason_code is not None):
            raise ValueError("INJECTED needs injected slots and no reason code")
        if self.outcome != "INJECTED" and (self.slots_injected != 0 or self.reason_code is None):
            raise ValueError("REFUSED/NOT_RUN carry a reason code and inject nothing")
        has_digests = self.template_boundary_digest is not None and self.mask_digest is not None
        no_digests = self.template_boundary_digest is None and self.mask_digest is None
        if self.outcome == "INJECTED" and not has_digests:
            raise ValueError("INJECTED needs template and mask digests")
        if self.outcome != "INJECTED" and not no_digests:
            raise ValueError("template and mask digests describe an injection that did not happen")
        return self


def verify_latent_use(capsule: LatentCapsuleV1, grant: ContextHandoffGrantV1, receipt: LatentUseReceiptV1) -> None:
    """Check that a capsule, its grant and the receiver's receipt bind to one another.

    Raises ``ValueError`` on the first failed check.
    """
    if grant.capsule_digest != digest(capsule):
        raise ValueError("grant does not bind this capsule")
    if receipt.grant_digest != digest(grant):
        raise ValueError("receipt does not bind this grant")
    if receipt.capsule_digest != grant.capsule_digest:
        raise ValueError("receipt capsule differs from the granted capsule")
    if grant.anchor_digest != capsule.anchor_digest:
        raise ValueError("grant anchor differs from the capsule anchor")
    if not (capsule.namespace == grant.namespace == receipt.namespace):
        raise ValueError("namespace mismatch across capsule, grant and receipt")
    if grant.lane_id != receipt.lane_id:
        raise ValueError("receipt lane differs from the grant lane")
    if grant.fence != receipt.fence:
        raise ValueError("receipt fence differs from the grant fence")
    if receipt.recipient_model_id != grant.recipient_model_id or receipt.recipient_adapter_id != grant.recipient_adapter_id:
        raise ValueError("receipt recipient differs from the granted recipient")
    if grant.max_slots > capsule.slots:
        raise ValueError("grant allows more slots than the capsule holds")
    if receipt.slots_injected > grant.max_slots:
        raise ValueError("receipt injected more slots than the grant allows")
    if not (grant.issued_at <= receipt.used_at < grant.expires_at):
        raise ValueError("receipt used outside the grant window")


__all__ = [
    "TensorRefV1", "LatentCapsuleV1", "ContextHandoffGrantV1", "LatentUseReceiptV1", "verify_latent_use",
]
