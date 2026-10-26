"""Handoff authority: issue a grant, admit before injecting, record what happened.

NumPy + pydantic only. Binding is checked here (and again by ``verify_latent_use``);
nothing is injected unless ``admission_reason`` returns ``None``.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from ..contracts import digest
from ..contracts.latent_v1 import ContextHandoffGrantV1, LatentCapsuleV1, LatentUseReceiptV1
from .tensor_codec import TensorIntegrityError, load_tensor


def issue_grant(
    capsule: LatentCapsuleV1, *, grant_id: str, lane_id: str, sender_principal_id: str,
    recipient_principal_id: str, fence: int, policy_digest: str, issued_at: int, ttl_seconds: int = 300,
    max_slots: Optional[int] = None,
) -> ContextHandoffGrantV1:
    """Grant one recipient of the same model and adapter the right to inject ``capsule``."""
    slots = capsule.slots if max_slots is None else max_slots
    if slots > capsule.slots:
        raise ValueError("grant cannot allow more slots than the capsule holds")
    return ContextHandoffGrantV1(
        grant_id=grant_id, namespace=capsule.namespace, lane_id=lane_id, capsule_digest=digest(capsule),
        anchor_digest=capsule.anchor_digest, sender_principal_id=sender_principal_id,
        recipient_principal_id=recipient_principal_id, recipient_model_id=capsule.source_model_id,
        recipient_adapter_id=capsule.source_adapter_id, max_slots=slots, fence=fence,
        policy_digest=policy_digest, issued_at=issued_at, expires_at=issued_at + ttl_seconds,
    )


def admission_reason(
    capsule: LatentCapsuleV1, grant: ContextHandoffGrantV1, tensor: bytes, *, model_id: str, adapter_id: str,
    width: int, now: int,
) -> Optional[str]:
    """Return ``None`` when the receiver may inject, else the first failing reason code."""
    if grant.capsule_digest != digest(capsule):
        return "CAPSULE_NOT_GRANTED"
    if grant.anchor_digest != capsule.anchor_digest:
        return "ANCHOR_MISMATCH"
    if grant.namespace != capsule.namespace:
        return "NAMESPACE_MISMATCH"
    if (grant.recipient_model_id, grant.recipient_adapter_id) != (model_id, adapter_id):
        return "RECIPIENT_MISMATCH"
    if (capsule.source_model_id, capsule.source_adapter_id) != (model_id, adapter_id):
        return "SOURCE_MISMATCH"
    if now < grant.issued_at:
        return "GRANT_NOT_YET_VALID"
    if now >= grant.expires_at:
        return "GRANT_EXPIRED"
    if grant.max_slots > capsule.slots:
        return "GRANT_EXCEEDS_CAPSULE"
    try:
        values = load_tensor(capsule.tensor, tensor)
    except TensorIntegrityError:
        return "TENSOR_DIGEST_MISMATCH"
    if values.ndim != 2 or values.shape[1] != width:
        return "TENSOR_SHAPE_MISMATCH"
    if not np.isfinite(values).all():
        return "TENSOR_NONFINITE"
    return None


def injected_receipt(
    grant: ContextHandoffGrantV1, *, receipt_id: str, model_id: str, adapter_id: str, positions: list[int],
    template_boundary_digest: str, mask_digest: str, used_at: int,
) -> LatentUseReceiptV1:
    return LatentUseReceiptV1(
        receipt_id=receipt_id, namespace=grant.namespace, lane_id=grant.lane_id, grant_digest=digest(grant),
        capsule_digest=grant.capsule_digest, recipient_model_id=model_id, recipient_adapter_id=adapter_id,
        injection_point="soft_prefix", slots_injected=len(positions), positions=list(positions),
        template_boundary_digest=template_boundary_digest, mask_digest=mask_digest, outcome="INJECTED",
        reason_code=None, fence=grant.fence, used_at=used_at,
    )


def refused_receipt(
    grant: ContextHandoffGrantV1, *, receipt_id: str, model_id: str, adapter_id: str, reason_code: str,
    used_at: int, outcome: Literal["REFUSED", "NOT_RUN"] = "REFUSED",
) -> LatentUseReceiptV1:
    return LatentUseReceiptV1(
        receipt_id=receipt_id, namespace=grant.namespace, lane_id=grant.lane_id, grant_digest=digest(grant),
        capsule_digest=grant.capsule_digest, recipient_model_id=model_id, recipient_adapter_id=adapter_id,
        injection_point="soft_prefix", slots_injected=0, positions=[], outcome=outcome,
        reason_code=reason_code, fence=grant.fence, used_at=used_at,
    )


__all__ = ["issue_grant", "admission_reason", "injected_receipt", "refused_receipt"]
