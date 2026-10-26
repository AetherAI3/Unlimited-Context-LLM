"""Grant issue, admission before injection, and receipts that bind under verify_latent_use."""
import hashlib

import numpy as np
import pytest

pytest.importorskip("pydantic")

from aether_context.contracts import NamespaceV1, digest  # noqa: E402
from aether_context.contracts.latent_v1 import LatentCapsuleV1, verify_latent_use  # noqa: E402
from aether_context.latent import handoff  # noqa: E402
from aether_context.latent.tensor_codec import TensorIntegrityError, load_tensor, tensor_bytes  # noqa: E402

NS = NamespaceV1(owner_id="owner", project_id="prj", objective_id="obj", context_cycle_id="cyc-1")
T0 = 1_700_000_000
MODEL, ADAPTER = "tiny/llama@0", "hf-soft-prefix@abc"


def make(slots=4, width=6, **over):
    ref, raw = tensor_bytes(np.ones((slots, width), np.float32), "float16")
    base = dict(capsule_id="cap-1", namespace=NS, source_model_id=MODEL, source_adapter_id=ADAPTER,
                codec_id="hf-soft-prefix", codec_version=1, codebook_digest=digest("proj"), slots=slots,
                tensor=ref, anchor_digest=digest("anchor"), base_revision=digest("rev"), created_at=T0)
    base.update(over)
    return LatentCapsuleV1(**base), raw


def grant_for(cap, **over):
    base = dict(grant_id="g-1", lane_id="lane-l7", sender_principal_id="sender", recipient_principal_id="receiver",
                fence=1, policy_digest=digest("policy"), issued_at=T0, ttl_seconds=300)
    base.update(over)
    return handoff.issue_grant(cap, **base)


def reason(cap, grant, raw, now=T0 + 1, model=MODEL, adapter=ADAPTER, width=6, **extra):
    return handoff.admission_reason(cap, grant, raw, model_id=model, adapter_id=adapter, width=width, now=now,
                                    **extra)


def test_issue_grant_binds_same_model_pair():
    cap, _ = make()
    g = grant_for(cap)
    assert g.capsule_digest == digest(cap) and g.anchor_digest == cap.anchor_digest
    assert (g.recipient_model_id, g.recipient_adapter_id) == (MODEL, ADAPTER)
    assert g.max_slots == cap.slots and g.expires_at == T0 + 300


def test_paired_capsule_admitted_and_receipt_verifies():
    cap, raw = make()
    g = grant_for(cap)
    assert reason(cap, g, raw) is None
    r = handoff.injected_receipt(g, receipt_id="r-1", model_id=MODEL, adapter_id=ADAPTER, positions=[3, 4, 5, 6],
                                 template_boundary_digest=digest("tpl"), mask_digest=digest("mask"), used_at=T0 + 1)
    assert r.outcome == "INJECTED" and r.injection_point == "soft_prefix" and r.slots_injected == 4
    verify_latent_use(cap, g, r)


def test_grant_for_other_capsule_refused():
    cap, raw = make()
    assert reason(cap, grant_for(make(capsule_id="cap-2")[0]), raw) == "CAPSULE_NOT_GRANTED"


def test_anchor_mismatch():
    cap, raw = make()
    g = grant_for(cap).model_copy(update={"anchor_digest": digest("other-anchor")})
    assert reason(cap, g, raw) == "ANCHOR_MISMATCH"


def test_namespace_mismatch():
    cap, raw = make()
    g = grant_for(cap).model_copy(update={"namespace": NS.model_copy(update={"context_cycle_id": "cyc-2"})})
    assert reason(cap, g, raw) == "NAMESPACE_MISMATCH"


def test_recipient_mismatch():
    cap, raw = make()
    assert reason(cap, grant_for(cap), raw, adapter="hf-soft-prefix@other") == "RECIPIENT_MISMATCH"


def test_source_mismatch():
    cap, raw = make(source_model_id="tiny/other@0")
    g = grant_for(cap).model_copy(update={"recipient_model_id": MODEL})
    assert reason(cap, g, raw) == "SOURCE_MISMATCH"


def test_grant_window():
    cap, raw = make()
    g = grant_for(cap)
    assert reason(cap, g, raw, now=T0 - 1) == "GRANT_NOT_YET_VALID"
    assert reason(cap, g, raw, now=T0 + 300) == "GRANT_EXPIRED"


def test_tampered_and_wrong_width():
    cap, raw = make()
    g = grant_for(cap)
    assert reason(cap, g, bytes([raw[0] ^ 1]) + raw[1:]) == "TENSOR_DIGEST_MISMATCH"
    assert reason(cap, g, raw, width=7) == "TENSOR_SHAPE_MISMATCH"


def test_nonfinite_tensor_refused():
    raw = np.full((4, 6), np.inf, np.float16).tobytes()
    cap, _ = make()
    cap = cap.model_copy(update={"tensor": cap.tensor.model_copy(update={"digest": hashlib.sha256(raw).hexdigest()})})
    assert reason(cap, grant_for(cap), raw) == "TENSOR_NONFINITE"


def test_refused_receipt_carries_reason_and_no_injection():
    cap, _ = make()
    g = grant_for(cap)
    r = handoff.refused_receipt(g, receipt_id="r-2", model_id=MODEL, adapter_id=ADAPTER,
                                reason_code="GRANT_EXPIRED", used_at=T0 + 2)
    assert r.outcome == "REFUSED" and r.slots_injected == 0 and r.positions == [] and r.mask_digest is None
    verify_latent_use(cap, g, r)  # binding holds; the outcome says nothing was injected


def test_max_slots_bounds():
    cap, _ = make()
    with pytest.raises(ValueError):
        grant_for(cap, max_slots=5)
    assert grant_for(cap, max_slots=2).max_slots == 2


def test_grant_exceeds_capsule():
    cap, raw = make()
    g = grant_for(cap).model_copy(update={"max_slots": 5})
    assert reason(cap, g, raw) == "GRANT_EXCEEDS_CAPSULE"


def test_forged_slots_claim_refused():
    cap, raw = make()
    forged = cap.model_copy(update={"slots": 8})
    assert reason(forged, grant_for(forged), raw) == "TENSOR_SHAPE_MISMATCH"


def test_forged_ref_shape_refused_without_raising():
    cap, raw = make()
    bad_ref = cap.tensor.model_copy(update={"shape": [5, 6]})  # 5*6*2 bytes != byte_count 48
    forged = cap.model_copy(update={"tensor": bad_ref})
    assert reason(forged, grant_for(forged), raw) == "TENSOR_SHAPE_MISMATCH"
    with pytest.raises(TensorIntegrityError):
        load_tensor(bad_ref, raw)


def test_truncated_bytes_refused_by_admission():
    cap, raw = make()
    assert reason(cap, grant_for(cap), raw[:-1]) == "TENSOR_DIGEST_MISMATCH"


def test_non_float_dtype_refused_before_loading():
    cap, raw = make()
    forged = cap.model_copy(update={"tensor": cap.tensor.model_copy(update={"dtype": "int8"})})
    assert reason(forged, grant_for(forged), raw) == "TENSOR_DTYPE_UNSUPPORTED"


def test_principal_mismatch():
    cap, raw = make()
    g = grant_for(cap)
    assert reason(cap, g, raw, principal_id="receiver") is None
    assert reason(cap, g, raw, principal_id="intruder") == "PRINCIPAL_MISMATCH"


def test_stale_fence():
    cap, raw = make()
    g = grant_for(cap)
    assert reason(cap, g, raw, current_fence=1) is None
    assert reason(cap, g, raw, current_fence=2) == "STALE_FENCE"


def test_reason_codes_listed_in_check_order():
    assert handoff.REASON_CODES == (
        "CAPSULE_NOT_GRANTED", "ANCHOR_MISMATCH", "NAMESPACE_MISMATCH", "RECIPIENT_MISMATCH", "PRINCIPAL_MISMATCH",
        "STALE_FENCE", "SOURCE_MISMATCH", "GRANT_NOT_YET_VALID", "GRANT_EXPIRED", "GRANT_EXCEEDS_CAPSULE",
        "TENSOR_DTYPE_UNSUPPORTED", "TENSOR_DIGEST_MISMATCH", "TENSOR_SHAPE_MISMATCH", "TENSOR_NONFINITE")


def test_refused_receipt_rejects_unknown_reason():
    cap, _ = make()
    with pytest.raises(ValueError, match="reason"):
        handoff.refused_receipt(grant_for(cap), receipt_id="r-3", model_id=MODEL, adapter_id=ADAPTER,
                                reason_code="MADE_UP", used_at=T0 + 2)


def test_injected_receipt_rejects_too_many_positions_and_wrong_recipient():
    cap, _ = make()
    g = grant_for(cap, max_slots=2)
    common = dict(template_boundary_digest=digest("tpl"), mask_digest=digest("mask"), used_at=T0 + 1)
    with pytest.raises(ValueError, match="positions"):
        handoff.injected_receipt(g, receipt_id="r-4", model_id=MODEL, adapter_id=ADAPTER, positions=[1, 2, 3],
                                 **common)
    with pytest.raises(ValueError, match="recipient"):
        handoff.injected_receipt(g, receipt_id="r-5", model_id=MODEL, adapter_id="other", positions=[1, 2], **common)
