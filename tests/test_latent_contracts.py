"""Additive latent-handoff manifests (integers, strings, hashes only)."""
import json
import subprocess
import sys
from pathlib import Path
from typing import get_args

import pytest

pytest.importorskip("pydantic")

from aether_context import contracts  # noqa: E402
from aether_context.contracts import NamespaceV1, canonical, digest  # noqa: E402
from aether_context.contracts import latent_v1  # noqa: E402
from aether_context.contracts.latent_v1 import (  # noqa: E402
    DTYPE_BYTES, SAFE_INT, ContextHandoffGrantV1, Dtype, LatentCapsuleV1, LatentUseReceiptV1, TensorRefV1,
    verify_latent_use,
)

ROOT = Path(__file__).resolve().parents[1]
NS = NamespaceV1(owner_id="owner", project_id="prj", objective_id="obj", context_cycle_id="cyc-1")
T0 = 1_700_000_000


def _flatten(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _flatten(child)
    elif isinstance(value, list):
        for child in value:
            yield from _flatten(child)
    else:
        yield value


def tensor(**over):
    base = dict(digest=digest("tensor"), dtype="float16", shape=[8, 4096], byte_count=8 * 4096 * 2)
    base.update(over)
    return TensorRefV1(**base)


def capsule(**over):
    base = dict(capsule_id="cap-1", namespace=NS, source_model_id="meta-llama/Llama-3.1-8B",
                source_adapter_id="prefix-adapter@3", codec_id="soft-prefix", codec_version=1,
                codebook_digest=None, slots=8, tensor=tensor(), anchor_digest=digest("exact-ir-anchor"),
                base_revision=digest("rev-184"), created_at=T0)
    base.update(over)
    return LatentCapsuleV1(**base)


def grant(cap: LatentCapsuleV1, **over):
    base = dict(grant_id="grant-1", namespace=NS, lane_id="lane-a", capsule_digest=digest(cap),
                anchor_digest=cap.anchor_digest, sender_principal_id="agent-3",
                recipient_principal_id="agent-4", recipient_model_id=cap.source_model_id,
                recipient_adapter_id=cap.source_adapter_id, max_slots=8, fence=1,
                policy_digest=digest("policy"), issued_at=T0, expires_at=T0 + 600)
    base.update(over)
    return ContextHandoffGrantV1(**base)


def receipt(cap: LatentCapsuleV1, g: ContextHandoffGrantV1, **over):
    base = dict(receipt_id="use-1", namespace=NS, lane_id="lane-a", grant_digest=digest(g),
                capsule_digest=digest(cap), recipient_model_id=cap.source_model_id,
                recipient_adapter_id=cap.source_adapter_id, injection_point="soft_prefix",
                slots_injected=8, positions=list(range(8)), template_boundary_digest=digest("template"),
                mask_digest=digest("mask"), outcome="INJECTED", reason_code=None, fence=1, used_at=T0 + 100)
    base.update(over)
    return LatentUseReceiptV1(**base)


def test_capsule_grant_receipt_are_canonical_and_float_free():
    cap = capsule()
    g = grant(cap)
    r = receipt(cap, g)
    for model in (cap, g, r):
        dumped = model.model_dump(mode="json")
        assert json.loads(canonical(model)) == dumped
        assert not any(isinstance(v, float) for v in _flatten(dumped))
        assert len(digest(model)) == 64
    assert cap.lossless is False


def test_tensor_ref_byte_count_must_match_dtype_and_shape():
    with pytest.raises(ValueError):
        tensor(byte_count=1)
    with pytest.raises(ValueError):
        tensor(shape=[0, 4])
    with pytest.raises(ValueError):
        tensor(shape=[-8, -4096])
    with pytest.raises(ValueError):
        tensor(shape=[2**30, 2**30, 16], byte_count=1)
    assert tensor(dtype="int8", shape=[3, 5], byte_count=15).byte_count == 15


def test_dtype_table_matches_literal():
    assert set(get_args(Dtype)) == set(DTYPE_BYTES)


def test_models_are_closed_and_strict():
    with pytest.raises(ValueError):
        capsule(extra_field=1)
    with pytest.raises(ValueError):
        capsule(slots="8")
    with pytest.raises(ValueError):
        capsule(created_at=1.5)
    with pytest.raises(ValueError):
        capsule(slots=4)                                  # tensor leading dim is 8
    with pytest.raises(ValueError):
        capsule(source_model_id="m x")                    # not an Ident


def test_grant_is_bounded():
    cap = capsule()
    with pytest.raises(ValueError):
        grant(cap, expires_at=T0 + 901)
    with pytest.raises(ValueError):
        grant(cap, expires_at=T0)
    with pytest.raises(ValueError):
        grant(cap, max_slots=0)
    with pytest.raises(ValueError):
        grant(cap, sender_principal_id="agent-4")        # sender == recipient


def test_receipt_outcome_rules():
    cap = capsule()
    g = grant(cap)
    with pytest.raises(ValueError):
        receipt(cap, g, outcome="REFUSED")                              # needs reason_code, zero slots
    with pytest.raises(ValueError):
        receipt(cap, g, slots_injected=3)                               # positions length differs
    with pytest.raises(ValueError):
        receipt(cap, g, positions=[0, 0, 1, 2, 3, 4, 5, 6])              # duplicate position
    with pytest.raises(ValueError):
        receipt(cap, g, positions=[1, 0, 2, 3, 4, 5, 6, 7])              # not increasing
    with pytest.raises(ValueError):
        receipt(cap, g, positions=[-1, 1, 2, 3, 4, 5, 6, 7])             # negative position
    with pytest.raises(ValueError):
        receipt(cap, g, slots_injected=1, positions=[SAFE_INT + 1])     # position beyond the safe range
    with pytest.raises(ValueError):
        receipt(cap, g, outcome="NOT_RUN", reason_code="transport")     # still injected 8
    with pytest.raises(ValueError):
        receipt(cap, g, outcome="REFUSED", reason_code="policy")        # still 8 slots
    refused = receipt(cap, g, outcome="REFUSED", reason_code="policy", slots_injected=0, positions=[],
                      template_boundary_digest=None, mask_digest=None)
    assert refused.outcome == "REFUSED"
    with pytest.raises(ValueError):
        receipt(cap, g, mask_digest=None)                               # INJECTED needs both digests
    with pytest.raises(ValueError):
        receipt(cap, g, outcome="NOT_RUN", reason_code="transport", slots_injected=0, positions=[],
                mask_digest=None)                                       # keeps its template digest
    ok = receipt(cap, g, outcome="NOT_RUN", reason_code="transport", slots_injected=0, positions=[],
                 template_boundary_digest=None, mask_digest=None)
    assert ok.outcome == "NOT_RUN"


def test_verify_latent_use_binds_capsule_grant_receipt():
    cap = capsule()
    g = grant(cap)
    verify_latent_use(cap, g, receipt(cap, g))
    verify_latent_use(cap, g, receipt(cap, g, used_at=T0))                       # equal to issued_at passes
    g9 = grant(cap, max_slots=9)
    ga = grant(cap, anchor_digest=digest("other"))
    ns2 = NamespaceV1(owner_id="owner", project_id="prj", objective_id="obj", context_cycle_id="cyc-2")
    g_ns2 = grant(cap, namespace=ns2)                   # grant + receipt agree, capsule is from another namespace
    cases = [
        ((cap, g_ns2, receipt(cap, g_ns2, namespace=ns2)), "namespace mismatch"),
        ((capsule(capsule_id="cap-2"), g, receipt(cap, g)), "grant does not bind this capsule"),
        ((cap, g, receipt(cap, g, grant_digest=digest("old"))), "receipt does not bind this grant"),
        ((cap, g, receipt(capsule(capsule_id="cap-2"), g)), "receipt capsule differs"),
        ((cap, ga, receipt(cap, ga)), "grant anchor differs"),
        ((cap, g, receipt(cap, g, namespace=ns2)), "namespace mismatch"),
        ((cap, g, receipt(cap, g, lane_id="lane-b")), "receipt lane differs"),
        ((cap, g, receipt(cap, g, fence=2)), "receipt fence differs"),
        ((cap, g, receipt(cap, g, recipient_model_id="other-model")), "receipt recipient differs"),
        ((cap, g, receipt(cap, g, recipient_adapter_id="other-adapter")), "receipt recipient differs"),
        ((cap, g9, receipt(cap, g9)), "grant allows more slots"),
        ((cap, g, receipt(cap, g, slots_injected=9, positions=list(range(9)))), "injected more slots"),
        ((cap, g, receipt(cap, g, used_at=T0 + 600)), "outside the grant window"),   # equal to expires_at
        ((cap, g, receipt(cap, g, used_at=T0 - 1)), "outside the grant window"),
    ]
    for args, message in cases:
        with pytest.raises(ValueError, match=message):
            verify_latent_use(*args)
    with pytest.raises(ValueError):
        receipt(cap, g, reason_code="has space", outcome="REFUSED", slots_injected=0, positions=[],
                template_boundary_digest=None, mask_digest=None)                 # reason_code is an Ident
    with pytest.raises(ValueError):
        grant(cap, fence=0)                                                      # fences start at 1
    with pytest.raises(ValueError):
        capsule(codec_version=0)


def test_integers_are_bounded_to_digestable_range():
    cap = capsule()
    with pytest.raises(ValueError):
        capsule(created_at=SAFE_INT + 1)
    with pytest.raises(ValueError):
        grant(cap, fence=SAFE_INT + 1)
    assert len(digest(capsule(created_at=SAFE_INT))) == 64


def test_existing_hosted_contracts_and_schema_export_unchanged():
    names = [
        "NamespaceV1", "ContextRequestV1", "ContextProfileV1", "ContextBindingV1", "CapabilityV1", "SignedV1",
        "PromotionCandidateV1", "ContextCheckpointReceiptV1", "ContextHydrateReceiptV1", "ContextFreezeReceiptV1",
        "AppendRequestV1", "RetrieveRequestV1", "RemoteDeletionReceiptV1", "RetentionCommandV1",
        "ManagedKeyProviderReceiptV1", "KeyDestructionReceiptV1", "RuntimeBuildManifestV1",
        "ExecutorStabilityReceiptV1", "DataPlaneIsolationReceiptV1", "HostedScaleReceiptV1", "ClosureProofV1",
    ]
    exported = json.loads((ROOT / "aether_context/contracts/schema-v1.json").read_text(encoding="utf-8"))
    assert list(exported) == names
    for name in names:
        assert getattr(contracts, name).model_json_schema() == exported[name]
    assert set(latent_v1.__all__).isdisjoint(dir(contracts))


def test_numpy_only_core_never_imports_latent_contracts_or_pydantic():
    code = (
        "import sys\n"
        "sys.modules['pydantic'] = None\n"
        "import aether_context\n"
        "from aether_context import Session\n"
        "assert 'aether_context.contracts' not in sys.modules\n"
        "assert 'aether_context.contracts.latent_v1' not in sys.modules\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
