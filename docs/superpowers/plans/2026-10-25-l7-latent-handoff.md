# L7 / W7 learned latent handoff — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test, against strict controls, whether a learned soft-prefix capsule carries usable context between a frozen same-model sender/receiver pair, emitting the merged `latent_v1` manifests (capsule, grant, use receipt) on every handoff.

**Architecture:** A new optional subpackage `aether_context/latent/` (behind a new `[latent]` extra) holds a NumPy-only tensor codec and handoff authority layer (pydantic + numpy, no torch), a torch cross-attention soft-prefix projector, and an HF backend that runs a frozen causal LM as sender (mid-layer hidden states → projector → K prefix vectors) and receiver (prefix spliced into `inputs_embeds` inside the chat template). Only the projector trains; the receiver stays differentiable through `inputs_embeds`. `bench/latent_transfer/` preregisters, trains, encodes (sender process), evaluates (receiver process) and reports paired vs blank / shuffled / mismatched / no-prefix / text-context arms on unseen documents, unseen question phrasings and an unseen yes/no task family, with measured cost including training.

**Tech Stack:** Python 3.13 (CPython, Windows, CPU-only torch 2.12), transformers 4.57, pydantic 2 (existing contracts), numpy. Model: `Qwen/Qwen2.5-0.5B-Instruct` from the official HF hub (≈1 GB), pinned by commit.

**Frozen inputs (never modified):** `aether_context/contracts/__init__.py`, `aether_context/contracts/latent_v1.py`, `schema-v1.json`, every NumPy-only core module (`local_llm.py`, `session.py`, `encoder.py`, …). If a needed field is missing from `latent_v1`, STOP and report BLOCKED; do not widen.

**Interpreters:** torch work runs in a scratch venv with system site packages (`venv-latent`: Python 3.13 + system torch 2.12.1+cpu + transformers 4.57.6). Core tests also run on the system 3.13 interpreter, which has no transformers (the torch tests must skip there).

---

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | add `latent` extra (torch, transformers, pydantic) |
| `.gitignore` | ignore `bench/latent_transfer/runs/` (weights, capsule bytes) |
| `aether_context/latent/__init__.py` | docstring + protocol export; importing it never imports torch |
| `aether_context/latent/protocol.py` | `NativeMemoryLLM` Protocol (encode_capsule / receive) |
| `aether_context/latent/tensor_codec.py` | ndarray ⇄ (`TensorRefV1`, bytes); digest-verified load (numpy + pydantic) |
| `aether_context/latent/handoff.py` | `issue_grant`, `admission_reason`, `injected_receipt`, `refused_receipt` (numpy + pydantic) |
| `aether_context/latent/projector.py` | `SoftPrefixProjector` (torch) |
| `aether_context/latent/hf_soft_prefix.py` | `HFSoftPrefixBackend` (torch + transformers) |
| `aether_context/latent/training.py` | `TrainExample`, `TrainStats`, `train_projector` (projector-only optimisation) |
| `tests/test_latent_tensor_codec.py`, `tests/test_latent_handoff.py`, `tests/test_latent_isolation.py` | run in CI (`.[dev]` has pydantic) |
| `tests/latent_tiny.py`, `tests/test_latent_projector.py`, `tests/test_latent_hf_backend.py`, `tests/test_latent_training.py` | `importorskip("torch")` / `("transformers")`; tiny random Llama + in-memory WordLevel tokenizer, no download |
| `bench/latent_transfer/PREREGISTRATION.md` | committed before any result file |
| `bench/latent_transfer/tasks.py` | deterministic documents, templates, eval items, controls |
| `bench/latent_transfer/analysis.py` | per-doc accuracy, paired bootstrap, verdicts |
| `bench/latent_transfer/cost.py` | FLOPs estimate, device info, peak RSS |
| `bench/latent_transfer/run.py` | CLI phases `prepare / train / encode / evaluate / report / all` (each phase its own process under `all`) |
| `tests/test_latent_bench_tasks.py`, `tests/test_latent_bench_analysis.py`, `tests/test_latent_bench_cost.py` | pure-python bench tests |
| `bench/latent_transfer/results/<ts>.json`, `bench/latent_transfer/REPORT.md` | evidence |
| `docs/LATENT_HANDOFF.md` | short usage + honesty notes |

---

### Task 1: `latent` extra, package skeleton, protocol, isolation test

**Files:** Modify `pyproject.toml`, `.gitignore`. Create `aether_context/latent/__init__.py`, `aether_context/latent/protocol.py`, `tests/test_latent_isolation.py`.

- [ ] **Step 1: failing test** `tests/test_latent_isolation.py`

```python
"""The NumPy-only core never imports torch, transformers or the latent package."""
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORE = ["aether_context", "aether_context.session", "aether_context.local_llm", "aether_context.encoder",
        "aether_context.engine", "aether_context.context_pool"]


def _modules_after(imports: list[str]) -> set[str]:
    code = "import sys\n" + "".join(f"import {m}\n" for m in imports) + "print('\\n'.join(sorted(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def test_core_never_imports_latent_or_torch():
    loaded = _modules_after(CORE)
    assert "torch" not in loaded
    assert "transformers" not in loaded
    assert not any(m == "aether_context.latent" or m.startswith("aether_context.latent.") for m in loaded)


def test_latent_package_import_is_torch_free():
    loaded = _modules_after(["aether_context.latent", "aether_context.latent.protocol"])
    assert "torch" not in loaded and "transformers" not in loaded


def test_latent_extra_declared_and_core_stays_numpy_only():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    latent = " ".join(project["optional-dependencies"]["latent"])
    for dep in ("torch", "transformers", "pydantic"):
        assert dep in latent
    assert project["dependencies"] == ["numpy>=1.24,<3.0"]
```

(`tomllib` is 3.11+; the repo floor is 3.10, so guard with `tomllib = pytest.importorskip("tomllib")` if the 3.10 CI leg would otherwise fail.)

- [ ] **Step 2: run** `python -m pytest tests/test_latent_isolation.py -q` → FAIL (`No module named aether_context.latent`, `KeyError: 'latent'`).

- [ ] **Step 3: implement**

`pyproject.toml` `[project.optional-dependencies]`, after `hosted`:

```toml
latent   = ["torch>=2.2", "transformers>=4.56", "pydantic>=2.10,<3"]   # learned soft-prefix handoff (W7)
```

`.gitignore` append:

```
# latent-transfer bench run artifacts (weights, capsule bytes); only results/ and REPORT.md are evidence
bench/latent_transfer/runs/
```

`aether_context/latent/__init__.py`:

```python
"""Learned latent handoff (W7): soft-prefix capsules between a frozen same-model pair.

Optional. Requires the ``[latent]`` extra (torch + transformers + pydantic). The
NumPy-only core never imports this package, and importing the package itself
does not import torch: ``tensor_codec`` and ``handoff`` need numpy + pydantic
only, while ``projector``, ``hf_soft_prefix`` and ``training`` import torch.

Capsules are lossy (``LatentCapsuleV1.lossless`` is always false). The exact
anchor is the source record whose digest every capsule carries.
"""

from .protocol import NativeMemoryLLM

__all__ = ["NativeMemoryLLM"]
```

`aether_context/latent/protocol.py`:

```python
"""Native-memory protocol: a local model that can emit and consume latent capsules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..contracts import NamespaceV1
    from ..contracts.latent_v1 import ContextHandoffGrantV1, LatentCapsuleV1, LatentUseReceiptV1


@runtime_checkable
class NativeMemoryLLM(Protocol):
    """Sender and receiver halves of a same-model latent handoff.

    ``encode_capsule`` returns the manifest and its tensor bytes (never inlined).
    ``receive`` checks the grant and tensor digest before injecting; it returns the
    generated text (``None`` when refused) and the ``LatentUseReceiptV1``.
    """

    model_id: str

    @property
    def adapter_id(self) -> str: ...

    def encode_capsule(
        self, text: str, *, namespace: "NamespaceV1", capsule_id: str, anchor_digest: str,
        base_revision: str, created_at: int,
    ) -> "tuple[LatentCapsuleV1, bytes]": ...

    def receive(
        self, capsule: "LatentCapsuleV1", grant: "ContextHandoffGrantV1", tensor: bytes, prompt: str, *,
        receipt_id: str, now: int, max_tokens: int = 32,
    ) -> "tuple[Optional[str], LatentUseReceiptV1]": ...
```

- [ ] **Step 4: run** → PASS (3). `python -m pytest tests/test_latent_contracts.py -q` still passes.
- [ ] **Step 5: commit** `feat(latent): optional [latent] extra and native-memory protocol`

---

### Task 2: tensor codec (numpy + pydantic, no torch)

**Files:** Create `aether_context/latent/tensor_codec.py`, `tests/test_latent_tensor_codec.py`.

- [ ] **Step 1: failing tests**

```python
"""Tensor bytes are separate hashed objects; manifests only reference them."""
import hashlib

import numpy as np
import pytest

pytest.importorskip("pydantic")

from aether_context.contracts import canonical  # noqa: E402
from aether_context.latent.tensor_codec import TensorIntegrityError, load_tensor, tensor_bytes  # noqa: E402


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_round_trip(dtype):
    arr = np.arange(24, dtype=np.float32).reshape(4, 6) / 7
    ref, raw = tensor_bytes(arr, dtype)
    assert ref.shape == [4, 6] and ref.dtype == dtype and ref.byte_count == len(raw)
    assert ref.digest == hashlib.sha256(raw).hexdigest()
    back = load_tensor(ref, raw)
    np.testing.assert_array_equal(back, arr.astype(dtype))
    canonical(ref)  # manifest stays float-free


def test_tampered_bytes_refused():
    ref, raw = tensor_bytes(np.ones((2, 3), np.float32), "float16")
    with pytest.raises(TensorIntegrityError, match="digest"):
        load_tensor(ref, bytes([raw[0] ^ 1]) + raw[1:])


def test_truncated_bytes_refused():
    ref, raw = tensor_bytes(np.ones((2, 3), np.float32), "float32")
    with pytest.raises(TensorIntegrityError, match="byte_count"):
        load_tensor(ref, raw[:-1])


def test_nonfinite_rejected():
    with pytest.raises(ValueError, match="finite"):
        tensor_bytes(np.array([[np.nan, 1.0]], np.float32), "float32")


def test_overflow_to_inf_rejected():
    with pytest.raises(ValueError, match="finite"):
        tensor_bytes(np.array([[1e6]], np.float32), "float16")


def test_unsupported_dtype():
    with pytest.raises(ValueError, match="dtype"):
        tensor_bytes(np.ones((1, 1), np.float32), "bfloat16")
```

- [ ] **Step 2: run** → FAIL (module missing).
- [ ] **Step 3: implement** `aether_context/latent/tensor_codec.py`

```python
"""ndarray <-> (TensorRefV1, bytes). Bytes are little-endian, C order, digest = sha256(bytes)."""

from __future__ import annotations

import hashlib

import numpy as np

from ..contracts.latent_v1 import TensorRefV1

# bfloat16 has no NumPy dtype; the contract allows it but this codec does not emit it.
_NP_DTYPES = {"float16": "<f2", "float32": "<f4", "int8": "i1", "uint8": "u1"}


class TensorIntegrityError(ValueError):
    """Tensor bytes do not match their reference."""


def tensor_bytes(array: np.ndarray, dtype: str = "float32") -> tuple[TensorRefV1, bytes]:
    if dtype not in _NP_DTYPES:
        raise ValueError(f"unsupported tensor dtype {dtype!r}")
    with np.errstate(over="ignore"):
        arr = np.ascontiguousarray(np.asarray(array).astype(np.dtype(_NP_DTYPES[dtype])))
    if arr.ndim < 1:
        raise ValueError("tensor must have at least one dimension")
    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
        raise ValueError("tensor values must be finite in the storage dtype")
    raw = arr.tobytes(order="C")
    ref = TensorRefV1(digest=hashlib.sha256(raw).hexdigest(), dtype=dtype,
                      shape=[int(d) for d in arr.shape], byte_count=len(raw))
    return ref, raw


def load_tensor(ref: TensorRefV1, raw: bytes) -> np.ndarray:
    if ref.dtype not in _NP_DTYPES:
        raise TensorIntegrityError(f"unsupported tensor dtype {ref.dtype!r}")
    if len(raw) != ref.byte_count:
        raise TensorIntegrityError("tensor bytes do not match byte_count")
    if hashlib.sha256(raw).hexdigest() != ref.digest:
        raise TensorIntegrityError("tensor bytes do not match digest")
    return np.frombuffer(raw, dtype=np.dtype(_NP_DTYPES[ref.dtype])).reshape(ref.shape).copy()


__all__ = ["TensorIntegrityError", "tensor_bytes", "load_tensor"]
```

- [ ] **Step 4: run** → PASS (7). **Step 5: commit** `feat(latent): digest-verified tensor codec for capsule bytes`

---

### Task 3: handoff authority layer (grant issue, admission, receipts)

**Files:** Create `aether_context/latent/handoff.py`, `tests/test_latent_handoff.py`.

Reason codes (Ident strings), checked in this order, first failure wins: `CAPSULE_NOT_GRANTED`, `ANCHOR_MISMATCH`, `NAMESPACE_MISMATCH`, `RECIPIENT_MISMATCH`, `SOURCE_MISMATCH`, `GRANT_NOT_YET_VALID`, `GRANT_EXPIRED`, `GRANT_EXCEEDS_CAPSULE`, `TENSOR_DIGEST_MISMATCH`, `TENSOR_SHAPE_MISMATCH`, `TENSOR_NONFINITE`.

- [ ] **Step 1: failing tests**

```python
"""Grant issue, admission before injection, and receipts that bind under verify_latent_use."""
import hashlib

import numpy as np
import pytest

pytest.importorskip("pydantic")

from aether_context.contracts import NamespaceV1, digest  # noqa: E402
from aether_context.contracts.latent_v1 import LatentCapsuleV1, verify_latent_use  # noqa: E402
from aether_context.latent import handoff  # noqa: E402
from aether_context.latent.tensor_codec import tensor_bytes  # noqa: E402

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


def reason(cap, grant, raw, now=T0 + 1, model=MODEL, adapter=ADAPTER, width=6):
    return handoff.admission_reason(cap, grant, raw, model_id=model, adapter_id=adapter, width=width, now=now)


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
```

`model_copy(update=...)` deliberately skips validation to build forged objects; if a forged object cannot be built that way, change the test, never the contract.

- [ ] **Step 2: run** → FAIL.
- [ ] **Step 3: implement** `aether_context/latent/handoff.py`

```python
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
```

- [ ] **Step 4: run** → PASS. **Step 5: commit** `feat(latent): grant issue, admission checks and use receipts`

---

### Task 4: soft-prefix projector (torch)

**Files:** Create `aether_context/latent/projector.py`, `tests/test_latent_projector.py`.

- [ ] **Step 1: failing test**

```python
import pytest

torch = pytest.importorskip("torch")

from aether_context.latent.projector import SoftPrefixProjector  # noqa: E402


def test_shape_norm_and_mask():
    torch.manual_seed(0)
    p = SoftPrefixProjector(d_model=16, slots=3, d_attn=8, embed_rms=0.5)
    states = torch.randn(2, 5, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    out = p(states, mask)
    assert out.shape == (2, 3, 16)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 0.5), atol=1e-4)
    states2 = states.clone()
    states2[0, 3:] = 100.0  # padding positions cannot influence the output
    assert torch.allclose(p(states2, mask)[0], out[0], atol=1e-5)


def test_gradients_reach_parameters():
    p = SoftPrefixProjector(d_model=16, slots=2, d_attn=8, embed_rms=1.0)
    p(torch.randn(1, 4, 16), torch.ones(1, 4, dtype=torch.bool)).sum().backward()
    assert all(t.grad is not None for t in p.parameters())


def test_rejects_bad_sizes():
    with pytest.raises(ValueError):
        SoftPrefixProjector(d_model=16, slots=0, d_attn=8, embed_rms=1.0)
```

- [ ] **Step 2: run (venv-latent)** → FAIL.
- [ ] **Step 3: implement**

```python
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
```

- [ ] **Step 4: run** → PASS. **Step 5: commit** `feat(latent): cross-attention soft-prefix projector`

---

### Task 5: HF soft-prefix backend (frozen sender/receiver)

**Files:** Create `aether_context/latent/hf_soft_prefix.py`, `tests/latent_tiny.py`, `tests/test_latent_hf_backend.py`.

Prompt layout: render the chat template with user content `MARK + question` (MARK = U+2063, never tokenized), split at MARK, tokenize each side with `add_special_tokens=False`. Receiver sequence = `before ⧺ [prefix slots | context tokens | nothing] ⧺ after ⧺ [answer tokens when training]`. `positions = range(len(before), len(before)+S)`. `template_boundary_digest = digest({"before": before, "after": after})`; `mask_digest = digest({"mask": [1]*L})` where L is the receiver length without the answer.

- [ ] **Step 1: tiny fixture** `tests/latent_tiny.py`

```python
"""Tiny random Llama + in-memory WordLevel tokenizer with a chat template (no download)."""
from __future__ import annotations

WORDS = ("Record for Ada . Ada lives in Paris Rome Oslo . What city does live ? Answer with one word "
         "Yes No is it true that the pet cat dog").split()
SPECIAL = ["<pad>", "<s>", "</s>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>"]
TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} {% endfor %}"
            "{% if add_generation_prompt %}<|assistant|> {% endif %}")


def tiny_pair(seed: int = 0):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(SPECIAL + sorted(set(WORDS)))}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="<pad>", bos_token="<s>", eos_token="</s>",
                                  unk_token="<unk>", additional_special_tokens=SPECIAL[4:])
    tok.chat_template = TEMPLATE
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=False, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg), tok
```

- [ ] **Step 2: failing tests** `tests/test_latent_hf_backend.py`

```python
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pydantic")

from aether_context.contracts import NamespaceV1, digest  # noqa: E402
from aether_context.contracts.latent_v1 import verify_latent_use  # noqa: E402
from aether_context.latent import NativeMemoryLLM, handoff  # noqa: E402
from aether_context.latent.hf_soft_prefix import HFSoftPrefixBackend  # noqa: E402
from tests.latent_tiny import tiny_pair  # noqa: E402

NS = NamespaceV1(owner_id="o", project_id="p", objective_id="obj", context_cycle_id="c")
T0 = 1_700_000_000
DOC = "Record for Ada . Ada lives in Paris ."
Q = "What city does Ada live ?"


@pytest.fixture()
def backend():
    model, tok = tiny_pair()
    return HFSoftPrefixBackend(model, tok, model_id="tiny/llama@0", slots=4, d_attn=16, system="Answer with one word")


def capsule_for(b, text=DOC, cid="cap-1"):
    return b.encode_capsule(text, namespace=NS, capsule_id=cid, anchor_digest=digest({"text": text}),
                            base_revision=digest("rev"), created_at=T0)


def grant(cap):
    return handoff.issue_grant(cap, grant_id="g", lane_id="l7", sender_principal_id="s", recipient_principal_id="r",
                               fence=1, policy_digest=digest("pol"), issued_at=T0)


def test_base_model_frozen_projector_trainable(backend):
    assert not any(p.requires_grad for p in backend.model.parameters())
    assert all(p.requires_grad for p in backend.projector.parameters())
    assert isinstance(backend, NativeMemoryLLM)


def test_layout_splits_template_once(backend):
    lay = backend.layout(Q)
    ids = backend.tokenizer(Q, add_special_tokens=False).input_ids
    assert lay.before and lay.after[: len(ids)] == ids


def test_encode_capsule_manifest(backend):
    cap, raw = capsule_for(backend)
    assert cap.slots == 4 and cap.tensor.shape == [4, 32] and cap.tensor.dtype == "float16"
    assert cap.source_model_id == "tiny/llama@0" and cap.source_adapter_id == backend.adapter_id
    assert cap.codebook_digest == backend.codebook_digest() and cap.lossless is False
    assert len(raw) == cap.tensor.byte_count


def test_admit_paired_injects_with_verifiable_receipt(backend):
    cap, raw = capsule_for(backend)
    g = grant(cap)
    adm = backend.admit(cap, g, raw, Q, receipt_id="r1", now=T0 + 1)
    assert adm.prefix is not None and tuple(adm.prefix.shape) == (4, 32)
    start = len(backend.layout(Q).before)
    assert adm.receipt.outcome == "INJECTED" and adm.receipt.positions == [start + i for i in range(4)]
    verify_latent_use(cap, g, adm.receipt)


def test_admit_tampered_refused(backend):
    cap, raw = capsule_for(backend)
    adm = backend.admit(cap, grant(cap), bytes([raw[0] ^ 1]) + raw[1:], Q, receipt_id="r1", now=T0 + 1)
    assert adm.prefix is None and adm.receipt.outcome == "REFUSED"
    assert adm.receipt.reason_code == "TENSOR_DIGEST_MISMATCH"


def test_next_token_logprobs_depend_on_prefix(backend):
    a = backend.encode_prefixes([DOC])[0]
    b = backend.encode_prefixes(["Record for Ada . Ada lives in Oslo ."])[0]
    lp = backend.next_token_logprobs([torch.from_numpy(a), torch.from_numpy(b), None], [Q, Q, Q])
    assert lp.shape[0] == 3
    assert torch.allclose(lp.exp().sum(-1), torch.ones(3), atol=1e-4)
    assert not torch.allclose(lp[0], lp[1]) and not torch.allclose(lp[0], lp[2])


def test_batched_scoring_matches_single(backend):
    p = torch.from_numpy(backend.encode_prefixes([DOC])[0])
    single = backend.next_token_logprobs([p], [Q])
    batched = backend.next_token_logprobs([p, None], [Q, "Answer with one word ?"])
    assert torch.allclose(single[0], batched[0], atol=1e-4)


def test_text_context_arm(backend):
    assert backend.next_token_logprobs([None], [Q], contexts=[DOC]).shape[0] == 1


def test_answer_loss_backprops_only_to_projector(backend):
    prefixes = backend.project(backend.sender_states([DOC, DOC]))
    loss, counts = backend.answer_loss(prefixes, [Q, Q], ["Paris", "Paris"])
    loss.backward()
    assert counts["head_tokens"] == 2 and counts["tokens"] > 2
    assert all(p.grad is None for p in backend.model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backend.projector.parameters())


def test_receive_generates_and_refuses_expired(backend):
    cap, raw = capsule_for(backend)
    g = grant(cap)
    text, receipt = backend.receive(cap, g, raw, Q, receipt_id="r1", now=T0 + 1, max_tokens=3)
    assert isinstance(text, str) and receipt.outcome == "INJECTED"
    expired, r2 = backend.receive(cap, g, raw, Q, receipt_id="r2", now=T0 + 10_000, max_tokens=3)
    assert expired is None and r2.reason_code == "GRANT_EXPIRED"


def test_projector_state_round_trip(backend, tmp_path):
    path = tmp_path / "proj.pt"
    backend.save_projector(path)
    model, tok = tiny_pair()
    other = HFSoftPrefixBackend(model, tok, model_id="tiny/llama@0", slots=4, d_attn=16,
                                system="Answer with one word", seed=99)
    assert other.codebook_digest() != backend.codebook_digest()
    other.load_projector(path)
    assert other.codebook_digest() == backend.codebook_digest() and other.adapter_id == backend.adapter_id


def test_flop_params(backend):
    blocks, head = backend.flop_params()
    assert blocks > 0 and head == 32 * backend.model.config.vocab_size
```

- [ ] **Step 3: run (venv-latent)** → FAIL.
- [ ] **Step 4: implement** `aether_context/latent/hf_soft_prefix.py`

```python
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
_MARK = "⁣"  # INVISIBLE SEPARATOR: splits the rendered template, never tokenized


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
        torch.manual_seed(seed)
        self.projector = SoftPrefixProjector(self.width, slots, d_attn, embed_rms=embed_rms)

    @classmethod
    def from_pretrained(cls, model_ref: str, *, revision: Optional[str] = None, **kw) -> "HFSoftPrefixBackend":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_ref, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(model_ref, revision=revision, dtype=torch.float32)
        return cls(model, tok, model_id=f"{model_ref}@{revision}" if revision else model_ref, **kw)

    # -- identity --------------------------------------------------------------------------
    def codebook_digest(self) -> str:
        h = hashlib.sha256()
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
            h.update(name.encode() + b"\0" + t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def flop_params(self) -> tuple[int, int]:
        """(transformer-block params excluding input embeddings, LM-head matmul params)."""
        blocks = sum(p.numel() for p in self._base.parameters()) - self._embed.weight.numel()
        return int(blocks), int(self._head.weight.numel())

    def save_projector(self, path: "str | Path") -> None:
        torch.save(self.projector.state_dict(), Path(path))

    def load_projector(self, path: "str | Path") -> None:
        self.projector.load_state_dict(torch.load(Path(path), weights_only=True))
        self.projector.eval()

    # -- sender ----------------------------------------------------------------------------
    @torch.no_grad()
    def sender_states(self, texts: Sequence[str]) -> list[torch.Tensor]:
        enc = self.tokenizer(list(texts), add_special_tokens=False, padding=True, return_tensors="pt")
        out = self._base(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
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
              receipt_id: str, now: int) -> Admission:
        adapter = self.adapter_id
        code = handoff.admission_reason(capsule, grant, tensor, model_id=self.model_id, adapter_id=adapter,
                                        width=self.width, now=now)
        if code is not None:
            return Admission(None, handoff.refused_receipt(grant, receipt_id=receipt_id, model_id=self.model_id,
                                                           adapter_id=adapter, reason_code=code, used_at=now))
        prefix = torch.from_numpy(load_tensor(capsule.tensor, tensor)[: grant.max_slots].astype(np.float32))
        seq, lay = self._sequence(prefix, question)
        start = len(lay.before)
        receipt = handoff.injected_receipt(
            grant, receipt_id=receipt_id, model_id=self.model_id, adapter_id=adapter,
            positions=list(range(start, start + prefix.shape[0])), template_boundary_digest=lay.boundary_digest(),
            mask_digest=digest({"mask": [1] * int(seq.shape[0])}), used_at=now)
        return Admission(prefix, receipt)

    @torch.no_grad()
    def next_token_logprobs(self, prefixes: Sequence[Optional[torch.Tensor]], questions: Sequence[str],
                            contexts: Optional[Sequence[Optional[str]]] = None) -> torch.Tensor:
        """Log-probabilities of the first answer token, one row per item."""
        ctx = list(contexts) if contexts is not None else [None] * len(questions)
        seqs = [self._sequence(p, q, c)[0] for p, q, c in zip(prefixes, questions, ctx)]
        batch, mask = self._pad(seqs)
        hidden = self._base(inputs_embeds=batch, attention_mask=mask).last_hidden_state
        last = hidden[torch.arange(len(seqs)), mask.sum(-1) - 1]
        return torch.log_softmax(self._head(last).float(), dim=-1)

    def answer_loss(self, prefixes: torch.Tensor, questions: Sequence[str], answers: Sequence[str],
                    ) -> tuple[torch.Tensor, dict]:
        """Mean token cross-entropy of ``answers`` given prefix + question (grad flows to ``prefixes``)."""
        seqs, targets, starts = [], [], []
        for prefix, q, a in zip(prefixes, questions, answers):
            ids = self._ids(a)
            seq, _ = self._sequence(prefix, q, answer=ids)
            seqs.append(seq)
            targets.append(ids)
            starts.append(seq.shape[0] - len(ids))
        batch, mask = self._pad(seqs)
        hidden = self._base(inputs_embeds=batch, attention_mask=mask).last_hidden_state
        rows = [hidden[i, s - 1: s - 1 + len(t)] for i, (s, t) in enumerate(zip(starts, targets))]
        logits = self._head(torch.cat(rows, dim=0)).float()
        gold = torch.tensor([tok for t in targets for tok in t], dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(logits, gold)
        return loss, {"tokens": int(mask.sum()), "head_tokens": int(gold.numel())}

    @torch.no_grad()
    def generate(self, prefix: Optional[torch.Tensor], question: str, *, max_tokens: int = 32) -> str:
        seq, _ = self._sequence(prefix, question)
        out = self.model.generate(inputs_embeds=seq[None], attention_mask=torch.ones(1, seq.shape[0], dtype=torch.long),
                                  max_new_tokens=max_tokens, do_sample=False,
                                  pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def receive(self, capsule: LatentCapsuleV1, grant: ContextHandoffGrantV1, tensor: bytes, prompt: str, *,
                receipt_id: str, now: int, max_tokens: int = 32) -> tuple[Optional[str], LatentUseReceiptV1]:
        adm = self.admit(capsule, grant, tensor, prompt, receipt_id=receipt_id, now=now)
        if adm.prefix is None:
            return None, adm.receipt
        return self.generate(adm.prefix, prompt, max_tokens=max_tokens), adm.receipt


__all__ = ["CODEC_ID", "CODEC_VERSION", "Layout", "Admission", "HFSoftPrefixBackend"]
```

`adapter_id` hashes the projector each call (≈ms on the real model); bench hot loops cache it.

- [ ] **Step 5: run** → PASS. **Step 6: commit** `feat(latent): HF soft-prefix backend for a frozen same-model pair`

---

### Task 6: projector-only training

**Files:** Create `aether_context/latent/training.py`, `tests/test_latent_training.py`.

- [ ] **Step 1: failing test**

```python
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pydantic")

from aether_context.latent.hf_soft_prefix import HFSoftPrefixBackend  # noqa: E402
from aether_context.latent.training import TrainExample, train_projector  # noqa: E402
from tests.latent_tiny import tiny_pair  # noqa: E402


def test_training_moves_only_the_projector_and_lowers_loss():
    model, tok = tiny_pair()
    b = HFSoftPrefixBackend(model, tok, model_id="tiny/llama@0", slots=2, d_attn=16, system="Answer with one word")
    ex = [TrainExample("Record for Ada . Ada lives in Paris .", "What city does Ada live ?", "Paris"),
          TrainExample("Record for Ada . Ada lives in Oslo .", "What city does Ada live ?", "Oslo")]
    book = b.codebook_digest()
    stats = train_projector(b, ex, steps=60, batch_size=2, lr=3e-3, seed=0)
    assert stats.base_digest_before == stats.base_digest_after
    assert b.codebook_digest() != book
    assert stats.steps == 60 and stats.examples_seen == 120 and len(stats.losses) == 60
    assert sum(stats.losses[-10:]) / 10 < sum(stats.losses[:10]) / 10
    assert stats.sender_tokens > 0 and stats.receiver_tokens > 0 and stats.head_tokens == 120
    assert stats.trainable_params == sum(p.numel() for p in b.projector.parameters())
    assert stats.wall_s > 0 and not b.projector.training


def test_rejects_empty_examples():
    model, tok = tiny_pair()
    b = HFSoftPrefixBackend(model, tok, model_id="tiny/llama@0", slots=2, d_attn=16)
    with pytest.raises(ValueError):
        train_projector(b, [], steps=1, batch_size=1, lr=1e-3, seed=0)
```

- [ ] **Step 2: run** → FAIL.
- [ ] **Step 3: implement** `aether_context/latent/training.py`

```python
"""Train only the soft-prefix projector; the base model must be bit-identical afterwards."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from .hf_soft_prefix import HFSoftPrefixBackend

SENDER_BATCH = 16


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
        torch.nn.utils.clip_grad_norm_(params, 1.0)
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
                      base_digest_after=after, losses=losses)


__all__ = ["TrainExample", "TrainStats", "train_projector"]
```

- [ ] **Step 4: run** → PASS. **Step 5: commit** `feat(latent): projector-only training with frozen-base digest check`

---

### Task 7: bench tasks + analysis (pure python, CI-tested)

> Superseded in review: the shipped generator uses 249 train / 234 eval names (eval names unique per
> document, none containing a value word), exactly 2 true and 2 false yes/no claims per document and
> strict paired bootstraps. `bench/latent_transfer/tasks.py` and `PREREGISTRATION.md` are normative.

**Files:** Create `bench/latent_transfer/__init__.py` (one-line docstring), `bench/latent_transfer/tasks.py`, `bench/latent_transfer/analysis.py`, `tests/test_latent_bench_tasks.py`, `tests/test_latent_bench_analysis.py`.

`tasks.py` normative constants (must match PREREGISTRATION.md):

```python
ATTRIBUTES = {
    "city":   ("Paris", "London", "Tokyo", "Cairo", "Berlin", "Madrid", "Rome", "Sydney"),
    "colour": ("Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Black", "White"),
    "pet":    ("Cat", "Dog", "Horse", "Rabbit", "Parrot", "Turtle", "Goat", "Fox"),
    "job":    ("Doctor", "Teacher", "Pilot", "Farmer", "Lawyer", "Chef", "Nurse", "Baker"),
}
FACTS = {  # two sentence variants per attribute; {v} is lower-cased except for cities
    "city":   ("{n} lives in {v}.", "{n}'s home is in {v}."),
    "colour": ("{n}'s favourite colour is {v}.", "{n} likes the colour {v} best."),
    "pet":    ("{n} has a pet {v}.", "{n} keeps a {v} at home."),
    "job":    ("{n} works as a {v}.", "{n} earns a living as a {v}."),
}
TRAIN_QUESTIONS = {
    "city":   ("Where does {n} live?", "Which city is {n}'s home?", "What city does {n} live in?"),
    "colour": ("What is {n}'s favourite colour?", "Which colour does {n} like best?", "Name {n}'s favourite colour."),
    "pet":    ("What pet does {n} have?", "Which animal does {n} keep?", "What kind of pet does {n} own?"),
    "job":    ("What is {n}'s job?", "What does {n} do for work?", "Which profession does {n} have?"),
}
UNSEEN_QUESTIONS = {  # never used in training
    "city":   ("In which city would you find {n}'s house?", "{n} resides in what city?"),
    "colour": ("If {n} picked one colour, which would it be?", "{n} prefers what colour above all others?"),
    "pet":    ("{n} looks after which animal?", "Which creature lives with {n}?"),
    "job":    ("What is {n}'s occupation?", "{n} is employed as what?"),
}
YES_NO = {  # unseen task family, never trained
    "city":   "Is it true that {n} lives in {v}? Answer Yes or No.",
    "colour": "Is {n}'s favourite colour {v}? Answer Yes or No.",
    "pet":    "Does {n} have a pet {v}? Answer Yes or No.",
    "job":    "Does {n} work as a {v}? Answer Yes or No.",
}
```

Behaviour: 48 train names and 24 eval names, disjoint, single capitalised words. `Document(doc_id, name, values, order, variants, text)`; text = `f"Record for {n}. " + " ".join(sentences in order)`. `blank(doc)` → same name, text `f"Record for {n}. No other details are on file."`. `mismatched(doc, rng)` → same name/order/variants, every value a different value of the same attribute. `anchor(doc)` → JSON-safe dict (`doc_id, name, values, order, variants, text`), `anchor_digest(doc) = contracts.digest(anchor(doc))`. `build(seed, n_train, n_eval) -> Dataset(train_docs, eval_docs, blanks, mismatched, shuffle_source, train_examples, eval_items)`; `shuffle_source` maps each eval doc id to another eval doc id (seeded permutation, cyclic shift by one ⇒ derangement). `EvalItem(item_id, doc_id, family ∈ {"unseen_template", "yes_no"}, attribute, question, candidates, claim)`; `gold_under(item, doc)` is the attribute value (or "Yes"/"No" for a claim) as if `doc` were the truth. Yes/no claims are true with probability 0.5, otherwise a different value. Each eval doc yields 4 unseen-template items and 4 yes/no items. Training examples: every train doc × every attribute, question template drawn from `TRAIN_QUESTIONS`. `to_json(dataset)` / `from_json` round trip.

Tests (`pytest.importorskip("pydantic")`): deterministic for one seed, different for another; names disjoint; no training example uses an unseen or yes/no template; derangement has no fixed point and covers every eval doc; mismatched differs in every value and keeps name/order/variants; `gold_under(item, own doc)` matches the doc; anchor digests stable and distinct; yes/no truth rate in [0.35, 0.65] at 200 docs; every eval doc text contains each value; json round trip equal.

`analysis.py`: `per_doc(records, arm, family, key="correct") -> dict[str, float]`; `bootstrap_diff(a, b, *, resamples=2000, seed=0) -> (mean, lo, hi, n)` over shared doc ids, percentile 2.5/97.5, numpy Generator; `verdict(diffs, margin) -> bool` (every lower bound ≥ margin). Tests: identical arms → (0, 0, 0); constant +1 → (1, 1, 1); verdict false when one lower bound < margin; same seed ⇒ same CI.

- [ ] Steps: failing tests → run → implement → run → commit `feat(bench): latent-transfer tasks and paired bootstrap analysis`

---

### Task 8: timing pilot, then PREREGISTRATION.md (committed before any result)

> Done after Task 9 in practice: the run CLI was written and reviewed first, then the preregistration
> was committed alone (4918fc8) before any confirmatory result.

Pilot (train split only; no eval doc is encoded and no eval arm is scored): seconds per training step at the chosen batch and per scoring batch on the real model. Fix sample sizes and steps so the full run fits about 3 h of CPU; record pilot numbers in the prereg. Write `bench/latent_transfer/PREREGISTRATION.md` (question, model + pin, generator + seed, arms, controls, families, metric, sample size, training budget, analysis, H1/H2 promotion rules, integrity probes, cost accounting, deviations policy). Commit alone: `docs(bench): preregister latent-transfer experiment (G-W7)`.

### Task 9: cost helpers + run CLI

**Files:** `bench/latent_transfer/cost.py`, `bench/latent_transfer/run.py`, `tests/test_latent_bench_cost.py`.

`cost.py`: `device_info()` (platform, processor, logical CPUs, torch/transformers versions, torch threads, device `"cpu"`), `peak_rss_bytes()` (psutil if present, else Windows `GetProcessMemoryInfo` via ctypes, else `resource.getrusage`), `forward_flops(blocks, head, tokens, head_tokens) = 2*blocks*tokens + 2*head*head_tokens`, `train_flops(...) = 2 * forward_flops(...)` (activation-gradient backward ≈ forward; frozen weights skip the weight-gradient matmuls; attention-score FLOPs ignored; labelled an estimate). Unit tests for the formulas.

`run.py` phases, each writing into `bench/latent_transfer/runs/<run_id>/`:
- `prepare` — build dataset, write `dataset.json` + digest; check candidate first tokens are distinct within every attribute and for Yes/No under the real tokenizer (abort otherwise).
- `train` — load the pinned model, train, write `projector.pt`, `train.json` (stats, base digest before/after, FLOPs, wall, RSS).
- `encode` — sender process: paired, blank and mismatched capsules for every eval doc; `capsules/<digest>.json` + `<digest>.bin`, `capsule_index.json` (doc_id → arm → digest). Reads document texts only.
- `evaluate` — receiver process: reads manifests/bytes and questions; issues each grant from the manifest just before use (wall clock, TTL 300 s); admits, scores first-token log-probs over the candidates, writes `items.jsonl`, `receipts.jsonl`; every INJECTED receipt passes `verify_latent_use`; integrity probes; text-context arm reads the anchor text after checking its digest.
- `report` — analysis → `results/<UTC ts>.json` + `REPORT.md`.
- `all` — the five phases as separate subprocesses, in order.

### Task 10: full run, report, docs

Run `all` in the background, then the narrative in `REPORT.md`, `docs/LATENT_HANDOFF.md`, CHANGELOG `Unreleased` entry. Commit evidence.

### Task 11: final whole-branch review, push, draft PR

Full suite on system Python 3.13 (core, torch tests skip) and venv-latent (all); whole-branch review; push; `gh pr create --draft`.
