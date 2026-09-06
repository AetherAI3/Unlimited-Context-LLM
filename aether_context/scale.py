"""Measured hosted-profile receipts and deterministic isolation qualification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    ContextProfileV1,
    DataPlaneIsolationReceiptV1,
    ExecutorStabilityReceiptV1,
    HostedScaleReceiptV1,
    RuntimeBuildManifestV1,
    canonical,
    digest,
)
from .crypto import ContextFault, verify
from .policy import namespace_matches, visible_values

MAX_HOSTED_PROCESS_BYTES = 536_870_912
MIN_RECALL_BASIS_POINTS = 8000


def runtime_package_tree_digest(root: str | Path | None = None) -> str:
    """Hash every shipped runtime source/schema byte under the imported package."""

    package_root = Path(root) if root is not None else Path(__file__).resolve().parent
    package_root = package_root.resolve(strict=True)
    entries: list[dict[str, str]] = []
    for path in sorted(package_root.rglob("*")):
        if path.is_symlink():
            raise ContextFault("context_runtime_build_invalid")
        if not path.is_file() or path.suffix not in {".py", ".json", ".typed"}:
            continue
        entries.append(
            {
                "path": path.relative_to(package_root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not entries:
        raise ContextFault("context_runtime_build_invalid")
    return hashlib.sha256(canonical(entries)).hexdigest()


def profile_benchmark_digest(profile: ContextProfileV1) -> str:
    """Digest the declared profile without its receipt pointer to avoid a cycle."""

    return digest(profile.model_dump(mode="json", exclude={"benchmark_receipt"}))


@dataclass(frozen=True)
class ScaleQualification:
    valid: bool
    failures: tuple[str, ...]
    reachable_tokens: int | None
    source_revision: str | None
    indexed_records: int | None
    indexed_bytes: int | None
    resident_peak_bytes: int | None
    receipt_digest: str | None
    recall_dataset_digest: str | None


@dataclass(frozen=True)
class BuildQualification:
    valid: bool
    failures: tuple[str, ...]
    source_revision: str | None
    recall_dataset_digest: str | None
    manifest_digest: str | None


def qualify_runtime_build(
    envelope: dict | None,
    keys: dict,
    *,
    expected_version: str,
    package_tree_digest: str,
) -> BuildQualification:
    """Verify signed build identity against the bytes of this imported package."""

    if envelope is None or not keys:
        return BuildQualification(False, ("runtime_build_manifest_missing",), None, None, None)
    try:
        manifest = RuntimeBuildManifestV1.model_validate(verify(envelope, keys))
    except (ValueError, TypeError) as exc:
        raise ContextFault("context_runtime_build_manifest_invalid") from exc
    failures: list[str] = []
    if manifest.version != expected_version:
        failures.append("runtime_version_mismatch")
    if manifest.package_tree_digest != package_tree_digest:
        failures.append("runtime_package_tree_mismatch")
    return BuildQualification(
        not failures,
        tuple(failures),
        manifest.source_revision,
        manifest.recall_dataset_digest,
        digest(envelope),
    )


def qualify_scale_receipt(
    profile: ContextProfileV1,
    envelope: dict | None,
    keys: dict,
    *,
    now: int,
    expected_source_revision: str | None = None,
    expected_recall_dataset_digest: str | None = None,
    executor_stability_keys: dict | None = None,
    data_plane_isolation_keys: dict | None = None,
) -> ScaleQualification:
    """Validate identity first, then report measured threshold failures honestly."""

    if envelope is None or profile.benchmark_receipt is None or not keys:
        return ScaleQualification(
            False, ("benchmark_receipt_missing",), None, None, None, None, None, None, None
        )
    if digest(envelope) != profile.benchmark_receipt:
        raise ContextFault("context_benchmark_receipt_mismatch")
    try:
        receipt = HostedScaleReceiptV1.model_validate(verify(envelope, keys))
    except (ValueError, TypeError) as exc:
        raise ContextFault("context_benchmark_receipt_invalid") from exc
    exact = (
        receipt.profile_benchmark_digest == profile_benchmark_digest(profile)
        and receipt.index_implementation == profile.index_implementation
        and receipt.index_version == profile.index_version
        and receipt.embedding_version == profile.embedding_version
        and receipt.storage_version == profile.storage_version
        and receipt.configured_max_records == profile.max_records
        and receipt.configured_max_indexed_bytes == profile.max_indexed_bytes
    )
    if not exact:
        raise ContextFault("context_benchmark_profile_mismatch")
    failures: list[str] = []
    if expected_source_revision is None:
        failures.append("source_revision_unbound")
    elif receipt.source_revision != expected_source_revision:
        failures.append("source_revision_mismatch")
    if expected_recall_dataset_digest is None:
        failures.append("recall_dataset_unbound")
    elif receipt.recall_dataset_digest != expected_recall_dataset_digest:
        failures.append("recall_dataset_mismatch")
    if not receipt.issued_at <= now < receipt.expires_at:
        failures.append("benchmark_receipt_expired")
    # A representative run fills at least 90% of both declared capacity axes.
    if receipt.indexed_records * 10 < profile.max_records * 9:
        failures.append("record_capacity_unmeasured")
    if receipt.indexed_bytes * 10 < profile.max_indexed_bytes * 9:
        failures.append("byte_capacity_unmeasured")
    # UTF-8 bytes/4 is the signed profile's conservative reach conversion.
    if receipt.reachable_tokens > receipt.indexed_bytes // 4:
        failures.append("reach_not_conservative")
    if receipt.resident_peak_bytes > MAX_HOSTED_PROCESS_BYTES:
        failures.append("resident_memory_slo_failed")
    if receipt.retrieval_p95_ms > profile.search_slo_ms:
        failures.append("retrieval_slo_failed")
    if receipt.checkpoint_ms > profile.rebuild_slo_seconds * 1000:
        failures.append("checkpoint_slo_failed")
    if receipt.hydrate_ms > profile.rebuild_slo_seconds * 1000:
        failures.append("hydrate_slo_failed")
    if receipt.rebuild_ms > profile.rebuild_slo_seconds * 1000:
        failures.append("rebuild_slo_failed")
    if receipt.recall_basis_points < MIN_RECALL_BASIS_POINTS:
        failures.append("recall_slo_failed")
    if receipt.isolation_cases < 1_000_000 or receipt.unauthorized_records != 0:
        failures.append("namespace_isolation_failed")
    if (
        receipt.isolation_evidence_class != "data_plane"
        or receipt.data_plane_isolation_receipt is None
    ):
        failures.append("namespace_data_plane_unmeasured")
    elif not data_plane_isolation_keys:
        failures.append("namespace_data_plane_keys_missing")
    else:
        try:
            isolation = DataPlaneIsolationReceiptV1.model_validate(
                verify(
                    receipt.data_plane_isolation_receipt.model_dump(mode="json"),
                    data_plane_isolation_keys,
                )
            )
        except (ValueError, TypeError) as exc:
            raise ContextFault("context_data_plane_isolation_receipt_invalid") from exc
        if (
            isolation.source_revision != receipt.source_revision
            or isolation.profile_benchmark_digest != receipt.profile_benchmark_digest
            or isolation.cases != receipt.isolation_cases
            or isolation.unauthorized_records != receipt.unauthorized_records
            or isolation.result_digest != receipt.isolation_result_digest
        ):
            raise ContextFault("context_data_plane_isolation_receipt_mismatch")
        if not isolation.issued_at <= now < isolation.expires_at:
            failures.append("namespace_data_plane_receipt_expired")
    if not receipt.executor_stable or receipt.executor_stability_receipt is None:
        failures.append("executor_stability_failed")
    elif not executor_stability_keys:
        failures.append("executor_stability_keys_missing")
    else:
        try:
            stability = ExecutorStabilityReceiptV1.model_validate(
                verify(receipt.executor_stability_receipt.model_dump(mode="json"), executor_stability_keys)
            )
        except (ValueError, TypeError) as exc:
            raise ContextFault("context_executor_stability_receipt_invalid") from exc
        if (
            stability.source_revision != receipt.source_revision
            or stability.profile_benchmark_digest != receipt.profile_benchmark_digest
            or stability.target_concurrency != receipt.target_concurrency
        ):
            raise ContextFault("context_executor_stability_receipt_mismatch")
        if not stability.issued_at <= now < stability.expires_at:
            failures.append("executor_stability_receipt_expired")
    return ScaleQualification(
        not failures,
        tuple(failures),
        receipt.reachable_tokens if not failures else None,
        receipt.source_revision,
        receipt.indexed_records,
        receipt.indexed_bytes,
        receipt.resident_peak_bytes,
        digest(envelope),
        receipt.recall_dataset_digest,
    )


def _step(value: int) -> int:
    value ^= (value << 13) & 0xFFFFFFFF
    value ^= value >> 17
    value ^= (value << 5) & 0xFFFFFFFF
    return value & 0xFFFFFFFF


def run_namespace_isolation(cases: int = 1_000_000, seed: int = 0xA37E2026) -> dict:
    """Exercise owner/project/objective/cycle and lane denial deterministically.

    This fast property harness shares the exact predicates used by the engine.
    Stored-engine tests remain responsible for targeted signature, SQL, and
    tampering paths.
    """

    if type(cases) is not int or cases < 1_000_000:
        raise ValueError("the production isolation harness requires at least 1,000,000 cases")
    value = seed & 0xFFFFFFFF or 1
    unauthorized = 0
    fields = ("owner_id", "project_id", "objective_id", "context_cycle_id")
    expected = {
        "owner_id": "owner_0",
        "project_id": "project_0",
        "objective_id": "objective_0",
        "context_cycle_id": "cycle_0",
    }
    for index in range(cases):
        value = _step(value)
        field = fields[value & 3]
        presented = expected.copy()
        presented[field] = f"{field}_{index + 1}"
        if namespace_matches(expected, presented):
            unauthorized += 1
        # A worker's private P3 record must also remain invisible to a peer lane.
        if visible_values("worker", "lane_a", "P3", f"lane_b_{value}"):
            unauthorized += 1
    result: dict[str, Any] = {
        "schema_version": "NamespaceIsolationResultV1",
        "algorithm": "xorshift32-exact-namespace/v1",
        "seed": seed,
        "cases": cases,
        "unauthorized_records": unauthorized,
    }
    result["result_digest"] = digest(result)
    return result


__all__ = [
    "BuildQualification",
    "ScaleQualification",
    "profile_benchmark_digest",
    "qualify_runtime_build",
    "qualify_scale_receipt",
    "run_namespace_isolation",
    "runtime_package_tree_digest",
]
