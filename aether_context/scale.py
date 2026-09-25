"""Measured hosted-profile receipts and deterministic isolation qualification."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import platform
import sqlite3
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    ContextProfileV1,
    DataPlaneIsolationReceiptV1,
    ExecutorStabilityReceiptV1,
    HostedScaleReceiptV1,
    RuntimeBuildManifestV1,
    RuntimeBuildManifestV2,
    RuntimeBuildManifestV3,
    RuntimeEnvironmentV1,
    RuntimeEnvironmentV2,
    canonical,
    digest,
)
from .crypto import ContextFault, verify
from .policy import namespace_matches, visible_values

MAX_HOSTED_PROCESS_BYTES = 536_870_912
MIN_RECALL_BASIS_POINTS = 8000
HOSTED_RUNTIME_DISTRIBUTIONS = (
    "annotated-types",
    "cffi",
    "cryptography",
    "numpy",
    "pydantic",
    "pydantic_core",
    "pycparser",
    "typing-inspection",
    "typing_extensions",
)


def runtime_package_tree_digest(
    root: str | Path | None = None, *, require_source_only: bool = False
) -> str:
    """Hash shipped source/schema bytes and reject executable shadow artifacts.

    A qualified hosted deployment is source-only: bytecode generation is disabled,
    no cache directory exists, and every regular package file is an expected wheel
    payload.  This closes Python's extension/bytecode precedence paths rather than
    claiming a source digest for bytes the interpreter may not execute.
    """

    package_root = Path(root) if root is not None else Path(__file__).resolve().parent
    package_root = package_root.resolve(strict=True)
    if require_source_only and not sys.dont_write_bytecode:
        raise ContextFault("context_runtime_bytecode_enabled")
    entries: list[dict[str, str]] = []
    for path in sorted(package_root.rglob("*")):
        if path.is_symlink():
            raise ContextFault("context_runtime_build_invalid")
        if path.is_dir():
            if require_source_only and path.name == "__pycache__":
                raise ContextFault("context_runtime_import_artifact")
            continue
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".json", ".typed"}:
            if require_source_only:
                raise ContextFault("context_runtime_import_artifact")
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


def runtime_environment_identity(
    root: str | Path | None = None,
    *,
    distributions: tuple[str, ...] = HOSTED_RUNTIME_DISTRIBUTIONS,
) -> RuntimeEnvironmentV2:
    """Measure the interpreter, hosted dependencies, and actual import root."""

    package_root = Path(root) if root is not None else Path(__file__).resolve().parent
    package_root = package_root.resolve(strict=True)
    try:
        executable = Path(sys.executable).resolve(strict=True)
        executable_digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        versions = {
            name: importlib.metadata.version(name)
            for name in distributions
        }
        artifacts = {
            name: _distribution_artifact_digest(name)
            for name in distributions
        }
    except (OSError, importlib.metadata.PackageNotFoundError) as exc:
        raise ContextFault("context_runtime_environment_unavailable") from exc
    abi = sysconfig.get_config_var("SOABI") or getattr(
        sys.implementation, "cache_tag", None
    )
    if not isinstance(abi, str) or not abi:
        raise ContextFault("context_runtime_environment_unavailable")
    return RuntimeEnvironmentV2(
        python_implementation=sys.implementation.name,
        python_version=platform.python_version(),
        python_abi=abi,
        python_executable_digest=executable_digest,
        dependency_import_closure=sorted(distributions),
        dependency_versions=versions,
        dependency_artifact_digests=artifacts,
        sqlite_version=sqlite3.sqlite_version,
        package_import_root_digest=digest(str(package_root.parent)),
    )


def _distribution_artifact_digest(name: str) -> str:
    """Hash and RECORD-verify every installed file owned by one dependency."""

    distribution = importlib.metadata.distribution(name)
    files = distribution.files
    if not files:
        raise ContextFault("context_runtime_environment_unavailable")
    entries: list[dict[str, str]] = []
    for entry in sorted(files, key=lambda item: str(item)):
        path = Path(str(distribution.locate_file(entry)))
        if path.is_symlink() or not path.is_file():
            raise ContextFault("context_runtime_environment_unavailable")
        sha256 = hashlib.sha256()
        recorded = entry.hash
        recorded_hash = hashlib.new(recorded.mode) if recorded is not None else None
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(1_048_576):
                    sha256.update(chunk)
                    if recorded_hash is not None:
                        recorded_hash.update(chunk)
        except OSError as exc:
            raise ContextFault("context_runtime_environment_unavailable") from exc
        if recorded is not None and recorded_hash is not None:
            observed = base64.urlsafe_b64encode(recorded_hash.digest()).rstrip(b"=").decode()
            if observed != recorded.value:
                raise ContextFault("context_runtime_dependency_artifact_mismatch")
        entries.append({"path": str(entry).replace("\\", "/"), "sha256": sha256.hexdigest()})
    return digest(entries)


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
    data_plane_case_generator_digest: str | None
    manifest_digest: str | None
    locked_environment_digest: str | None
    python_implementation: str | None
    python_version: str | None
    python_abi: str | None
    python_executable_digest: str | None
    dependency_versions: dict[str, str]
    dependency_artifact_digests: dict[str, str]
    sqlite_version: str | None
    package_import_root_digest: str | None
    bytecode_policy: str | None


def qualify_runtime_build(
    envelope: dict | None,
    keys: dict,
    *,
    expected_version: str,
    package_tree_digest: str,
    runtime_environment: RuntimeEnvironmentV1 | RuntimeEnvironmentV2 | None = None,
) -> BuildQualification:
    """Verify signed build identity against the bytes of this imported package."""

    if envelope is None or not keys:
        return BuildQualification(
            valid=False,
            failures=("runtime_build_manifest_missing",),
            source_revision=None,
            recall_dataset_digest=None,
            data_plane_case_generator_digest=None,
            manifest_digest=None,
            locked_environment_digest=None,
            python_implementation=None,
            python_version=None,
            python_abi=None,
            python_executable_digest=None,
            dependency_versions={},
            dependency_artifact_digests={},
            sqlite_version=None,
            package_import_root_digest=None,
            bytecode_policy=None,
        )
    try:
        payload = verify(envelope, keys)
        if payload.get("schema_version") == "ContextRuntimeBuildManifestV3":
            manifest: RuntimeBuildManifestV1 | RuntimeBuildManifestV2 | RuntimeBuildManifestV3 = (
                RuntimeBuildManifestV3.model_validate(payload)
            )
        elif payload.get("schema_version") == "ContextRuntimeBuildManifestV2":
            manifest = RuntimeBuildManifestV2.model_validate(payload)
        else:
            manifest = RuntimeBuildManifestV1.model_validate(payload)
        if isinstance(manifest, (RuntimeBuildManifestV2, RuntimeBuildManifestV3)) and (
            payload != manifest.model_dump(mode="json")
        ):
            raise ValueError("runtime build manifest is not exact")
    except (ValueError, TypeError) as exc:
        raise ContextFault("context_runtime_build_manifest_invalid") from exc
    failures: list[str] = []
    if manifest.version != expected_version:
        failures.append("runtime_version_mismatch")
    if manifest.package_tree_digest != package_tree_digest:
        failures.append("runtime_package_tree_mismatch")
    locked_environment_digest: str | None = None
    environment: RuntimeEnvironmentV1 | RuntimeEnvironmentV2 | None = None
    if isinstance(manifest, RuntimeBuildManifestV3):
        environment = manifest.runtime_environment
        locked_environment_digest = manifest.locked_environment_digest
        if runtime_environment is None:
            failures.append("runtime_environment_unavailable")
        elif runtime_environment != environment:
            failures.append("runtime_environment_mismatch")
        elif digest(runtime_environment) != locked_environment_digest:
            failures.append("runtime_environment_digest_mismatch")
    elif isinstance(manifest, RuntimeBuildManifestV2):
        environment = manifest.runtime_environment
        locked_environment_digest = manifest.locked_environment_digest
        failures.append("runtime_environment_contract_legacy")
    else:
        failures.append("runtime_environment_unbound")
    return BuildQualification(
        valid=not failures,
        failures=tuple(failures),
        source_revision=manifest.source_revision,
        recall_dataset_digest=manifest.recall_dataset_digest,
        data_plane_case_generator_digest=manifest.data_plane_case_generator_digest,
        manifest_digest=digest(envelope),
        locked_environment_digest=locked_environment_digest,
        python_implementation=(environment.python_implementation if environment else None),
        python_version=(environment.python_version if environment else None),
        python_abi=(environment.python_abi if environment else None),
        python_executable_digest=(
            environment.python_executable_digest if environment else None
        ),
        dependency_versions=(dict(environment.dependency_versions) if environment else {}),
        dependency_artifact_digests=(
            dict(environment.dependency_artifact_digests) if environment else {}
        ),
        sqlite_version=(environment.sqlite_version if environment else None),
        package_import_root_digest=(
            environment.package_import_root_digest if environment else None
        ),
        bytecode_policy=(environment.bytecode_policy if environment else None),
    )


def qualify_scale_receipt(
    profile: ContextProfileV1,
    envelope: dict | None,
    keys: dict,
    *,
    now: int,
    expected_source_revision: str | None = None,
    expected_recall_dataset_digest: str | None = None,
    expected_data_plane_case_generator_digest: str | None = None,
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
        and receipt.target_concurrency == profile.max_concurrent_cycles
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
        if expected_data_plane_case_generator_digest is None:
            failures.append("data_plane_case_generator_unbound")
        elif (
            isolation.case_generator_digest
            != expected_data_plane_case_generator_digest
        ):
            failures.append("data_plane_case_generator_mismatch")
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
    "runtime_environment_identity",
    "runtime_package_tree_digest",
]
