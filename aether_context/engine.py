"""Model-independent context plane. All external calls require signed capabilities."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from .contracts import (
    AppendRequestV1,
    CapabilityV1,
    ClosureProofV1,
    ContextBindingV1,
    ContextBindingCommandV2,
    ContextBindingCommandV3,
    ContextCheckpointDurabilityCommandV1,
    ContextCheckpointDurabilityReceiptV1,
    ContextCheckpointDurabilityReceiptV2,
    ContextCheckpointReceiptV1,
    ContextDeletionReceiptV2,
    ContextFreezeReceiptV1,
    ContextHealthV1,
    ContextHealthV2,
    ContextHealthV3,
    ContextHydrateCommandV2,
    ContextHydrateReceiptV1,
    ContextHydrateReceiptV2,
    ContextProfileV1,
    ContextReservationAbortCommandV1,
    ContextReservationAbortExpiryCommandV1,
    ContextReservationAbortReceiptV1,
    ContextReservationCommandV2,
    ContextReservationCommandV3,
    ContextInvocationNoDispatchReceiptV1,
    ContextReservationExpiryCommandV1,
    ContextReservationExpiryReceiptV1,
    ContextReservationReceiptV1,
    ContextReservationReceiptV2,
    ContextReservationReceiptV3,
    ContextReservationReleaseCommandV1,
    ContextReservationReleaseReceiptV1,
    ManagedKeyProviderReceiptV1,
    PromotionCandidateV1,
    RetrieveRequestV1,
    RuntimeEnvironmentV2,
    SignedV1,
    HOSTED_CONTEXT_CAPABILITIES,
    LIVE_IPC_MAX_CHECKPOINT_BYTES,
    canonical,
    digest,
)
from .crypto import ContextFault, EnvelopeCipher, ReceiptSigner, verify, require_disjoint_keys
from .policy import NOTICE, filter_text, namespace_matches, token_bound, visible, words, writable
from .scale import (
    HOSTED_RUNTIME_DISTRIBUTIONS,
    qualify_runtime_build,
    qualify_scale_receipt,
    runtime_environment_identity,
    runtime_package_tree_digest,
)
from .storage_v2 import SegmentStoreV2

ZERO = "0" * 64
TERMINAL = {"SEALED", "EXPIRED", "ABORTED", "QUARANTINED"}
HOT_ATTESTATION_SECONDS = 60


class ContextEngine:
    def __init__(
        self,
        path: str | Path,
        profile: ContextProfileV1,
        cipher: EnvelopeCipher,
        signer: ReceiptSigner,
        authority_keys: dict,
        proof_keys: dict,
        *,
        clock: Callable[[], float] = time.time,
        cycle_keys: Any = None,
        benchmark_receipt: dict | None = None,
        benchmark_keys: dict | None = None,
        remote_deletion_keys: dict | None = None,
        executor_stability_keys: dict | None = None,
        managed_key_provider_receipt: dict | None = None,
        managed_key_provider_receipts: list[dict] | None = None,
        key_provider_keys: dict | None = None,
        key_destruction_keys: dict | None = None,
        runtime_build_manifest: dict | None = None,
        runtime_build_keys: dict | None = None,
        data_plane_isolation_keys: dict | None = None,
    ):
        if profile.index_version != sqlite3.sqlite_version:
            raise ContextFault("context_index_version_mismatch")
        if (
            managed_key_provider_receipt is not None
            and managed_key_provider_receipts is not None
        ):
            raise ContextFault("context_key_provider_config_invalid")
        if not authority_keys or not proof_keys:
            raise ContextFault("context_verification_keys_required")
        key_groups = [authority_keys, proof_keys, {signer.key_id: signer.key.public_key()}]
        if benchmark_keys:
            key_groups.append(benchmark_keys)
        if remote_deletion_keys:
            key_groups.append(remote_deletion_keys)
        if executor_stability_keys:
            key_groups.append(executor_stability_keys)
        if key_provider_keys:
            key_groups.append(key_provider_keys)
        if key_destruction_keys:
            key_groups.append(key_destruction_keys)
        if runtime_build_keys:
            key_groups.append(runtime_build_keys)
        if data_plane_isolation_keys:
            key_groups.append(data_plane_isolation_keys)
        require_disjoint_keys(*key_groups)
        self.profile, self.cipher, self.signer = profile, cipher, signer
        self.authority_keys, self.proof_keys, self.clock = authority_keys, proof_keys, clock
        self.cycle_keys = cycle_keys
        self.remote_deletion_keys = remote_deletion_keys or {}
        self.key_destruction_keys = key_destruction_keys or {}
        self.managed_key_provider_receipts = tuple(
            managed_key_provider_receipts
            if managed_key_provider_receipts is not None
            else (
                [managed_key_provider_receipt]
                if managed_key_provider_receipt is not None
                else []
            )
        )
        self.key_provider_keys = key_provider_keys or {}
        self._benchmark_receipt = benchmark_receipt
        self._benchmark_keys = benchmark_keys or {}
        self._executor_stability_keys = executor_stability_keys or {}
        self._data_plane_isolation_keys = data_plane_isolation_keys or {}
        self._runtime_build_manifest = runtime_build_manifest
        self._runtime_build_keys = runtime_build_keys or {}
        self.store = SegmentStoreV2(path, profile.resident_cache_bytes)
        from . import __version__

        self._runtime_source_only = bool(
            isinstance(runtime_build_manifest, dict)
            and isinstance(runtime_build_manifest.get("payload"), dict)
            and runtime_build_manifest["payload"].get("schema_version")
            in {"ContextRuntimeBuildManifestV2", "ContextRuntimeBuildManifestV3"}
        )
        manifest_payload = (
            runtime_build_manifest.get("payload", {})
            if isinstance(runtime_build_manifest, dict)
            else {}
        )
        configured_closure = (
            manifest_payload.get("runtime_environment", {}).get(
                "dependency_import_closure"
            )
            if manifest_payload.get("schema_version")
            == "ContextRuntimeBuildManifestV3"
            else None
        )
        self._runtime_distributions = (
            tuple(configured_closure)
            if isinstance(configured_closure, list)
            and all(isinstance(item, str) for item in configured_closure)
            else HOSTED_RUNTIME_DISTRIBUTIONS
        )
        self._runtime_startup_failure: str | None = None
        self._runtime_package_tree_digest: str | None
        runtime_environment = None
        try:
            self._runtime_package_tree_digest = runtime_package_tree_digest(
                require_source_only=self._runtime_source_only
            )
            if self._runtime_source_only:
                runtime_environment = runtime_environment_identity(
                    distributions=self._runtime_distributions
                )
        except (ContextFault, OSError):
            self._runtime_package_tree_digest = None
            self._runtime_startup_failure = "runtime_package_tree_unavailable"
        self._runtime_environment_identity = runtime_environment
        self.build_qualification = qualify_runtime_build(
            self._runtime_build_manifest,
            self._runtime_build_keys,
            expected_version=__version__,
            package_tree_digest=self._runtime_package_tree_digest or ZERO,
            runtime_environment=runtime_environment,
        )
        self.scale_qualification = self._qualify_scale()
        self._attestation_lock = threading.RLock()
        self._hot_attestation_checked_at = -1
        self._hot_attestation_valid_until = -1
        self._hot_attestation_ready = False
        self._hot_reach_claim_enabled = False
        # Startup already measured the package and executable environment. This
        # validates the remaining signed evidence once without repeating either
        # expensive measurement.
        self._health(version=3, measure_environment=False, measure_package=False)
        from .retention import RetentionManager

        self.retention_manager = RetentionManager(self)
        self._audit_startup()

    def _cipher(self, cycle, namespace: dict) -> EnvelopeCipher:
        key_ref = cycle["key_ref"]
        if key_ref is None:
            return self.cipher
        if self.cycle_keys is None:
            raise ContextFault("context_key_provider_required")
        return self.cycle_keys.cipher(key_ref, namespace)

    def _provider_current_key_version(self) -> str | None:
        if self.cycle_keys is None:
            return None
        try:
            value = getattr(self.cycle_keys, "current_key_version", None)
            if callable(value):
                value = value()
            if value is None:
                value = getattr(self.cycle_keys, "key_version", None)
        except Exception:  # Provider availability is reflected as unqualified health.
            return None
        return value if isinstance(value, str) and value else None

    def _provider_key_version(self, key_ref: str) -> str | None:
        if self.cycle_keys is None:
            return None
        try:
            resolver = getattr(self.cycle_keys, "version_for", None)
            value = (
                resolver(key_ref)
                if callable(resolver)
                else self._provider_current_key_version()
            )
        except Exception:  # Provider availability is reflected as unqualified health.
            return None
        return value if isinstance(value, str) and value else None

    def _audit_startup(self):
        # A bounded canary verifies its complete encrypted chain and derived
        # index before serving. Larger profiles need a measured rebuild gate.
        with self.store.transaction() as db:
            cycles = db.execute(
                "SELECT * FROM cycles WHERE state IN ('ACTIVE','DEGRADED','SEALING')"
            ).fetchall()
            for cycle in cycles:
                if cycle["profile"] != digest(self.profile):
                    raise ContextFault("context_profile_mismatch")
                try:
                    binding_payload = json.loads(cycle["binding"])
                    binding = ContextBindingV1.model_validate(binding_payload)
                    if digest(binding_payload) != cycle["binding_digest"]:
                        raise ContextFault("context_integrity_failure")
                    ns = binding.namespace.model_dump()
                    cipher = self._cipher(cycle, ns)
                    self._snapshot(cycle, ns)
                    authority = cipher.decrypt(
                        cycle["authority"], ns, self.profile.max_record_bytes
                    )
                    if digest(authority) != binding.authorization_digest:
                        raise ContextFault("context_integrity_failure")
                    previous, count = ZERO, 0
                    for record in db.execute(
                        "SELECT * FROM segments WHERE cycle=? ORDER BY seq", (cycle["id"],)
                    ):
                        count += 1
                        if (
                            count > self.profile.max_records
                            or record["seq"] != count
                            or record["previous"] != previous
                        ):
                            raise ContextFault("context_chain_corrupt")
                        value = self._record(record, ns, cipher)
                        postings = {
                            row[0]
                            for row in db.execute(
                                "SELECT term FROM postings WHERE cycle=? AND seq=?",
                                (cycle["id"], count),
                            )
                        }
                        if postings != {
                            cipher.token(ns, word) for word in words(value["text"])
                        }:
                            raise ContextFault("context_integrity_failure")
                        previous = record["root"]
                    if count != cycle["cursor"] or previous != cycle["root"]:
                        raise ContextFault("context_chain_corrupt")
                except (ContextFault, ValueError, KeyError, TypeError):
                    db.execute(
                        "UPDATE cycles SET state='QUARANTINED',revision=revision+1 WHERE id=?",
                        (cycle["id"],),
                    )
                    self._event(
                        db, cycle["id"], "context.quarantined", reason="startup_integrity_failure"
                    )

    def now(self) -> int:
        return int(self.clock())

    @contextmanager
    def _transaction(self, capability):
        # The cycle is authenticated before it can be quarantined. Persist the
        # quarantine in a new transaction after rolling back the failed call.
        cap = CapabilityV1.model_validate(verify(capability, self.authority_keys))
        try:
            with self.store.transaction() as db:
                yield db
        except ContextFault as exc:
            if exc.code in {"context_integrity_failure", "context_chain_corrupt"}:
                with self.store.transaction() as db:
                    cycle = cap.namespace.context_cycle_id
                    changed = db.execute(
                        "UPDATE cycles SET state='QUARANTINED',revision=revision+1 WHERE id=? AND state NOT IN ('EXPIRED','QUARANTINED')",
                        (cycle,),
                    ).rowcount
                    if changed:
                        self._event(db, cycle, "context.quarantined", reason=exc.code)
            raise

    def _qualify_scale(self):
        return qualify_scale_receipt(
            self.profile,
            self._benchmark_receipt,
            self._benchmark_keys,
            now=self.now(),
            expected_source_revision=self.build_qualification.source_revision,
            expected_recall_dataset_digest=(
                self.build_qualification.recall_dataset_digest
            ),
            expected_data_plane_case_generator_digest=(
                self.build_qualification.data_plane_case_generator_digest
            ),
            executor_stability_keys=self._executor_stability_keys,
            data_plane_isolation_keys=self._data_plane_isolation_keys,
        )

    def _managed_key_qualification(
        self, *, expected_key_version: str | None = None
    ) -> tuple[bool, tuple[str, ...], dict[str, str]]:
        failures: list[str] = []
        if self.cycle_keys is None:
            failures.append("cycle_key_provider_missing")
        if not self.managed_key_provider_receipts or not self.key_provider_keys:
            failures.append("managed_key_provider_receipt_missing")
            return False, tuple(failures), {}
        required_versions: set[str] = set()
        if expected_key_version is not None:
            required_versions.add(expected_key_version)
        else:
            current_version = self._provider_current_key_version()
            if current_version is None:
                failures.append("managed_key_provider_version_missing")
            else:
                required_versions.add(current_version)
            with self.store.connection() as db:
                stored_keys = db.execute(
                    "SELECT DISTINCT key_ref,key_version FROM cycles WHERE key_ref IS NOT NULL "
                    "AND state!='EXPIRED'"
                ).fetchall()
            if any(row[1] is None for row in stored_keys):
                failures.append("cycle_key_version_missing")
            for key_ref, key_version in stored_keys:
                if key_version is not None:
                    required_versions.add(key_version)
                    if self._provider_key_version(key_ref) != key_version:
                        failures.append("cycle_key_version_unresolvable")
        receipt_digests: dict[str, str] = {}
        seen_versions: set[str] = set()
        for envelope in self.managed_key_provider_receipts:
            try:
                receipt = ManagedKeyProviderReceiptV1.model_validate(
                    verify(envelope, self.key_provider_keys)
                )
            except (ContextFault, ValueError, TypeError):
                failures.append("managed_key_provider_receipt_invalid")
                continue
            if receipt.key_version in seen_versions:
                failures.append("managed_key_provider_receipt_duplicate")
                continue
            seen_versions.add(receipt.key_version)
            if receipt.provider != getattr(self.cycle_keys, "provider", None):
                failures.append("managed_key_provider_mismatch")
                continue
            if receipt.provider == "file-wrapped-dek/v1":
                failures.append("file_key_provider_canary_only")
                continue
            if not receipt.issued_at <= self.now() < receipt.expires_at:
                if receipt.key_version in required_versions:
                    failures.append("managed_key_provider_receipt_expired")
                continue
            receipt_digests[receipt.key_version] = digest(envelope)
        if missing := sorted(required_versions - receipt_digests.keys()):
            failures.extend(f"managed_key_provider_version_unqualified:{item}" for item in missing)
        if not self.remote_deletion_keys:
            failures.append("remote_deletion_keys_missing")
        if not self.key_destruction_keys:
            failures.append("key_destruction_keys_missing")
        return not failures, tuple(failures), receipt_digests

    def health(self) -> dict:
        """Rolling-compatible ContextHealthV1 response."""
        return self._health(version=1, measure_environment=True, measure_package=True)

    def health_v2(self) -> dict:
        """ContextHealthV2 adds the current imported package-tree digest."""
        return self._health(version=2, measure_environment=True, measure_package=True)

    def health_v3(self) -> dict:
        """ContextHealthV3 proves the guarded complete import closure."""
        return self._health(version=3, measure_environment=True, measure_package=True)

    def _health(
        self, *, version: int, measure_environment: bool, measure_package: bool
    ) -> dict:
        # Package bytes plus time-bounded benchmark/provider evidence are
        # rechecked on every health/admission call.
        runtime_package_digest: str | None
        runtime_environment = None
        runtime_hash_failure: str | None = None
        if measure_package:
            try:
                runtime_package_digest = runtime_package_tree_digest(
                    require_source_only=self._runtime_source_only
                )
                if self._runtime_source_only:
                    runtime_environment = (
                        runtime_environment_identity(
                            distributions=self._runtime_distributions
                        )
                        if measure_environment
                        else self._runtime_environment_identity
                    )
                from . import __version__

                self.build_qualification = qualify_runtime_build(
                    self._runtime_build_manifest,
                    self._runtime_build_keys,
                    expected_version=__version__,
                    package_tree_digest=runtime_package_digest,
                    runtime_environment=runtime_environment,
                )
            except (ContextFault, OSError):
                runtime_package_digest = None
                runtime_hash_failure = "runtime_package_tree_unavailable"
        else:
            runtime_package_digest = self._runtime_package_tree_digest
            runtime_environment = self._runtime_environment_identity
            runtime_hash_failure = self._runtime_startup_failure
        self.scale_qualification = self._qualify_scale()
        free = shutil.disk_usage(self.store.path.parent).free
        with self.store.connection() as db:
            db.execute("SELECT count(*) FROM cycles").fetchone()
        benchmark_ready = self.scale_qualification.valid
        managed_keys_ready, managed_key_failures, managed_key_receipt_digests = (
            self._managed_key_qualification()
        )
        runtime_build_failures = list(self.build_qualification.failures)
        if runtime_hash_failure is not None:
            runtime_build_failures.append(runtime_hash_failure)
        runtime_package_stable = (
            runtime_package_digest is not None
            and runtime_package_digest == self._runtime_package_tree_digest
            and self._runtime_startup_failure is None
        )
        runtime_build_ready = (
            runtime_hash_failure is None
            and runtime_package_stable
            and self.build_qualification.valid
        )
        if self._runtime_startup_failure is not None:
            runtime_build_failures.append("runtime_package_tree_startup_unavailable")
        if runtime_package_digest is not None and not runtime_package_stable:
            runtime_build_failures.append("runtime_package_tree_changed")
        profile_ready = self.profile.name != "hosted-v1" or (
            benchmark_ready and managed_keys_ready and runtime_build_ready
        )
        payload = {
            "schema_version": "ContextHealthV1",
            "ready": (
                free >= self.profile.disk_free_floor
                and profile_ready
                and runtime_package_stable
            ),
            "profile_digest": digest(self.profile),
            "index_implementation": self.profile.index_implementation,
            "index_version": sqlite3.sqlite_version,
            "disk_free_bytes": free,
            "resident_cache_limit_bytes": self.profile.resident_cache_bytes,
            "resident_process_limit_bytes": 536_870_912,
            "resident_process_observed_peak_bytes": (
                self.scale_qualification.resident_peak_bytes
            ),
            "configured_max_records": self.profile.max_records,
            "configured_max_indexed_bytes": self.profile.max_indexed_bytes,
            "configured_reachable_tokens_upper_bound": (
                self.profile.max_indexed_bytes // 4
            ),
            "measured_indexed_records": self.scale_qualification.indexed_records,
            "measured_indexed_bytes": self.scale_qualification.indexed_bytes,
            "estimated_reachable_tokens": self.scale_qualification.reachable_tokens,
            "reach_claim_enabled": benchmark_ready and profile_ready,
            "benchmark_failures": list(self.scale_qualification.failures),
            "benchmark_source_revision": self.scale_qualification.source_revision,
            "benchmark_receipt_digest": self.scale_qualification.receipt_digest,
            "benchmark_recall_dataset_digest": (
                self.scale_qualification.recall_dataset_digest
            ),
            "runtime_build_ready": runtime_build_ready,
            "runtime_build_failures": runtime_build_failures,
            "runtime_build_manifest_digest": self.build_qualification.manifest_digest,
            "runtime_source_revision": self.build_qualification.source_revision,
            "runtime_data_plane_case_generator_digest": (
                self.build_qualification.data_plane_case_generator_digest
            ),
            "cycle_key_provider": getattr(self.cycle_keys, "provider", None),
            "managed_cycle_keys_ready": managed_keys_ready,
            "managed_cycle_key_failures": list(managed_key_failures),
            "managed_key_provider_receipt_digests": managed_key_receipt_digests,
            "managed_key_provider_receipt_digest": (
                next(iter(managed_key_receipt_digests.values()))
                if len(managed_key_receipt_digests) == 1
                else None
            ),
        }
        if version >= 2:
            payload["schema_version"] = (
                "ContextHealthV3" if version == 3 else "ContextHealthV2"
            )
            payload["runtime_package_tree_digest"] = runtime_package_digest
            payload["runtime_locked_environment_digest"] = (
                digest(runtime_environment) if runtime_environment is not None else None
            )
            payload["runtime_python_implementation"] = (
                runtime_environment.python_implementation
                if runtime_environment is not None
                else None
            )
            payload["runtime_python_version"] = (
                runtime_environment.python_version if runtime_environment is not None else None
            )
            payload["runtime_python_abi"] = (
                runtime_environment.python_abi if runtime_environment is not None else None
            )
            payload["runtime_python_executable_digest"] = (
                runtime_environment.python_executable_digest
                if runtime_environment is not None
                else None
            )
            payload["runtime_dependency_versions"] = (
                dict(runtime_environment.dependency_versions)
                if runtime_environment is not None
                else {}
            )
            payload["runtime_dependency_artifact_digests"] = (
                dict(runtime_environment.dependency_artifact_digests)
                if runtime_environment is not None
                else {}
            )
            payload["runtime_sqlite_version"] = (
                runtime_environment.sqlite_version if runtime_environment is not None else None
            )
            payload["runtime_package_import_root_digest"] = (
                runtime_environment.package_import_root_digest
                if runtime_environment is not None
                else None
            )
            payload["runtime_bytecode_policy"] = (
                runtime_environment.bytecode_policy
                if version == 3 and runtime_environment is not None
                else "source-only/no-bytecode/v1"
                if runtime_environment is not None
                else None
            )
            if version == 3:
                payload["runtime_dependency_import_closure"] = (
                    list(runtime_environment.dependency_import_closure)
                    if isinstance(runtime_environment, RuntimeEnvironmentV2)
                    else []
                )
                payload["context_protocol_version"] = 3
                payload["capabilities"] = list(HOSTED_CONTEXT_CAPABILITIES)
                payload["live_ipc_checkpoint_bytes"] = min(
                    self.profile.max_checkpoint_bytes,
                    LIVE_IPC_MAX_CHECKPOINT_BYTES,
                )
                result = ContextHealthV3.model_validate(payload).model_dump(mode="json")
            else:
                result = ContextHealthV2.model_validate(payload).model_dump(mode="json")
        else:
            result = ContextHealthV1.model_validate(payload).model_dump(mode="json")
        now = self.now()
        self._hot_attestation_checked_at = now
        self._hot_attestation_valid_until = min(
            now + HOT_ATTESTATION_SECONDS,
            self._evidence_valid_until(now + HOT_ATTESTATION_SECONDS),
        )
        self._hot_attestation_ready = bool(result["ready"])
        self._hot_reach_claim_enabled = bool(result["reach_claim_enabled"])
        return result

    def _evidence_valid_until(self, fallback: int) -> int:
        expiries: list[int] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                if type(value.get("expires_at")) is int:
                    expiries.append(value["expires_at"])
                for nested in value.values():
                    collect(nested)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    collect(nested)

        collect(self._benchmark_receipt)
        collect(self.managed_key_provider_receipts)
        return min(expiries, default=fallback)

    def _space(self, *, full_environment: bool = False) -> None:
        if full_environment:
            health = self._health(
                version=2, measure_environment=True, measure_package=True
            )
            ready = bool(health["ready"])
            free = health["disk_free_bytes"]
        else:
            now = self.now()
            if (
                now >= self._hot_attestation_valid_until
                or now - self._hot_attestation_checked_at >= HOT_ATTESTATION_SECONDS
            ):
                with self._attestation_lock:
                    now = self.now()
                    if (
                        now >= self._hot_attestation_valid_until
                        or now - self._hot_attestation_checked_at
                        >= HOT_ATTESTATION_SECONDS
                    ):
                        self._health(
                            version=3,
                            measure_environment=True,
                            measure_package=True,
                        )
            free = shutil.disk_usage(self.store.path.parent).free
            ready = self._hot_attestation_ready and now < self._hot_attestation_valid_until
        if free < self.profile.disk_free_floor:
            raise ContextFault("context_disk_watermark")
        if not ready:
            raise ContextFault("context_hosted_gate_unready")

    def _cycle(self, db, cycle: str):
        row = db.execute("SELECT * FROM cycles WHERE id=?", (cycle,)).fetchone()
        if row is None:
            raise ContextFault("context_not_found")
        return row

    def _event(self, db, cycle: str, kind: str, **metadata) -> None:
        row = self._cycle(db, cycle)
        number = db.execute(
            "SELECT COALESCE(MAX(cursor),0)+1 FROM events WHERE cycle=?", (cycle,)
        ).fetchone()[0]
        event = {
            "type": kind,
            "context_cycle_id": cycle,
            "project_id": row["project"],
            "event_cursor": number,
            "state_revision": row["revision"],
            "timestamp": self.now(),
            "service": self.signer.key_id,
            **metadata,
        }
        db.execute("INSERT INTO events VALUES(?,?,?)", (cycle, number, canonical(event).decode()))

    def reserve(self, signed_reservation: dict) -> dict:
        """Only Gateway can mint this command, before consuming run authority."""
        return self._reserve(signed_reservation, version=1)

    def reserve_v2(self, signed_reservation: dict) -> dict:
        """Revisioned reservation wire required for explicit release revival."""
        return self._reserve(signed_reservation, version=2)

    def reserve_v3(self, signed_reservation: dict) -> dict:
        """Revive only from an exact service-signed no-dispatch expiry."""
        return self._reserve(signed_reservation, version=3)

    def _reserve(self, signed_reservation: dict, *, version: int) -> dict:
        request = verify(signed_reservation, self.authority_keys)
        base_fields = {
            "operation",
            "owner_id",
            "project_id",
            "idempotency_key",
            "request_digest",
            "profile_digest",
            "expires_at",
        }
        reservation_v3: ContextReservationCommandV3 | None = None
        if version == 3:
            try:
                reservation_v3 = ContextReservationCommandV3.model_validate(request)
            except (TypeError, ValueError) as exc:
                raise ContextFault("context_reservation_schema") from exc
            if request != reservation_v3.model_dump(mode="json"):
                raise ContextFault("context_reservation_schema")
        elif version == 2:
            try:
                reservation_v2 = ContextReservationCommandV2.model_validate(request)
            except (TypeError, ValueError) as exc:
                raise ContextFault("context_reservation_schema") from exc
            if request != reservation_v2.model_dump(mode="json"):
                raise ContextFault("context_reservation_schema")
        elif version != 1 or set(request) != base_fields:
            raise ContextFault("context_reservation_schema")
        if request["operation"] != "reserve" or request["profile_digest"] != digest(self.profile):
            raise ContextFault("context_profile_mismatch")
        reservation_authority_valid = (
            self.now() < request["expires_at"] <= self.now() + 3600
        )
        for field in ("owner_id", "project_id", "idempotency_key", "request_digest"):
            if not isinstance(request[field], str) or not 1 <= len(request[field]) <= 200:
                raise ContextFault("context_reservation_schema")
        if (
            len(request["request_digest"]) != 64
            or any(item not in "0123456789abcdef" for item in request["request_digest"])
        ):
            raise ContextFault("context_reservation_schema")
        released_revision = request.get("released_revision")
        if released_revision is not None and (
            type(released_revision) is not int or released_revision < 2
        ):
            raise ContextFault("context_reservation_schema")
        expiry_envelope: dict | None = None
        expiry_claim: ContextReservationExpiryReceiptV1 | None = None
        if version == 3:
            if reservation_v3 is None:
                raise ContextFault("context_reservation_schema")
            try:
                expiry_envelope = reservation_v3.expired_reservation_receipt.model_dump(
                    mode="json"
                )
                expiry_payload = verify(
                    expiry_envelope,
                    {self.signer.key_id: self.signer.key.public_key()},
                )
                expiry_claim = ContextReservationExpiryReceiptV1.model_validate(
                    expiry_payload
                )
            except (ContextFault, TypeError, ValueError) as exc:
                raise ContextFault("context_reservation_expiry_receipt_invalid") from exc
            if expiry_payload != expiry_claim.model_dump(mode="json"):
                raise ContextFault("context_reservation_expiry_receipt_invalid")
        self._space(full_environment=True)
        identity = {k: request[k] for k in ("owner_id", "project_id", "idempotency_key")}
        cycle = "ctx_" + digest(identity)[:32]
        with self.store.transaction() as db:
            for expired in db.execute(
                "SELECT id FROM cycles WHERE state='RESERVED' AND expires<=?", (self.now(),)
            ).fetchall():
                changed = db.execute(
                    "UPDATE cycles SET state='EXPIRED',revision=revision+1 WHERE id=?",
                    (expired[0],),
                )
                self._event(db, expired[0], "context.reservation_expired")
            existing = db.execute("SELECT * FROM cycles WHERE id=?", (cycle,)).fetchone()
            if existing is not None:
                if (
                    existing["owner"] != request["owner_id"]
                    or existing["project"] != request["project_id"]
                    or existing["request_key"] != request["idempotency_key"]
                    or existing["profile"] != request["profile_digest"]
                    or existing["request_digest"] != request["request_digest"]
                ):
                    raise ContextFault("context_idempotency_conflict")
                if not reservation_authority_valid:
                    if existing["state"] in TERMINAL:
                        raise ContextFault("context_reservation_terminal")
                    raise ContextFault("context_reservation_expired")
                if (
                    existing["state"] == "EXPIRED"
                    and existing["reservation_release_digest"] is not None
                ):
                    if released_revision is None:
                        raise ContextFault("context_reservation_terminal")
                    if released_revision != existing["revision"]:
                        raise ContextFault("context_stale_state")
                    if not self._reservation_never_bound(
                        existing
                    ) or not self._reservation_data_empty(db, cycle):
                        raise ContextFault("context_integrity_failure")
                    self._reservation_release_replay(
                        db, existing, existing["reservation_release_digest"]
                    )
                    count = db.execute(
                        "SELECT count(*) FROM cycles WHERE owner=? AND state NOT IN "
                        "('SEALED','EXPIRED','ABORTED','QUARANTINED')",
                        (request["owner_id"],),
                    ).fetchone()[0]
                    if count >= self.profile.max_concurrent_cycles:
                        raise ContextFault("context_cycle_quota")
                    changed = db.execute(
                        "UPDATE cycles SET state='RESERVED',expires=?,revision=revision+1,"
                        "reservation_release_digest=NULL,reservation_release_receipt=NULL,"
                        "reservation_revival_from_revision=?,reservation_revival_kind='release',"
                        "reservation_revival_command_digest=?,"
                        "reservation_revival_receipt_digest=NULL,"
                        "reservation_revival_no_dispatch_receipt_digest=NULL "
                        "WHERE id=? AND state='EXPIRED' AND revision=?",
                        (
                            request["expires_at"],
                            released_revision,
                            digest(request),
                            cycle,
                            existing["revision"],
                        ),
                    ).rowcount
                    if changed != 1:
                        raise ContextFault("context_stale_state")
                    self._event(db, cycle, "context.reservation_revived")
                    existing = self._cycle(db, cycle)
                elif (
                    existing["state"] == "EXPIRED"
                    and existing["reservation_expiry_digest"] is not None
                ):
                    if version != 3 or expiry_envelope is None or expiry_claim is None:
                        raise ContextFault("context_reservation_terminal")
                    if (
                        existing["reservation_expiry_receipt"]
                        != canonical(expiry_envelope).decode()
                        or expiry_claim.expiry_command_digest
                        != existing["reservation_expiry_digest"]
                        or expiry_claim.context_cycle_id != cycle
                        or expiry_claim.owner_id != request["owner_id"]
                        or expiry_claim.project_id != request["project_id"]
                        or expiry_claim.idempotency_key != request["idempotency_key"]
                        or expiry_claim.request_digest != request["request_digest"]
                        or expiry_claim.profile_digest != request["profile_digest"]
                        or expiry_claim.revision != existing["revision"]
                        or not self._reservation_never_bound(existing)
                        or not self._reservation_data_empty(db, cycle)
                        or existing["reservation_release_digest"] is not None
                        or existing["reservation_abort_digest"] is not None
                    ):
                        raise ContextFault("context_integrity_failure")
                    count = db.execute(
                        "SELECT count(*) FROM cycles WHERE owner=? AND state NOT IN "
                        "('SEALED','EXPIRED','ABORTED','QUARANTINED')",
                        (request["owner_id"],),
                    ).fetchone()[0]
                    if count >= self.profile.max_concurrent_cycles:
                        raise ContextFault("context_cycle_quota")
                    changed = db.execute(
                        "UPDATE cycles SET state='RESERVED',expires=?,revision=revision+1,"
                        "reservation_expiry_digest=NULL,reservation_expiry_receipt=NULL,"
                        "reservation_revival_from_revision=?,reservation_revival_kind='expiry',"
                        "reservation_revival_command_digest=?,"
                        "reservation_revival_receipt_digest=?,"
                        "reservation_revival_no_dispatch_receipt_digest=? "
                        "WHERE id=? AND state='EXPIRED' AND revision=?",
                        (
                            request["expires_at"],
                            expiry_claim.revision,
                            digest(request),
                            digest(expiry_envelope),
                            expiry_claim.invocation_no_dispatch_receipt_digest,
                            cycle,
                            existing["revision"],
                        ),
                    ).rowcount
                    if changed != 1:
                        raise ContextFault("context_stale_state")
                    self._event(db, cycle, "context.reservation_revived")
                    existing = self._cycle(db, cycle)
                elif existing["state"] in TERMINAL:
                    raise ContextFault("context_reservation_terminal")
                elif existing["reservation_revival_from_revision"] is not None:
                    if (
                        version == 1
                        or existing["state"] != "RESERVED"
                        or existing["reservation_revival_command_digest"]
                        != digest(request)
                        or existing["revision"]
                        != existing["reservation_revival_from_revision"] + 1
                    ):
                        raise ContextFault("context_stale_state")
                    if version == 2 and existing["reservation_revival_kind"] != "release":
                        raise ContextFault("context_stale_state")
                    if version == 3 and (
                        expiry_envelope is None
                        or expiry_claim is None
                        or existing["reservation_revival_kind"] != "expiry"
                        or existing["reservation_revival_receipt_digest"]
                        != digest(expiry_envelope)
                        or existing["reservation_revival_no_dispatch_receipt_digest"]
                        != expiry_claim.invocation_no_dispatch_receipt_digest
                    ):
                        raise ContextFault("context_integrity_failure")
                elif released_revision is not None or version == 3:
                    raise ContextFault("context_stale_state")
            else:
                if released_revision is not None or version == 3:
                    raise ContextFault("context_stale_state")
                count = db.execute(
                    "SELECT count(*) FROM cycles WHERE owner=? AND state NOT IN ('SEALED','EXPIRED','ABORTED','QUARANTINED')",
                    (request["owner_id"],),
                ).fetchone()[0]
                if count >= self.profile.max_concurrent_cycles:
                    raise ContextFault("context_cycle_quota")
                db.execute(
                    "INSERT INTO cycles(id,owner,project,request_key,request_digest,profile,state,"
                    "root,expires,checkpoint_durability_mode,checkpoint_durability_legacy,"
                    "reservation_protocol_version) "
                    "VALUES(?,?,?,?,?,?,'RESERVED',?,?,?,?,?)",
                    (
                        cycle,
                        request["owner_id"],
                        request["project_id"],
                        request["idempotency_key"],
                        request["request_digest"],
                        digest(self.profile),
                        ZERO,
                        request["expires_at"],
                        "remote_registered" if version == 1 else "unregistered",
                        1 if version == 1 else 0,
                        version,
                    ),
                )
                self._event(db, cycle, "context.reserved")
                existing = self._cycle(db, cycle)
            receipt_fields = {
                "context_cycle_id": cycle,
                "context_bucket_id": "bucket_" + cycle[4:],
                "request_digest": request["request_digest"],
                "profile_digest": digest(self.profile),
            }
            receipt: (
                ContextReservationReceiptV1
                | ContextReservationReceiptV2
                | ContextReservationReceiptV3
            )
            if version == 3:
                if (
                    existing["reservation_revival_from_revision"] is None
                    or existing["reservation_revival_receipt_digest"] is None
                    or existing["reservation_revival_no_dispatch_receipt_digest"] is None
                    or existing["reservation_revival_command_digest"] is None
                ):
                    raise ContextFault("context_integrity_failure")
                receipt = ContextReservationReceiptV3(
                    **receipt_fields,
                    revision=existing["revision"],
                    prior_expiry_receipt_digest=existing[
                        "reservation_revival_receipt_digest"
                    ],
                    prior_expiry_revision=existing[
                        "reservation_revival_from_revision"
                    ],
                    invocation_no_dispatch_receipt_digest=existing[
                        "reservation_revival_no_dispatch_receipt_digest"
                    ],
                    revival_command_digest=existing[
                        "reservation_revival_command_digest"
                    ],
                )
            elif version == 2:
                receipt = ContextReservationReceiptV2(
                    **receipt_fields, revision=existing["revision"]
                )
            else:
                if not reservation_authority_valid:
                    raise ContextFault("context_reservation_expired")
                receipt = ContextReservationReceiptV1(**receipt_fields)
            return self.signer.sign(receipt.model_dump(mode="json"))

    @staticmethod
    def _reservation_unbound_metadata(row: sqlite3.Row) -> bool:
        return (
            row["binding"] is None
            and row["binding_digest"] is None
            and row["authority"] is None
            and row["snapshot"] is None
            and row["cursor"] == 0
            and row["root"] == ZERO
            and row["bytes"] == 0
            and row["checkpoint"] == 0
            and row["key_ref"] is None
            and row["key_version"] is None
            and row["seal_digest"] is None
            and row["checkpoint_manifest_checksum"] is None
            and row["checkpoint_object_ref_digest"] is None
            and row["checkpoint_durability_mode"] == "unregistered"
            and row["checkpoint_durability_legacy"] == 0
            and row["checkpoint_durability_command_digest"] is None
            and row["checkpoint_durability_receipt"] is None
            and row["hydrate_command_digest"] is None
            and row["hydrate_request_digest"] is None
            and row["hydrate_receipt"] is None
            and row["hold"] == 0
            and row["hold_reason"] is None
            and row["reservation_binding_receipt_digest"] is None
            and row["reservation_binding_revision"] is None
        )

    @classmethod
    def _reservation_never_bound(cls, row: sqlite3.Row) -> bool:
        return (
            cls._reservation_unbound_metadata(row)
            and row["cleanup_state"] == "NONE"
            and row["cleanup_remote_receipt"] is None
            and row["cleanup_operation_key"] is None
            and row["cleanup_request_digest"] is None
            and row["deletion_receipt"] is None
        )

    @staticmethod
    def _reservation_data_empty(
        db: sqlite3.Connection, cycle: str, *, include_retention: bool = True
    ) -> bool:
        tables = ["segments", "postings", "operations", "fences", "call_budgets"]
        if include_retention:
            tables.append("retention_operations")
        return all(
            db.execute(
                f"SELECT 1 FROM {table} WHERE cycle=? LIMIT 1", (cycle,)
            ).fetchone()
            is None
            for table in tables
        )

    def _reservation_release_replay(
        self, db: sqlite3.Connection, row: sqlite3.Row, command_digest: str
    ) -> dict | None:
        stored_digest = row["reservation_release_digest"]
        stored_receipt = row["reservation_release_receipt"]
        if stored_digest is None and stored_receipt is None:
            return None
        if stored_digest is None or stored_receipt is None:
            raise ContextFault("context_integrity_failure")
        if stored_digest != command_digest:
            raise ContextFault("context_reservation_release_conflict")
        try:
            envelope = json.loads(stored_receipt)
            payload = verify(
                envelope,
                {self.signer.key_id: self.signer.key.public_key()},
            )
            receipt = ContextReservationReleaseReceiptV1.model_validate(payload)
        except (ContextFault, TypeError, ValueError) as exc:
            raise ContextFault("context_integrity_failure") from exc
        if (
            payload != receipt.model_dump(mode="json")
            or receipt.context_cycle_id != row["id"]
            or receipt.owner_id != row["owner"]
            or receipt.project_id != row["project"]
            or receipt.idempotency_key != row["request_key"]
            or receipt.request_digest != row["request_digest"]
            or receipt.profile_digest != row["profile"]
            or receipt.release_command_digest != stored_digest
            or receipt.revision != row["revision"]
            or row["state"] != "EXPIRED"
            or row["reservation_protocol_version"] < 2
            or row["reservation_abort_digest"] is not None
            or row["reservation_abort_receipt"] is not None
            or row["reservation_expiry_digest"] is not None
            or row["reservation_expiry_receipt"] is not None
            or row["reservation_revival_from_revision"] is not None
            or row["reservation_revival_kind"] is not None
            or row["reservation_revival_command_digest"] is not None
            or row["reservation_revival_receipt_digest"] is not None
            or row["reservation_revival_no_dispatch_receipt_digest"] is not None
            or not self._reservation_never_bound(row)
            or not self._reservation_data_empty(db, row["id"])
        ):
            raise ContextFault("context_integrity_failure")
        return envelope

    def release_reservation(self, signed_command: dict) -> dict:
        """Release only a Gateway-attested pre-dispatch failure, with CAS replay safety."""

        payload = verify(signed_command, self.authority_keys)
        try:
            command = ContextReservationReleaseCommandV1.model_validate(payload)
        except (TypeError, ValueError) as exc:
            raise ContextFault("context_reservation_release_schema") from exc
        if payload != command.model_dump(mode="json"):
            raise ContextFault("context_reservation_release_schema")
        command_digest = digest(payload)
        with self.store.transaction() as db:
            row = self._cycle(db, command.context_cycle_id)
            replay = self._reservation_release_replay(db, row, command_digest)
            if replay is not None:
                return replay
            if not command.issued_at <= self.now() < command.expires_at:
                raise ContextFault("context_reservation_release_expired")
            if (
                row["owner"] != command.owner_id
                or row["project"] != command.project_id
                or row["request_key"] != command.idempotency_key
                or row["request_digest"] != command.request_digest
                or row["profile"] != command.profile_digest
                or row["reservation_protocol_version"] < 2
                or not self._reservation_never_bound(row)
                or not self._reservation_data_empty(db, row["id"])
                or row["state"] not in {"RESERVED", "EXPIRED"}
                or row["reservation_abort_digest"] is not None
                or row["reservation_abort_receipt"] is not None
                or row["reservation_expiry_digest"] is not None
                or row["reservation_expiry_receipt"] is not None
            ):
                raise ContextFault("context_reservation_release_denied")
            allowed_revisions = {row["revision"]}
            if row["state"] == "EXPIRED":
                # The reservation expiry sweep is the only never-bound transition
                # that may advance the receipt revision before this proof arrives.
                allowed_revisions.add(row["revision"] - 1)
            if command.expected_revision not in allowed_revisions:
                raise ContextFault("context_stale_state")
            revision = row["revision"] + 1
            receipt = self.signer.sign(
                ContextReservationReleaseReceiptV1(
                    context_cycle_id=row["id"],
                    owner_id=row["owner"],
                    project_id=row["project"],
                    idempotency_key=row["request_key"],
                    request_digest=row["request_digest"],
                    profile_digest=row["profile"],
                    authorization_failure_receipt_digest=(
                        command.authorization_failure_receipt_digest
                    ),
                    release_command_digest=command_digest,
                    revision=revision,
                ).model_dump(mode="json")
            )
            changed = db.execute(
                "UPDATE cycles SET state='EXPIRED',expires=?,revision=?,"
                "reservation_release_digest=?,reservation_release_receipt=?,"
                "reservation_revival_from_revision=NULL,reservation_revival_kind=NULL,"
                "reservation_revival_command_digest=NULL,"
                "reservation_revival_receipt_digest=NULL,"
                "reservation_revival_no_dispatch_receipt_digest=NULL "
                "WHERE id=? AND revision=? AND binding IS NULL",
                (
                    self.now(),
                    revision,
                    command_digest,
                    canonical(receipt).decode(),
                    row["id"],
                    row["revision"],
                ),
            ).rowcount
            if changed != 1:
                raise ContextFault("context_stale_state")
            self._event(
                db,
                row["id"],
                "context.reservation_released",
                authorization_failure_receipt_digest=(
                    command.authorization_failure_receipt_digest
                ),
                release_command_digest=command_digest,
            )
            return receipt

    def _reservation_expiry_replay(
        self,
        db: sqlite3.Connection,
        row: sqlite3.Row,
        command_digest: str,
        reservation_receipt_digest: str,
        no_dispatch_receipt_digest: str,
    ) -> dict | None:
        stored_digest = row["reservation_expiry_digest"]
        stored_receipt = row["reservation_expiry_receipt"]
        if stored_digest is None and stored_receipt is None:
            return None
        if stored_digest is None or stored_receipt is None:
            raise ContextFault("context_integrity_failure")
        if stored_digest != command_digest:
            raise ContextFault("context_reservation_expiry_conflict")
        try:
            envelope = json.loads(stored_receipt)
            payload = verify(
                envelope,
                {self.signer.key_id: self.signer.key.public_key()},
            )
            receipt = ContextReservationExpiryReceiptV1.model_validate(payload)
        except (ContextFault, TypeError, ValueError) as exc:
            raise ContextFault("context_integrity_failure") from exc
        if (
            payload != receipt.model_dump(mode="json")
            or receipt.context_cycle_id != row["id"]
            or receipt.context_bucket_id != "bucket_" + row["id"][4:]
            or receipt.objective_id == ""
            or receipt.owner_id != row["owner"]
            or receipt.project_id != row["project"]
            or receipt.idempotency_key != row["request_key"]
            or receipt.request_digest != row["request_digest"]
            or receipt.profile_digest != row["profile"]
            or receipt.reservation_receipt_digest != reservation_receipt_digest
            or receipt.invocation_no_dispatch_receipt_digest != no_dispatch_receipt_digest
            or receipt.expiry_command_digest != stored_digest
            or receipt.revision != row["revision"]
            or receipt.expired_at != row["expires"]
            or row["state"] != "EXPIRED"
            or row["reservation_release_digest"] is not None
            or row["reservation_release_receipt"] is not None
            or row["reservation_abort_digest"] is not None
            or row["reservation_abort_receipt"] is not None
            or row["reservation_revival_from_revision"] is not None
            or row["reservation_revival_kind"] is not None
            or row["reservation_revival_command_digest"] is not None
            or row["reservation_revival_receipt_digest"] is not None
            or row["reservation_revival_no_dispatch_receipt_digest"] is not None
            or not self._reservation_never_bound(row)
            or not self._reservation_data_empty(db, row["id"])
        ):
            raise ContextFault("context_integrity_failure")
        return envelope

    def _validated_reservation_receipt(
        self, envelope: dict
    ) -> ContextReservationReceiptV2 | ContextReservationReceiptV3:
        try:
            payload = verify(
                envelope,
                {self.signer.key_id: self.signer.key.public_key()},
            )
            if payload.get("schema_version") == "ContextReservationReceiptV3":
                claim: ContextReservationReceiptV2 | ContextReservationReceiptV3 = (
                    ContextReservationReceiptV3.model_validate(payload)
                )
            else:
                claim = ContextReservationReceiptV2.model_validate(payload)
        except (ContextFault, TypeError, ValueError) as exc:
            raise ContextFault("context_reservation_receipt_invalid") from exc
        if payload != claim.model_dump(mode="json"):
            raise ContextFault("context_reservation_receipt_invalid")
        return claim

    @staticmethod
    def _reservation_receipt_lineage_matches(
        row: sqlite3.Row,
        claim: ContextReservationReceiptV2 | ContextReservationReceiptV3,
    ) -> bool:
        if isinstance(claim, ContextReservationReceiptV3):
            return (
                row["reservation_revival_kind"] == "expiry"
                and row["reservation_revival_from_revision"]
                == claim.prior_expiry_revision
                and row["reservation_revival_receipt_digest"]
                == claim.prior_expiry_receipt_digest
                and row["reservation_revival_no_dispatch_receipt_digest"]
                == claim.invocation_no_dispatch_receipt_digest
                and row["reservation_revival_command_digest"]
                == claim.revival_command_digest
            )
        return row["reservation_revival_kind"] != "expiry"

    def expire_reservation(self, signed_command: dict) -> dict:
        """Expire a never-dispatched reservation only after ledger closure proof."""

        payload = verify(signed_command, self.authority_keys)
        try:
            command = ContextReservationExpiryCommandV1.model_validate(payload)
        except (TypeError, ValueError) as exc:
            raise ContextFault("context_reservation_expiry_schema") from exc
        if payload != command.model_dump(mode="json"):
            raise ContextFault("context_reservation_expiry_schema")
        reservation_envelope = command.reservation_receipt.model_dump(mode="json")
        try:
            reservation_claim = self._validated_reservation_receipt(
                reservation_envelope
            )
        except ContextFault as exc:
            raise ContextFault("context_reservation_expiry_receipt_invalid") from exc
        try:
            no_dispatch_envelope = command.invocation_no_dispatch_receipt.model_dump(mode="json")
            no_dispatch_payload = verify(no_dispatch_envelope, self.authority_keys)
            no_dispatch_claim = ContextInvocationNoDispatchReceiptV1.model_validate(
                no_dispatch_payload
            )
        except (ContextFault, TypeError, ValueError) as exc:
            raise ContextFault("context_invocation_no_dispatch_receipt_invalid") from exc
        if no_dispatch_payload != no_dispatch_claim.model_dump(mode="json"):
            raise ContextFault("context_invocation_no_dispatch_receipt_invalid")
        command_digest = digest(payload)
        reservation_receipt_digest = digest(reservation_envelope)
        no_dispatch_receipt_digest = digest(no_dispatch_envelope)
        no_dispatch_snapshot = no_dispatch_claim.ledger_snapshot
        with self.store.transaction() as db:
            row = self._cycle(db, command.context_cycle_id)
            replay = self._reservation_expiry_replay(
                db,
                row,
                command_digest,
                reservation_receipt_digest,
                no_dispatch_receipt_digest,
            )
            if replay is not None:
                return replay
            if row["state"] == "RESERVED":
                if not (
                    reservation_claim.revision
                    == command.expected_revision
                    == row["revision"]
                ):
                    raise ContextFault("context_stale_state")
            elif row["state"] == "EXPIRED" and not (
                reservation_claim.revision == row["revision"] - 1
                and command.expected_revision
                in {row["revision"] - 1, row["revision"]}
            ):
                raise ContextFault("context_stale_state")
            now = self.now()
            if (
                not command.issued_at <= now < command.expires_at
                or not no_dispatch_claim.issued_at <= now < no_dispatch_claim.expires_at
            ):
                raise ContextFault("context_reservation_expiry_expired")
            bucket = "bucket_" + row["id"][4:]
            if (
                row["owner"] != command.owner_id
                or row["project"] != command.project_id
                or row["request_key"] != command.idempotency_key
                or row["request_digest"] != command.request_digest
                or row["profile"] != command.profile_digest
                or command.context_bucket_id != bucket
                or command.objective_id != no_dispatch_snapshot.objective_id
                or reservation_claim.context_cycle_id != row["id"]
                or reservation_claim.context_bucket_id != bucket
                or reservation_claim.request_digest != row["request_digest"]
                or reservation_claim.profile_digest != row["profile"]
                or no_dispatch_snapshot.context_cycle_id != row["id"]
                or no_dispatch_snapshot.reservation_revision != reservation_claim.revision
                or no_dispatch_snapshot.reservation_receipt_digest
                != reservation_receipt_digest
                or no_dispatch_snapshot.owner_user_id != row["owner"]
                or no_dispatch_snapshot.idempotency_key != row["request_key"]
                or no_dispatch_snapshot.request_digest
                != "sha256:" + row["request_digest"]
                or command.invocation_no_dispatch_receipt_digest
                != no_dispatch_receipt_digest
                or not self._reservation_receipt_lineage_matches(
                    row, reservation_claim
                )
                or row["state"] not in {"RESERVED", "EXPIRED"}
                or not self._reservation_never_bound(row)
                or not self._reservation_data_empty(db, row["id"])
                or row["reservation_release_digest"] is not None
                or row["reservation_release_receipt"] is not None
                or row["reservation_abort_digest"] is not None
                or row["reservation_abort_receipt"] is not None
            ):
                raise ContextFault("context_reservation_expiry_denied")
            revision = row["revision"] + 1
            receipt = self.signer.sign(
                ContextReservationExpiryReceiptV1(
                    context_cycle_id=row["id"],
                    context_bucket_id=bucket,
                    objective_id=command.objective_id,
                    owner_id=row["owner"],
                    project_id=row["project"],
                    idempotency_key=row["request_key"],
                    request_digest=row["request_digest"],
                    profile_digest=row["profile"],
                    reason_code=command.reason_code,
                    reservation_receipt_digest=reservation_receipt_digest,
                    invocation_no_dispatch_receipt_digest=no_dispatch_receipt_digest,
                    expiry_command_digest=command_digest,
                    expired_at=now,
                    revision=revision,
                ).model_dump(mode="json")
            )
            changed = db.execute(
                "UPDATE cycles SET state='EXPIRED',expires=?,revision=?,"
                "reservation_expiry_digest=?,reservation_expiry_receipt=?,"
                "reservation_revival_from_revision=NULL,reservation_revival_kind=NULL,"
                "reservation_revival_command_digest=NULL,"
                "reservation_revival_receipt_digest=NULL,"
                "reservation_revival_no_dispatch_receipt_digest=NULL "
                "WHERE id=? AND state=? AND revision=? AND binding IS NULL",
                (
                    now,
                    revision,
                    command_digest,
                    canonical(receipt).decode(),
                    row["id"],
                    row["state"],
                    row["revision"],
                ),
            ).rowcount
            if changed != 1:
                raise ContextFault("context_stale_state")
            self._event(
                db,
                row["id"],
                "context.reservation_expired_without_dispatch",
                reservation_receipt_digest=reservation_receipt_digest,
                invocation_no_dispatch_receipt_digest=no_dispatch_receipt_digest,
                expiry_command_digest=command_digest,
            )
            return receipt

    def _reservation_abort_replay(
        self,
        db: sqlite3.Connection,
        row: sqlite3.Row,
        command_digest: str,
        reservation_receipt_digest: str,
    ) -> dict | None:
        stored_digest = row["reservation_abort_digest"]
        stored_receipt = row["reservation_abort_receipt"]
        if stored_digest is None and stored_receipt is None:
            return None
        if stored_digest is None or stored_receipt is None:
            raise ContextFault("context_integrity_failure")
        if stored_digest != command_digest:
            raise ContextFault("context_reservation_abort_conflict")
        try:
            envelope = json.loads(stored_receipt)
            payload = verify(
                envelope,
                {self.signer.key_id: self.signer.key.public_key()},
            )
            receipt = ContextReservationAbortReceiptV1.model_validate(payload)
        except (ContextFault, TypeError, ValueError) as exc:
            raise ContextFault("context_integrity_failure") from exc
        if (
            payload != receipt.model_dump(mode="json")
            or receipt.context_cycle_id != row["id"]
            or receipt.owner_id != row["owner"]
            or receipt.project_id != row["project"]
            or receipt.idempotency_key != row["request_key"]
            or receipt.request_digest != row["request_digest"]
            or receipt.profile_digest != row["profile"]
            or receipt.abort_command_digest != stored_digest
            or receipt.reservation_receipt_digest != reservation_receipt_digest
            or receipt.retention_expires_at != row["expires"]
            or row["reservation_expiry_digest"] is not None
            or row["reservation_expiry_receipt"] is not None
            or row["reservation_revival_from_revision"] is not None
            or row["reservation_revival_kind"] is not None
            or row["reservation_revival_command_digest"] is not None
            or row["reservation_revival_receipt_digest"] is not None
            or row["reservation_revival_no_dispatch_receipt_digest"] is not None
        ):
            raise ContextFault("context_integrity_failure")
        active_abort = (
            row["state"] == "ABORTED"
            and receipt.revision == row["revision"]
            and self._reservation_never_bound(row)
            and self._reservation_data_empty(db, row["id"])
        )
        completed_cleanup = self._reservation_abort_cleanup_valid(db, row, receipt)
        if not active_abort and not completed_cleanup:
            raise ContextFault("context_integrity_failure")
        return envelope

    def _reservation_abort_cleanup_valid(
        self,
        db: sqlite3.Connection,
        row: sqlite3.Row,
        abort_receipt: ContextReservationAbortReceiptV1,
    ) -> bool:
        if (
            row["state"] != "EXPIRED"
            or row["revision"] != abort_receipt.revision + 1
            or row["cleanup_state"] != "COMPLETE"
            or row["cleanup_remote_receipt"] is not None
            or row["cleanup_operation_key"] is None
            or row["cleanup_request_digest"] is None
            or row["deletion_receipt"] is None
            or not self._reservation_unbound_metadata(row)
            or not self._reservation_data_empty(
                db, row["id"], include_retention=False
            )
        ):
            return False
        try:
            deletion_envelope = json.loads(row["deletion_receipt"])
            deletion_payload = verify(
                deletion_envelope,
                {self.signer.key_id: self.signer.key.public_key()},
            )
            deletion = ContextDeletionReceiptV2.model_validate(deletion_payload)
        except (ContextFault, TypeError, ValueError):
            return False
        if (
            deletion_payload != deletion.model_dump(mode="json")
            or deletion.context_cycle_id != row["id"]
            or deletion.segment_chain_root != ZERO
            or deletion.context_seal_digest is not None
            or deletion.retention_class != row["retention_class"]
            or deletion.retention_expired_at != abort_receipt.retention_expires_at
            or deletion.retention_operation_key != row["cleanup_operation_key"]
            or deletion.retention_request_digest != row["cleanup_request_digest"]
            or deletion.cryptographic_erasure
            or deletion.checkpoint_durability_mode != "unregistered"
        ):
            return False
        operation = db.execute(
            "SELECT request_digest,result FROM retention_operations "
            "WHERE cycle=? AND operation='expire_aborted_reservation' AND key=?",
            (row["id"], row["cleanup_operation_key"]),
        ).fetchone()
        count = db.execute(
            "SELECT count(*) FROM retention_operations WHERE cycle=?",
            (row["id"],),
        ).fetchone()[0]
        return (
            operation is not None
            and count == 1
            and operation["request_digest"] == row["cleanup_request_digest"]
            and operation["result"] == canonical(deletion_envelope).decode()
        )

    def abort_reservation(self, signed_command: dict) -> dict:
        """Abort an exact dispatched objective that never reached context bind."""

        payload = verify(signed_command, self.authority_keys)
        try:
            command = ContextReservationAbortCommandV1.model_validate(payload)
        except (TypeError, ValueError) as exc:
            raise ContextFault("context_reservation_abort_schema") from exc
        if payload != command.model_dump(mode="json"):
            raise ContextFault("context_reservation_abort_schema")
        reservation_envelope = command.reservation_receipt.model_dump(mode="json")
        try:
            reservation_claim = self._validated_reservation_receipt(
                reservation_envelope
            )
        except ContextFault as exc:
            raise ContextFault("context_reservation_abort_receipt_invalid") from exc
        command_digest = digest(payload)
        with self.store.transaction() as db:
            row = self._cycle(db, command.context_cycle_id)
            replay = self._reservation_abort_replay(
                db, row, command_digest, digest(reservation_envelope)
            )
            if replay is not None:
                return replay
            if row["state"] == "RESERVED":
                if not (
                    reservation_claim.revision
                    == command.expected_revision
                    == row["revision"]
                ):
                    raise ContextFault("context_stale_state")
            elif row["state"] == "EXPIRED" and not (
                reservation_claim.revision == row["revision"] - 1
                and command.expected_revision
                in {row["revision"] - 1, row["revision"]}
            ):
                raise ContextFault("context_stale_state")
            now = self.now()
            if not command.issued_at <= now < command.expires_at:
                raise ContextFault("context_reservation_abort_expired")
            if (
                row["owner"] != command.owner_id
                or row["project"] != command.project_id
                or row["request_key"] != command.idempotency_key
                or row["request_digest"] != command.request_digest
                or row["profile"] != command.profile_digest
                or reservation_claim.context_cycle_id != row["id"]
                or reservation_claim.context_bucket_id
                != "bucket_" + row["id"][4:]
                or reservation_claim.request_digest != row["request_digest"]
                or reservation_claim.profile_digest != row["profile"]
                or not self._reservation_receipt_lineage_matches(
                    row, reservation_claim
                )
                or row["state"] not in {"RESERVED", "EXPIRED"}
                or not self._reservation_never_bound(row)
                or not self._reservation_data_empty(db, row["id"])
                or row["reservation_release_digest"] is not None
                or row["reservation_release_receipt"] is not None
                or row["reservation_expiry_digest"] is not None
                or row["reservation_expiry_receipt"] is not None
            ):
                raise ContextFault("context_reservation_abort_denied")
            revision = row["revision"] + 1
            retention_expires_at = now + self.profile.retention_seconds
            receipt = self.signer.sign(
                ContextReservationAbortReceiptV1(
                    context_cycle_id=row["id"],
                    owner_id=row["owner"],
                    project_id=row["project"],
                    objective_id=command.objective_id,
                    idempotency_key=row["request_key"],
                    request_digest=row["request_digest"],
                    profile_digest=row["profile"],
                    reason_code=command.reason_code,
                    objective_dispatch_receipt_digest=(
                        command.objective_dispatch_receipt_digest
                    ),
                    reservation_receipt_digest=digest(reservation_envelope),
                    abort_command_digest=command_digest,
                    aborted_at=now,
                    retention_expires_at=retention_expires_at,
                    revision=revision,
                ).model_dump(mode="json")
            )
            changed = db.execute(
                "UPDATE cycles SET state='ABORTED',expires=?,revision=?,"
                "reservation_abort_digest=?,reservation_abort_receipt=?,"
                "reservation_revival_from_revision=NULL,reservation_revival_kind=NULL,"
                "reservation_revival_command_digest=NULL,"
                "reservation_revival_receipt_digest=NULL,"
                "reservation_revival_no_dispatch_receipt_digest=NULL "
                "WHERE id=? AND state=? AND revision=? AND binding IS NULL "
                "AND binding_digest IS NULL AND authority IS NULL AND snapshot IS NULL "
                "AND cursor=0 AND root=? AND bytes=0 AND checkpoint=0 "
                "AND key_ref IS NULL AND key_version IS NULL AND seal_digest IS NULL "
                "AND checkpoint_manifest_checksum IS NULL",
                (
                    retention_expires_at,
                    revision,
                    command_digest,
                    canonical(receipt).decode(),
                    row["id"],
                    row["state"],
                    row["revision"],
                    ZERO,
                ),
            ).rowcount
            if changed != 1:
                raise ContextFault("context_stale_state")
            self._event(
                db,
                row["id"],
                "context.reservation_aborted",
                objective_id=command.objective_id,
                reason_code=command.reason_code,
                objective_dispatch_receipt_digest=(
                    command.objective_dispatch_receipt_digest
                ),
                reservation_receipt_digest=digest(reservation_envelope),
                abort_command_digest=command_digest,
            )
            return receipt

    def expire_aborted_reservation(self, signed_command: dict) -> dict:
        """Clean an empty unbound abort after its retention deadline."""

        payload = verify(signed_command, self.authority_keys)
        try:
            command = ContextReservationAbortExpiryCommandV1.model_validate(payload)
        except (TypeError, ValueError) as exc:
            raise ContextFault("context_reservation_abort_expiry_schema") from exc
        if payload != command.model_dump(mode="json"):
            raise ContextFault("context_reservation_abort_expiry_schema")
        try:
            abort_envelope = command.abort_receipt.model_dump(mode="json")
            abort_payload = verify(
                abort_envelope,
                {self.signer.key_id: self.signer.key.public_key()},
            )
            abort_receipt = ContextReservationAbortReceiptV1.model_validate(
                abort_payload
            )
        except (ContextFault, TypeError, ValueError) as exc:
            raise ContextFault("context_reservation_abort_receipt_invalid") from exc
        if abort_payload != abort_receipt.model_dump(mode="json"):
            raise ContextFault("context_reservation_abort_receipt_invalid")
        request_digest = digest(payload)
        with self.store.transaction() as db:
            row = self._cycle(db, command.namespace.context_cycle_id)
            replay = db.execute(
                "SELECT request_digest,result FROM retention_operations "
                "WHERE cycle=? AND operation='expire_aborted_reservation' AND key=?",
                (row["id"], command.idempotency_key),
            ).fetchone()
            if replay is not None:
                if replay["request_digest"] != request_digest:
                    raise ContextFault("context_idempotency_conflict")
                if replay["result"] is None:
                    raise ContextFault("context_integrity_failure")
                try:
                    result = json.loads(replay["result"])
                    result_payload = verify(
                        result,
                        {self.signer.key_id: self.signer.key.public_key()},
                    )
                    result_claim = ContextDeletionReceiptV2.model_validate(
                        result_payload
                    )
                except (ContextFault, TypeError, ValueError) as exc:
                    raise ContextFault("context_integrity_failure") from exc
                if (
                    result_payload != result_claim.model_dump(mode="json")
                    or row["deletion_receipt"] != canonical(result).decode()
                    or not self._reservation_abort_cleanup_valid(
                        db, row, abort_receipt
                    )
                ):
                    raise ContextFault("context_integrity_failure")
                return result
            now = self.now()
            if not command.issued_at <= now < command.expires_at:
                raise ContextFault("context_reservation_abort_expiry_expired")
            namespace = command.namespace.model_dump(mode="json")
            if (
                row["state"] != "ABORTED"
                or row["owner"] != namespace["owner_id"]
                or row["project"] != namespace["project_id"]
                or row["revision"] != command.expected_revision
                or abort_receipt.context_cycle_id != row["id"]
                or abort_receipt.owner_id != row["owner"]
                or abort_receipt.project_id != row["project"]
                or abort_receipt.objective_id != namespace["objective_id"]
                or abort_receipt.idempotency_key != row["request_key"]
                or abort_receipt.request_digest != row["request_digest"]
                or abort_receipt.profile_digest != row["profile"]
                or abort_receipt.revision != row["revision"]
                or abort_receipt.retention_expires_at != row["expires"]
                or abort_receipt.abort_command_digest
                != row["reservation_abort_digest"]
                or row["reservation_abort_receipt"]
                != canonical(abort_envelope).decode()
                or now < row["expires"]
                or not self._reservation_never_bound(row)
                or not self._reservation_data_empty(db, row["id"])
            ):
                raise ContextFault("context_reservation_abort_expiry_denied")
            db.execute(
                "INSERT INTO retention_operations(cycle,operation,key,request_digest,result) "
                "VALUES(?,'expire_aborted_reservation',?,?,NULL)",
                (row["id"], command.idempotency_key, request_digest),
            )
            deletion_payload = ContextDeletionReceiptV2(
                context_cycle_id=row["id"],
                segment_chain_root=ZERO,
                context_seal_digest=None,
                retention_class=row["retention_class"],
                retention_expired_at=row["expires"],
                deleted_at=now,
                reason_code=command.reason_code,
                retention_operation_key=command.idempotency_key,
                retention_request_digest=request_digest,
                logical_deletion=True,
                cryptographic_erasure=False,
                local_key_destruction=None,
                key_destruction_receipt=None,
                key_destruction_receipt_digest=None,
                managed_key_provider_receipt_digest=None,
                remote_deletion_receipt_digest=None,
                checkpoint_manifest_checksum=None,
                checkpoint_object_ref_digest=None,
                checkpoint_durability_mode="unregistered",
                checkpoint_durability_registration_receipt_digest=None,
            ).model_dump(mode="json")
            receipt = self.signer.sign(deletion_payload)
            changed = db.execute(
                "UPDATE cycles SET state='EXPIRED',cleanup_state='COMPLETE',"
                "cleanup_operation_key=?,cleanup_request_digest=?,"
                "deletion_receipt=?,revision=revision+1 "
                "WHERE id=? AND state='ABORTED' AND revision=? AND binding IS NULL",
                (
                    command.idempotency_key,
                    request_digest,
                    canonical(receipt).decode(),
                    row["id"],
                    row["revision"],
                ),
            ).rowcount
            if changed != 1:
                raise ContextFault("context_stale_state")
            db.execute(
                "UPDATE retention_operations SET result=? WHERE cycle=? "
                "AND operation='expire_aborted_reservation' AND key=?",
                (
                    canonical(receipt).decode(),
                    row["id"],
                    command.idempotency_key,
                ),
            )
            self._event(
                db,
                row["id"],
                "context.expired",
                deletion_receipt_digest=digest(receipt),
                cryptographic_erasure=False,
            )
            return receipt

    def register_checkpoint_durability(self, signed_command: dict) -> dict:
        """Register an immutable checkpoint persistence mode before the first pack."""

        payload = verify(signed_command, self.authority_keys)
        try:
            command = ContextCheckpointDurabilityCommandV1.model_validate(payload)
        except (TypeError, ValueError) as exc:
            raise ContextFault("context_checkpoint_durability_schema") from exc
        if payload != command.model_dump(mode="json"):
            raise ContextFault("context_checkpoint_durability_schema")
        command_digest = digest(payload)
        with self.store.transaction() as db:
            row = self._cycle(db, command.namespace.context_cycle_id)
            stored_digest = row["checkpoint_durability_command_digest"]
            stored_receipt = row["checkpoint_durability_receipt"]
            if stored_digest is not None or stored_receipt is not None:
                if stored_digest is None or stored_receipt is None:
                    raise ContextFault("context_integrity_failure")
                if stored_digest != command_digest:
                    raise ContextFault("context_checkpoint_durability_conflict")
                try:
                    envelope = json.loads(stored_receipt)
                    receipt, receipt_payload = self._validated_durability_receipt(
                        envelope,
                        {self.signer.key_id: self.signer.key.public_key()},
                    )
                except (ContextFault, TypeError, ValueError) as exc:
                    raise ContextFault("context_integrity_failure") from exc
                if (
                    receipt_payload != receipt.model_dump(mode="json")
                    or receipt.namespace != command.namespace
                    or receipt.idempotency_key != command.idempotency_key
                    or receipt.mode != command.mode
                    or row["checkpoint_durability_mode"] != receipt.mode
                    or row["checkpoint_durability_legacy"]
                    or receipt.command_digest != stored_digest
                    or receipt.revision > row["revision"]
                ):
                    raise ContextFault("context_integrity_failure")
                return envelope
            if not command.issued_at <= self.now() < command.expires_at:
                raise ContextFault("context_checkpoint_durability_expired")
            try:
                binding_payload = json.loads(row["binding"])
                binding = ContextBindingV1.model_validate(binding_payload)
            except (TypeError, ValueError) as exc:
                raise ContextFault("context_checkpoint_durability_denied") from exc
            if (
                binding.namespace != command.namespace
                or row["binding_digest"] != digest(binding_payload)
                or row["state"] != "ACTIVE"
                or row["checkpoint"] != 0
                or row["checkpoint_manifest_checksum"] is not None
            ):
                raise ContextFault("context_checkpoint_durability_denied")
            if row["revision"] != command.expected_revision:
                raise ContextFault("context_stale_state")
            revision = row["revision"] + 1
            receipt_envelope = self.signer.sign(
                ContextCheckpointDurabilityReceiptV1(
                    namespace=command.namespace,
                    idempotency_key=command.idempotency_key,
                    mode=command.mode,
                    command_digest=command_digest,
                    effective_at=self.now(),
                    revision=revision,
                ).model_dump(mode="json")
            )
            changed = db.execute(
                "UPDATE cycles SET checkpoint_durability_mode=?,"
                "checkpoint_durability_command_digest=?,checkpoint_durability_receipt=?,"
                "checkpoint_durability_legacy=0,revision=? WHERE id=? AND revision=?",
                (
                    command.mode,
                    command_digest,
                    canonical(receipt_envelope).decode(),
                    revision,
                    row["id"],
                    row["revision"],
                ),
            ).rowcount
            if changed != 1:
                raise ContextFault("context_stale_state")
            self._event(
                db,
                row["id"],
                "context.checkpoint_durability_registered",
                mode=command.mode,
                command_digest=command_digest,
            )
            return receipt_envelope

    @staticmethod
    def _validated_durability_receipt(
        envelope: dict, keys: dict
    ) -> tuple[ContextCheckpointDurabilityReceiptV1, dict]:
        payload = verify(envelope, keys)
        model = (
            ContextCheckpointDurabilityReceiptV2
            if payload.get("schema_version")
            == "ContextCheckpointDurabilityReceiptV2"
            else ContextCheckpointDurabilityReceiptV1
        )
        receipt = model.model_validate(payload)
        if payload != receipt.model_dump(mode="json"):
            raise ContextFault("context_integrity_failure")
        return receipt, payload

    def _verify_checkpoint_durability_registration(self, row: sqlite3.Row) -> None:
        mode = row["checkpoint_durability_mode"]
        command_digest = row["checkpoint_durability_command_digest"]
        stored_receipt = row["checkpoint_durability_receipt"]
        if row["checkpoint_durability_legacy"]:
            if (
                mode != "remote_registered"
                or command_digest is not None
                or stored_receipt is not None
            ):
                raise ContextFault("context_integrity_failure")
            return
        if mode == "unregistered":
            if command_digest is not None or stored_receipt is not None:
                raise ContextFault("context_integrity_failure")
            raise ContextFault("context_checkpoint_durability_unregistered")
        if command_digest is None or stored_receipt is None:
            raise ContextFault("context_checkpoint_durability_unregistered")
        try:
            envelope = json.loads(stored_receipt)
            receipt, payload = self._validated_durability_receipt(
                envelope,
                {self.signer.key_id: self.signer.key.public_key()},
            )
        except (ContextFault, TypeError, ValueError) as exc:
            raise ContextFault("context_integrity_failure") from exc
        if (
            payload != receipt.model_dump(mode="json")
            or receipt.namespace.context_cycle_id != row["id"]
            or receipt.namespace.owner_id != row["owner"]
            or receipt.namespace.project_id != row["project"]
            or receipt.mode != mode
            or receipt.command_digest != command_digest
            or receipt.revision > row["revision"]
        ):
            raise ContextFault("context_integrity_failure")

    def _restore_checkpoint_durability_registration(
        self,
        snapshot: dict,
        namespace: dict,
        mode: str,
        receipt_keys: dict,
        hydrated_revision: int,
    ) -> tuple[str | None, dict | None, int, str | None]:
        fields = {
            "checkpoint_durability_command_digest",
            "checkpoint_durability_receipt",
        }
        present = fields & snapshot.keys()
        if not present:
            if mode != "remote_registered":
                raise ContextFault("context_hydrate_durability_denied")
            return None, None, 1, None
        if present != fields:
            raise ContextFault("context_integrity_failure")
        command_digest = snapshot["checkpoint_durability_command_digest"]
        source_envelope = snapshot["checkpoint_durability_receipt"]
        if (
            not isinstance(command_digest, str)
            or len(command_digest) != 64
            or any(character not in "0123456789abcdef" for character in command_digest)
            or not isinstance(source_envelope, dict)
        ):
            raise ContextFault("context_integrity_failure")
        try:
            source_receipt, source_payload = self._validated_durability_receipt(
                source_envelope, receipt_keys
            )
        except (ContextFault, TypeError, ValueError) as exc:
            raise ContextFault("context_integrity_failure") from exc
        if (
            source_payload != source_receipt.model_dump(mode="json")
            or source_receipt.namespace.model_dump(mode="json") != namespace
            or source_receipt.mode != mode
            or source_receipt.command_digest != command_digest
            or source_receipt.revision > hydrated_revision
        ):
            raise ContextFault("context_integrity_failure")
        if isinstance(source_receipt, ContextCheckpointDurabilityReceiptV2):
            # Each replacement verifies the immediate source host's V2 receipt,
            # then carries the original signed V1 registration forward. This
            # keeps the evidence constant-sized while preserving a resolvable
            # authority-issued root through arbitrarily many replacements.
            original_envelope = source_receipt.source_registration_receipt.model_dump(
                mode="json"
            )
            try:
                original_receipt = (
                    ContextCheckpointDurabilityReceiptV1.model_validate(
                        original_envelope["payload"]
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ContextFault("context_integrity_failure") from exc
            if (
                original_envelope["payload"]
                != original_receipt.model_dump(mode="json")
                or source_receipt.source_registration_receipt_digest
                != digest(original_envelope)
            ):
                raise ContextFault("context_integrity_failure")
        else:
            original_envelope = source_envelope
            original_receipt = source_receipt
        reattested = self.signer.sign(
            ContextCheckpointDurabilityReceiptV2(
                namespace=original_receipt.namespace,
                idempotency_key=original_receipt.idempotency_key,
                mode=original_receipt.mode,
                command_digest=original_receipt.command_digest,
                effective_at=original_receipt.effective_at,
                revision=original_receipt.revision,
                source_registration_receipt=SignedV1.model_validate(
                    original_envelope
                ),
                source_registration_receipt_digest=digest(original_envelope),
                reattested_at=self.now(),
            ).model_dump(mode="json")
        )
        return command_digest, reattested, 0, digest(original_envelope)

    def bind(self, signed_binding: dict, authority: dict, project_snapshot: dict) -> dict:
        binding_payload = verify(signed_binding, self.authority_keys)
        try:
            binding = ContextBindingV1.model_validate(binding_payload)
        except (TypeError, ValueError) as exc:
            raise ContextFault("context_binding_invalid") from exc
        return self._bind(
            binding,
            binding_payload,
            authority,
            project_snapshot,
            reservation_claim=None,
            reservation_receipt_digest=None,
        )

    def bind_v2(self, signed_command: dict, authority: dict, project_snapshot: dict) -> dict:
        payload = verify(signed_command, self.authority_keys)
        try:
            command = ContextBindingCommandV2.model_validate(payload)
            reservation_envelope = command.reservation_receipt.model_dump(mode="json")
            reservation_payload = verify(
                reservation_envelope,
                {self.signer.key_id: self.signer.key.public_key()},
            )
            reservation_claim = ContextReservationReceiptV2.model_validate(
                reservation_payload
            )
        except (ContextFault, TypeError, ValueError) as exc:
            raise ContextFault("context_binding_invalid") from exc
        if (
            payload != command.model_dump(mode="json")
            or reservation_payload != reservation_claim.model_dump(mode="json")
            or command.expected_reservation_revision != reservation_claim.revision
        ):
            raise ContextFault("context_binding_invalid")
        binding_payload = command.binding.model_dump(mode="json")
        return self._bind(
            command.binding,
            binding_payload,
            authority,
            project_snapshot,
            reservation_claim=reservation_claim,
            reservation_receipt_digest=digest(reservation_envelope),
        )

    def bind_v3(self, signed_command: dict, authority: dict, project_snapshot: dict) -> dict:
        payload = verify(signed_command, self.authority_keys)
        try:
            command = ContextBindingCommandV3.model_validate(payload)
            reservation_envelope = command.reservation_receipt.model_dump(mode="json")
            reservation_payload = verify(
                reservation_envelope,
                {self.signer.key_id: self.signer.key.public_key()},
            )
            reservation_claim = ContextReservationReceiptV3.model_validate(
                reservation_payload
            )
        except (ContextFault, TypeError, ValueError) as exc:
            raise ContextFault("context_binding_invalid") from exc
        if (
            payload != command.model_dump(mode="json")
            or reservation_payload != reservation_claim.model_dump(mode="json")
            or command.expected_reservation_revision != reservation_claim.revision
        ):
            raise ContextFault("context_binding_invalid")
        binding_payload = command.binding.model_dump(mode="json")
        return self._bind(
            command.binding,
            binding_payload,
            authority,
            project_snapshot,
            reservation_claim=reservation_claim,
            reservation_receipt_digest=digest(reservation_envelope),
        )

    def _bind(
        self,
        binding: ContextBindingV1,
        binding_payload: dict,
        authority: dict,
        project_snapshot: dict,
        *,
        reservation_claim: ContextReservationReceiptV2 | ContextReservationReceiptV3 | None,
        reservation_receipt_digest: str | None,
    ) -> dict:
        binding_digest = digest(binding_payload)
        # Authority comes from the same authenticated control plane; its exact
        # digest is already committed by the authorization receipt.
        if digest(authority) != binding.authorization_digest:
            raise ContextFault("context_authority_mismatch")
        if len(canonical(authority)) > self.profile.max_record_bytes:
            raise ContextFault("context_authority_limit")
        filter_text(canonical(authority).decode())
        snapshot_bytes = canonical(project_snapshot)
        if (
            len(snapshot_bytes) > self.profile.max_snapshot_bytes
            or digest(project_snapshot) != binding.graph_checksum
            or project_snapshot.get("project_id") != binding.namespace.project_id
            or project_snapshot.get("graph_id") != binding.project_graph_id
            or project_snapshot.get("policy_digest") != binding.policy_digest
        ):
            raise ContextFault("context_snapshot_mismatch")
        filter_text(snapshot_bytes.decode())
        ns = binding.namespace.model_dump()
        cycle = ns["context_cycle_id"]
        if binding.profile_digest != digest(self.profile) or binding.expires_at <= self.now():
            raise ContextFault("context_binding_invalid")
        with self.store.transaction() as db:
            row = self._cycle(db, cycle)
            if row["owner"] != ns["owner_id"] or row["project"] != ns["project_id"]:
                raise ContextFault("context_namespace_denied")
            if isinstance(reservation_claim, ContextReservationReceiptV3):
                if (
                    row["reservation_revival_kind"] != "expiry"
                    or row["reservation_revival_from_revision"]
                    != reservation_claim.prior_expiry_revision
                    or row["reservation_revival_receipt_digest"]
                    != reservation_claim.prior_expiry_receipt_digest
                    or row["reservation_revival_no_dispatch_receipt_digest"]
                    != reservation_claim.invocation_no_dispatch_receipt_digest
                    or row["reservation_revival_command_digest"]
                    != reservation_claim.revival_command_digest
                ):
                    raise ContextFault("context_stale_state")
            elif row["reservation_revival_kind"] == "expiry":
                raise ContextFault("context_binding_revision_required")
            if row["binding"]:
                if row["binding_digest"] != binding_digest:
                    raise ContextFault("context_binding_conflict")
                if reservation_claim is None:
                    if row["reservation_binding_revision"] is not None:
                        raise ContextFault("context_binding_revision_required")
                elif (
                    row["reservation_binding_receipt_digest"]
                    != reservation_receipt_digest
                    or row["reservation_binding_revision"]
                    != reservation_claim.revision
                    or reservation_claim.context_cycle_id != row["id"]
                    or reservation_claim.context_bucket_id != binding.context_bucket_id
                    or reservation_claim.request_digest != row["request_digest"]
                    or reservation_claim.profile_digest != row["profile"]
                ):
                    raise ContextFault("context_stale_state")
            else:
                if row["state"] != "RESERVED" or row["expires"] <= self.now():
                    raise ContextFault("context_state_conflict")
                if row["reservation_revival_from_revision"] is not None and (
                    reservation_claim is None
                ):
                    raise ContextFault("context_binding_revision_required")
                if reservation_claim is not None and (
                    reservation_claim.context_cycle_id != row["id"]
                    or reservation_claim.context_bucket_id != binding.context_bucket_id
                    or reservation_claim.request_digest != row["request_digest"]
                    or reservation_claim.profile_digest != row["profile"]
                    or reservation_claim.revision != row["revision"]
                ):
                    raise ContextFault("context_stale_state")
                key_ref = self.cycle_keys.ensure(ns) if self.cycle_keys is not None else None
                key_version = (
                    self._provider_key_version(key_ref) if key_ref is not None else None
                )
                if key_ref is not None and key_version is None:
                    raise ContextFault("context_key_version_unavailable")
                if (
                    key_ref is not None
                    and getattr(self.cycle_keys, "provider", None)
                    != "file-wrapped-dek/v1"
                ):
                    managed_ready, _, _ = self._managed_key_qualification(
                        expected_key_version=key_version
                    )
                    if not managed_ready:
                        raise ContextFault("context_managed_key_provider_unqualified")
                cipher = (
                    self.cycle_keys.cipher(key_ref, ns) if key_ref is not None else self.cipher
                )
                changed = db.execute(
                    "UPDATE cycles SET binding=?,binding_digest=?,authority=?,snapshot=?,"
                    "state='BOUND',expires=?,key_ref=?,key_version=?,retention_class=?,"
                    "reservation_binding_receipt_digest=?,reservation_binding_revision=?,"
                    "revision=revision+1 WHERE id=? AND revision=?",
                    (
                        canonical(binding_payload).decode(),
                        binding_digest,
                        cipher.encrypt(authority, ns),
                        cipher.encrypt(project_snapshot, {"namespace": ns, "plane": "P1"}),
                        binding.expires_at,
                        key_ref,
                        key_version,
                        binding.retention_class,
                        reservation_receipt_digest,
                        reservation_claim.revision if reservation_claim is not None else None,
                        cycle,
                        row["revision"],
                    ),
                ).rowcount
                if changed != 1:
                    raise ContextFault("context_stale_state")
                self._event(db, cycle, "context.bound", binding_digest=binding_digest)
                db.execute(
                    "UPDATE cycles SET state='HYDRATING',revision=revision+1 WHERE id=?", (cycle,)
                )
                self._event(db, cycle, "context.hydration_started")
                db.execute(
                    "UPDATE cycles SET state='ACTIVE',revision=revision+1 WHERE id=?", (cycle,)
                )
                self._event(db, cycle, "context.active")
            return self.signer.sign(
                {
                    "schema_version": "ContextBindingReceiptV1",
                    "context_cycle_id": cycle,
                    "binding_digest": binding_digest,
                }
            )

    def _authorize(self, db, envelope: dict, operation: str) -> tuple[CapabilityV1, sqlite3.Row]:
        cap = CapabilityV1.model_validate(verify(envelope, self.authority_keys))
        row = self._cycle(db, cap.namespace.context_cycle_id)
        if not row["binding"]:
            raise ContextFault("context_unbound")
        binding = json.loads(row["binding"])
        if digest(binding) != row["binding_digest"]:
            raise ContextFault("context_integrity_failure")
        if not namespace_matches(binding["namespace"], cap.namespace.model_dump()):
            raise ContextFault("context_namespace_denied")
        if (
            operation not in cap.operations
            or cap.expires_at <= self.now()
            or (
                operation not in {"expire", "retention", "status"}
                and cap.expires_at > binding["expires_at"]
            )
        ):
            raise ContextFault("context_capability_denied")
        if (
            cap.binding_digest != row["binding_digest"]
            or cap.policy_digest != binding["policy_digest"]
            or cap.profile_digest != row["profile"]
        ):
            raise ContextFault("context_authority_mismatch")
        lane = db.execute(
            "SELECT fence FROM fences WHERE cycle=? AND lane=?", (row["id"], cap.lane_id)
        ).fetchone()
        if cap.fence != row["fence"] or (lane and cap.fence != lane["fence"]):
            raise ContextFault("context_stale_fence")
        if row["state"] in {"EXPIRED", "ABORTED", "QUARANTINED"} and operation not in {
            "status",
            "retention",
        }:
            raise ContextFault("context_terminal")
        return cap, row

    def _operation_aad(self, cycle: str, operation: str, key: str) -> dict:
        return {"cycle": cycle, "operation": operation, "key": key}

    def _replay(
        self,
        db,
        cycle: str,
        operation: str,
        key: str,
        request_digest: str,
        cipher: EnvelopeCipher,
    ):
        row = db.execute(
            "SELECT * FROM operations WHERE cycle=? AND operation=? AND key=?",
            (cycle, operation, key),
        ).fetchone()
        if row is None:
            return None
        if row["request_digest"] != request_digest:
            raise ContextFault("context_idempotency_conflict")
        return cipher.decrypt(
            row["result"],
            self._operation_aad(cycle, operation, key),
            self.profile.max_checkpoint_bytes * (2 if operation == "checkpoint" else 1),
        )

    def _remember(
        self,
        db,
        cycle: str,
        operation: str,
        key: str,
        request_digest: str,
        result: dict,
        cipher: EnvelopeCipher,
    ):
        limit = self.profile.max_operations_per_cycle + (
            4 if operation in {"checkpoint", "seal"} else 0
        )
        if (
            db.execute("SELECT count(*) FROM operations WHERE cycle=?", (cycle,)).fetchone()[0]
            >= limit
        ):
            raise ContextFault("context_operation_quota")
        blob = cipher.encrypt(result, self._operation_aad(cycle, operation, key))
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?,?)",
            (cycle, operation, key, request_digest, blob),
        )
        return result

    def _rate(self, db, cycle):
        minute = self.now() // 60
        db.execute("DELETE FROM call_budgets WHERE cycle=? AND minute<?", (cycle, minute))
        row = db.execute(
            "SELECT calls FROM call_budgets WHERE cycle=? AND minute=?", (cycle, minute)
        ).fetchone()
        if row and row[0] >= self.profile.max_calls_per_minute:
            raise ContextFault("context_rate_quota")
        db.execute(
            "INSERT INTO call_budgets VALUES(?,?,1) ON CONFLICT(cycle,minute) DO UPDATE SET calls=calls+1",
            (cycle, minute),
        )

    def _snapshot(self, cycle, ns):
        if not cycle["snapshot"]:
            raise ContextFault("context_snapshot_required")
        value = self._cipher(cycle, ns).decrypt(
            cycle["snapshot"], {"namespace": ns, "plane": "P1"}, self.profile.max_snapshot_bytes
        )
        if digest(value) != json.loads(cycle["binding"])["graph_checksum"]:
            raise ContextFault("context_integrity_failure")
        return value

    def rebuild_index(self, capability: dict, idempotency_key: str) -> dict:
        """Atomically rebuild one cycle's derived postings from its verified chain."""

        with self._transaction(capability) as db:
            cap, cycle = self._authorize(db, capability, "checkpoint")
            if cap.role != "durability" or cycle["state"] not in {"ACTIVE", "SEALING"}:
                raise ContextFault("context_rebuild_denied")
            namespace = cap.namespace.model_dump()
            cipher = self._cipher(cycle, namespace)
            request_digest = digest({"key": idempotency_key})
            replay = self._replay(
                db, cycle["id"], "rebuild", idempotency_key, request_digest, cipher
            )
            if replay is not None:
                return replay
            db.execute("DELETE FROM postings WHERE cycle=?", (cycle["id"],))
            previous, count = ZERO, 0
            for row in db.execute(
                "SELECT * FROM segments WHERE cycle=? ORDER BY seq", (cycle["id"],)
            ):
                count += 1
                if row["seq"] != count or row["previous"] != previous:
                    raise ContextFault("context_chain_corrupt")
                value = self._record(row, namespace, cipher)
                for word in words(value["text"]):
                    db.execute(
                        "INSERT INTO postings VALUES(?,?,?)",
                        (cycle["id"], cipher.token(namespace, word), count),
                    )
                previous = row["root"]
            if count != cycle["cursor"] or previous != cycle["root"]:
                raise ContextFault("context_chain_corrupt")
            db.execute(
                "UPDATE cycles SET revision=revision+1 WHERE id=?", (cycle["id"],)
            )
            self._event(
                db,
                cycle["id"],
                "context.index_rebuilt",
                records=count,
                root=previous,
            )
            result = self.signer.sign(
                {
                    "schema_version": "ContextIndexRebuildReceiptV1",
                    "context_cycle_id": cycle["id"],
                    "records": count,
                    "root": previous,
                    "index_implementation": self.profile.index_implementation,
                    "index_version": self.profile.index_version,
                }
            )
            return self._remember(
                db,
                cycle["id"],
                "rebuild",
                idempotency_key,
                request_digest,
                result,
                cipher,
            )

    def append(self, capability: dict, request: AppendRequestV1) -> dict:
        text = filter_text(request.text)
        size = len(text.encode())
        self._space()
        with self._transaction(capability) as db:
            cap, cycle = self._authorize(db, capability, "append")
            writable(cap, request.plane)
            ns = cap.namespace.model_dump()
            cipher = self._cipher(cycle, ns)
            req_digest = digest(
                {
                    "request": request.model_dump(),
                    "principal": cap.principal_id,
                    "lane": cap.lane_id,
                    "task": cap.task_id,
                    "source": cap.source_class,
                    "action": request.action_ref,
                }
            )
            replay = self._replay(
                db, cycle["id"], "append", request.idempotency_key, req_digest, cipher
            )
            if replay:
                return replay
            if cycle["state"] != "ACTIVE" or cycle["expires"] <= self.now():
                raise ContextFault("context_not_active")
            self._rate(db, cycle["id"])
            if (
                size > min(self.profile.max_record_bytes, cap.max_bytes)
                or cycle["bytes"] + size > self.profile.max_indexed_bytes
                or cycle["cursor"] >= self.profile.max_records
            ):
                raise ContextFault("context_write_quota")
            if request.plane in {"P2", "P4"} and not request.evidence_refs:
                raise ContextFault("context_evidence_required")
            seq = cycle["cursor"] + 1
            record_id = (
                "rec_" + digest({"namespace": ns, "sequence": seq, "request": req_digest})[:40]
            )
            binding = json.loads(cycle["binding"])
            metadata = {
                "schema_version": "ContextRecordV1",
                "record_id": record_id,
                "namespace": ns,
                "sequence": seq,
                "previous_digest": cycle["root"],
                "plane": request.plane,
                "lane_id": cap.lane_id,
                "task_id": cap.task_id,
                "role": cap.role,
                "source_class": cap.source_class,
                "producer_principal": cap.principal_id,
                "action_ref": request.action_ref,
                "evidence_refs": request.evidence_refs,
                "repo_sha": binding["repo_main_sha"],
                "graph_revision": binding["graph_revision"],
                "shared_ir_digest": binding["shared_ir_digest"],
                "trust_class": "verified"
                if cap.source_class
                in {"project_memory_verified", "repository_verified", "ci_verified"}
                else "untrusted",
                "content_type": request.content_type,
                "sensitivity": "filtered",
                "redaction_digest": binding["redaction_digest"],
                "content_digest": hashlib.sha256(text.encode()).hexdigest(),
                "token_estimate": token_bound(text),
                "embedding_version": self.profile.embedding_version,
                "created_at": self.now(),
                "expires_at": cycle["expires"],
                "supersedes": request.supersedes,
                "promotion_state": "candidate" if request.plane == "P4" else "none",
            }
            root = digest(metadata)
            encrypted = cipher.encrypt(
                {"text": text}, {"namespace": ns, "record_id": record_id, "metadata_digest": root}
            )
            if request.supersedes:
                old = db.execute(
                    "SELECT plane,lane FROM segments WHERE id=? AND cycle=?",
                    (request.supersedes, cycle["id"]),
                ).fetchone()
                if old is None or old["plane"] != request.plane or old["lane"] != cap.lane_id:
                    raise ContextFault("context_supersession_denied")
                db.execute("UPDATE segments SET superseded=1 WHERE id=?", (request.supersedes,))
            db.execute(
                "INSERT INTO segments(cycle,seq,id,plane,lane,metadata,payload,digest,previous,root,tokens,expires) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cycle["id"],
                    seq,
                    record_id,
                    request.plane,
                    cap.lane_id,
                    canonical(metadata).decode(),
                    encrypted,
                    metadata["content_digest"],
                    cycle["root"],
                    root,
                    metadata["token_estimate"],
                    cycle["expires"],
                ),
            )
            for word in words(text):
                db.execute(
                    "INSERT INTO postings VALUES(?,?,?)",
                    (cycle["id"], cipher.token(ns, word), seq),
                )
            db.execute(
                "UPDATE cycles SET cursor=?,root=?,bytes=bytes+?,revision=revision+1 WHERE id=?",
                (seq, root, size, cycle["id"]),
            )
            self._event(
                db,
                cycle["id"],
                "context.record_appended",
                record_id=record_id,
                root=root,
                sequence=seq,
            )
            result = self.signer.sign(
                {
                    "schema_version": "ContextAppendReceiptV1",
                    "context_cycle_id": cycle["id"],
                    "record_id": record_id,
                    "content_digest": metadata["content_digest"],
                    "cursor": seq,
                    "root": root,
                }
            )
            return self._remember(
                db, cycle["id"], "append", request.idempotency_key, req_digest, result, cipher
            )

    def _record(self, row, namespace: dict, cipher: EnvelopeCipher) -> dict:
        metadata = json.loads(row["metadata"])
        if (
            metadata["namespace"] != namespace
            or digest(metadata) != row["root"]
            or metadata["previous_digest"] != row["previous"]
            or metadata["record_id"] != row["id"]
            or metadata["sequence"] != row["seq"]
            or metadata["plane"] != row["plane"]
            or metadata["lane_id"] != row["lane"]
            or metadata["expires_at"] != row["expires"]
            or metadata["token_estimate"] != row["tokens"]
            or metadata["content_digest"] != row["digest"]
        ):
            raise ContextFault("context_integrity_failure")
        body = cipher.decrypt(
            row["payload"],
            {"namespace": namespace, "record_id": row["id"], "metadata_digest": row["root"]},
            self.profile.max_record_bytes * 2,
        )
        text = filter_text(body["text"])
        if hashlib.sha256(text.encode()).hexdigest() != row["digest"]:
            raise ContextFault("context_integrity_failure")
        return {"metadata": metadata, "text": text}

    def retrieve(self, capability: dict, request: RetrieveRequestV1) -> dict:
        self._space()
        filter_text(request.query)
        with self._transaction(capability) as db:
            cap, cycle = self._authorize(db, capability, "retrieve")
            ns = cap.namespace.model_dump()
            cipher = self._cipher(cycle, ns)
            req_digest = digest(
                {
                    "request": request.model_dump(),
                    "principal": cap.principal_id,
                    "lane": cap.lane_id,
                    "task": cap.task_id,
                    "role": cap.role,
                }
            )
            replay = self._replay(
                db, cycle["id"], "retrieve", request.turn_id, req_digest, cipher
            )
            if replay:
                return replay
            if cycle["state"] != "ACTIVE" or cycle["expires"] <= self.now():
                raise ContextFault("context_not_active")
            self._rate(db, cycle["id"])
            if request.event_cursor is not None and request.event_cursor != cycle["cursor"]:
                raise ContextFault("context_stale_cursor")
            budget = min(
                self.profile.max_capsule_tokens,
                cap.max_tokens,
                request.native_window // 4,
                request.remaining_prompt_budget,
                max(0, request.native_window * 4 // 5 - request.used_prompt_tokens),
            )
            if budget < token_bound(NOTICE) + 32:
                raise ContextFault("context_prompt_budget")
            terms = [cipher.token(ns, word) for word in words(request.query)[:32]]
            rows = []
            if terms:
                # Exact namespace AND lane/plane eligibility precede ranking and LIMIT.
                shared = "s.plane='P2'"
                own = "(s.plane='P3' AND s.lane=?)"
                visibility = f"({shared} OR {own})"
                params = [cycle["id"], self.now(), cap.lane_id]
                if cap.role in {"coordinator", "reviewer", "verifier"}:
                    visibility = (
                        "(s.plane IN ('P2','P3')"
                        + (" OR s.plane='P4'" if cap.role != "reviewer" else "")
                        + ")"
                    )
                    params = [cycle["id"], self.now()]
                query = f"""SELECT s.*,count(p.term) AS score FROM segments s JOIN postings p
                    ON p.cycle=s.cycle AND p.seq=s.seq WHERE s.cycle=? AND s.expires>?
                    AND s.superseded=0 AND s.quarantined=0 AND {visibility}
                    AND p.term IN ({",".join("?" for _ in terms)})
                    GROUP BY s.cycle,s.seq ORDER BY score DESC,s.seq DESC LIMIT ?"""
                rows = db.execute(query, [*params, *terms, self.profile.max_candidates]).fetchall()
            entries: list[dict] = []
            considered, dropped = 0, 0
            snapshot = self._snapshot(cycle, ns)
            binding = json.loads(cycle["binding"])
            pinned: dict[str, Any] = {
                "graph_id": binding["project_graph_id"],
                "revision": binding["graph_revision"],
                "checksum": binding["graph_checksum"],
                "nodes": [],
            }
            query_words = set(words(request.query))
            ranked = sorted(
                snapshot.get("nodes", []),
                key=lambda node: (
                    -len(query_words & set(words(canonical(node).decode()))),
                    node["id"],
                ),
            )
            for node in ranked[: self.profile.max_candidates]:
                if node.get("tombstone") or not query_words & set(words(canonical(node).decode())):
                    continue
                candidate: dict[str, Any] = {
                    "notice": NOTICE,
                    "project_snapshot": {**pinned, "nodes": [*pinned["nodes"], node]},
                    "entries": [],
                }
                if token_bound(canonical(candidate).decode()) <= budget // 2:
                    pinned["nodes"].append(node)
            for row in rows:
                if not visible(cap, row["plane"], row["lane"]):
                    raise ContextFault("context_namespace_denied")
                entry = self._record(row, ns, cipher)
                considered += row["tokens"]
                candidate = {
                    "notice": NOTICE,
                    "project_snapshot": pinned,
                    "entries": [*entries, entry],
                }
                if token_bound(canonical(candidate).decode()) > budget:
                    dropped += 1
                    continue
                entries.append(entry)
            content = canonical(
                {"notice": NOTICE, "project_snapshot": pinned, "entries": entries}
            ).decode()
            if token_bound(content) > budget:
                raise ContextFault("context_prompt_budget")
            payload = {
                "schema_version": "ContextCapsuleV1",
                "capsule_id": "cap_" + digest({"cycle": cycle["id"], "request": req_digest})[:40],
                "context_cycle_id": cycle["id"],
                "request_digest": req_digest,
                "cursor": cycle["cursor"],
                "lane_id": cap.lane_id,
                "task_id": cap.task_id,
                "role": cap.role,
                "retrieval_policy_digest": cap.policy_digest,
                "entries": entries,
                "considered_tokens": considered,
                "selected_records": len(entries),
                "dropped_records": dropped,
                "injected_tokens": token_bound(content),
                "exclusion_reasons": {"budget": dropped},
                "content": content,
                "expires_at": min(cap.expires_at, self.now() + 300),
            }
            payload["capsule_digest"] = digest(payload)
            result = self.signer.sign(payload)
            self._event(
                db,
                cycle["id"],
                "context.capsule_built",
                capsule_digest=payload["capsule_digest"],
                injected_tokens=payload["injected_tokens"],
            )
            return self._remember(
                db, cycle["id"], "retrieve", request.turn_id, req_digest, result, cipher
            )

    def checkpoint(
        self,
        capability: dict,
        idempotency_key: str,
        *,
        transport_max_bytes: int | None = None,
        transport_snapshot_max_bytes: int | None = None,
    ) -> dict:
        self._space()
        checkpoint_limit = self.profile.max_checkpoint_bytes
        snapshot_limit = self.profile.max_checkpoint_bytes
        if transport_max_bytes is not None:
            if type(transport_max_bytes) is not int or transport_max_bytes < 1024:
                raise ContextFault("context_checkpoint_transport_limit")
            checkpoint_limit = min(checkpoint_limit, transport_max_bytes)
        if transport_snapshot_max_bytes is not None:
            if (
                type(transport_snapshot_max_bytes) is not int
                or transport_snapshot_max_bytes < 1024
            ):
                raise ContextFault("context_checkpoint_transport_limit")
            snapshot_limit = min(
                snapshot_limit, transport_snapshot_max_bytes
            )
        with self._transaction(capability) as db:
            cap, cycle = self._authorize(db, capability, "checkpoint")
            if cap.role not in {"coordinator", "verifier", "durability"}:
                raise ContextFault("context_role_denied")
            ns = cap.namespace.model_dump()
            cipher = self._cipher(cycle, ns)
            replay = self._replay(
                db,
                cycle["id"],
                "checkpoint",
                idempotency_key,
                digest({"key": idempotency_key}),
                cipher,
            )
            if replay:
                if transport_max_bytes is not None:
                    try:
                        replay_pack = base64.b64decode(
                            replay.get("pack", ""), validate=True
                        )
                    except (TypeError, ValueError) as exc:
                        raise ContextFault("context_integrity_failure") from exc
                    if (
                        base64.b64encode(replay_pack).decode("ascii")
                        != replay.get("pack")
                    ):
                        raise ContextFault("context_integrity_failure")
                    if len(replay_pack) > checkpoint_limit:
                        raise ContextFault("context_checkpoint_transport_limit")
                return replay
            if cycle["state"] not in {"ACTIVE", "SEALING"}:
                raise ContextFault("context_not_active")
            if cycle["checkpoint_durability_mode"] not in {
                "unregistered",
                "local_ephemeral",
                "remote_registered",
            }:
                raise ContextFault("context_integrity_failure")
            self._verify_checkpoint_durability_registration(cycle)
            records: list[dict] = []
            previous = ZERO
            for row in db.execute(
                "SELECT * FROM segments WHERE cycle=? ORDER BY seq", (cycle["id"],)
            ):
                if row["previous"] != previous or row["seq"] != len(records) + 1:
                    raise ContextFault("context_chain_corrupt")
                entry = self._record(row, ns, cipher)
                records.append(
                    {
                        **entry,
                        "root": row["root"],
                        "superseded": row["superseded"],
                        "quarantined": row["quarantined"],
                    }
                )
                previous = row["root"]
            if previous != cycle["root"] or len(records) != cycle["cursor"]:
                raise ContextFault("context_chain_corrupt")
            snapshot = {
                "schema_version": "ContextCheckpointPackV1",
                "binding": json.loads(cycle["binding"]),
                "authority": cipher.decrypt(
                    cycle["authority"], ns, self.profile.max_record_bytes
                ),
                "project_snapshot": self._snapshot(cycle, ns),
                "records": records,
                "cursor": cycle["cursor"],
                "root": previous,
                "request_key": cycle["request_key"],
                "request_digest": cycle["request_digest"],
                "fence": cycle["fence"],
                "revision": cycle["revision"],
                "checkpoint": cycle["checkpoint"] + 1,
                "key_ref": cycle["key_ref"],
                "key_provider": (
                    getattr(self.cycle_keys, "provider", None)
                    if cycle["key_ref"] is not None
                    else None
                ),
                "key_version": cycle["key_version"],
                "retention_class": cycle["retention_class"],
                "checkpoint_durability_mode": cycle["checkpoint_durability_mode"],
                "reservation_protocol_version": cycle["reservation_protocol_version"],
            }
            if not cycle["checkpoint_durability_legacy"]:
                snapshot["checkpoint_durability_command_digest"] = cycle[
                    "checkpoint_durability_command_digest"
                ]
                snapshot["checkpoint_durability_receipt"] = json.loads(
                    cycle["checkpoint_durability_receipt"]
                )
            snapshot["state"] = cycle["state"]
            # Replays survive host replacement too, including immutable capsules.
            snapshot["operations"] = [
                {**dict(row), "result": base64.b64encode(row["result"]).decode()}
                for row in db.execute(
                    "SELECT * FROM operations WHERE cycle=? AND operation!='checkpoint'",
                    (cycle["id"],),
                )
            ]
            if len(canonical(snapshot)) > snapshot_limit:
                raise ContextFault("context_checkpoint_quota")
            pack = cipher.encrypt(snapshot, {"namespace": ns, "domain": "checkpoint/v1"})
            if len(pack) > checkpoint_limit:
                raise ContextFault("context_checkpoint_quota")
            checksum = hashlib.sha256(pack).hexdigest()
            number = cycle["checkpoint"] + 1
            receipt = self.signer.sign(
                ContextCheckpointReceiptV1(
                    namespace=cap.namespace,
                    checkpoint_number=number,
                    cursor=cycle["cursor"],
                    segment_chain_root=previous,
                    manifest_checksum=checksum,
                    durability="local",
                    bytes=len(pack),
                    records=len(records),
                    policy_digest=cap.policy_digest,
                    key_ref=cycle["key_ref"],
                    key_provider=(
                        getattr(self.cycle_keys, "provider", None)
                        if cycle["key_ref"] is not None
                        else None
                    ),
                    key_version=cycle["key_version"],
                ).model_dump()
            )
            result = {"receipt": receipt, "pack": base64.b64encode(pack).decode()}
            db.execute(
                "UPDATE cycles SET checkpoint=?,checkpoint_manifest_checksum=?,"
                "revision=revision+1 WHERE id=?",
                (number, checksum, cycle["id"]),
            )
            self._event(
                db,
                cycle["id"],
                "context.checkpoint_completed",
                manifest_checksum=checksum,
                checkpoint_number=number,
            )
            return self._remember(
                db,
                cycle["id"],
                "checkpoint",
                idempotency_key,
                digest({"key": idempotency_key}),
                result,
                cipher,
            )

    def hydrate(
        self,
        signed_grant: dict,
        receipt: dict,
        pack: bytes,
        receipt_keys: dict,
        *,
        snapshot_max_bytes: int | None = None,
    ) -> dict:
        """Host replacement needs a fresh fence and the exact cloud-selected root."""
        hydrate_snapshot_limit = self.profile.max_checkpoint_bytes
        if snapshot_max_bytes is not None:
            if type(snapshot_max_bytes) is not int or snapshot_max_bytes < 1024:
                raise ContextFault("context_checkpoint_transport_limit")
            hydrate_snapshot_limit = min(
                hydrate_snapshot_limit, snapshot_max_bytes
            )
        grant = verify(signed_grant, self.authority_keys)
        legacy_grant = {"operation", "namespace", "manifest_checksum", "fence", "expires_at"}
        keyed_grant = legacy_grant | {"key_ref", "key_provider", "key_version"}
        hydrate_command_digest: str | None = None
        expected_source_revision: str | None = None
        if grant.get("schema_version") == "ContextHydrateCommandV2":
            try:
                command = ContextHydrateCommandV2.model_validate(grant)
            except (TypeError, ValueError) as exc:
                raise ContextFault("context_hydrate_grant_invalid") from exc
            if grant != command.model_dump(mode="json"):
                raise ContextFault("context_hydrate_grant_invalid")
            grant = command.model_dump(mode="json")
            hydrate_command_digest = digest(grant)
            expected_source_revision = command.expected_source_revision
        elif (
            frozenset(grant) not in {frozenset(legacy_grant), frozenset(keyed_grant)}
            or grant.get("operation") != "hydrate"
        ):
            raise ContextFault("context_hydrate_grant_invalid")
        try:
            claim = ContextCheckpointReceiptV1.model_validate(
                verify(receipt, receipt_keys)
            ).model_dump()
        except (ValueError, TypeError) as exc:
            raise ContextFault("context_checkpoint_receipt_invalid") from exc
        checksum = hashlib.sha256(pack).hexdigest()
        hydrate_request_digest = digest(
            {
                "grant": signed_grant,
                "checkpoint_receipt": receipt,
                "manifest_checksum": checksum,
            }
        )
        if (
            claim.get("schema_version") != "ContextCheckpointReceiptV1"
            or checksum != grant["manifest_checksum"]
            or checksum != claim["manifest_checksum"]
            or claim["namespace"] != grant["namespace"]
            or len(pack) > self.profile.max_checkpoint_bytes
        ):
            raise ContextFault("context_integrity_failure")
        if hydrate_command_digest is None:
            legacy_now = self.now()
            if (
                type(grant.get("expires_at")) is not int
                or not legacy_now < grant["expires_at"] <= legacy_now + 900
            ):
                raise ContextFault("context_capability_denied")
        key_ref = claim.get("key_ref")
        key_provider = claim.get("key_provider")
        key_version = claim.get("key_version")
        if hydrate_command_digest is not None and (
            grant.get("key_ref") != key_ref
            or grant.get("key_provider") != key_provider
            or grant.get("key_version") != key_version
        ):
            raise ContextFault("context_hydrate_grant_invalid")
        if key_ref is None:
            if (
                (hydrate_command_digest is None and set(grant) != legacy_grant)
                or key_provider is not None
                or key_version is not None
            ):
                raise ContextFault("context_hydrate_grant_invalid")
            cipher = self.cipher
        else:
            if (
                (hydrate_command_digest is None and set(grant) != keyed_grant)
                or grant["key_ref"] != key_ref
                or grant["key_provider"] != key_provider
                or grant["key_version"] != key_version
                or self.cycle_keys is None
                or key_provider != getattr(self.cycle_keys, "provider", None)
                or key_version != self._provider_key_version(key_ref)
            ):
                raise ContextFault("context_hydrate_grant_invalid")
            if key_provider != "file-wrapped-dek/v1":
                managed_ready, _, _ = self._managed_key_qualification(
                    expected_key_version=key_version
                )
                if not managed_ready:
                    raise ContextFault("context_managed_key_provider_unqualified")
            cipher = self.cycle_keys.cipher(key_ref, grant["namespace"])
        snapshot = cipher.decrypt(
            pack,
            {"namespace": grant["namespace"], "domain": "checkpoint/v1"},
            hydrate_snapshot_limit,
        )
        binding_payload = snapshot["binding"]
        binding = ContextBindingV1.model_validate(binding_payload)
        hydrated_revision = snapshot.get("revision", 1)
        reservation_protocol_version = snapshot.get(
            "reservation_protocol_version", 1
        )
        checkpoint_durability_mode = snapshot.get(
            "checkpoint_durability_mode", "remote_registered"
        )
        if (
            binding.namespace.model_dump() != grant["namespace"]
            or binding.profile_digest != digest(self.profile)
            or grant["fence"] <= snapshot["fence"]
            or binding.expires_at <= self.now()
            or snapshot["root"] != claim["segment_chain_root"]
            or snapshot["cursor"] != claim["cursor"]
            or snapshot["checkpoint"] != claim["checkpoint_number"]
            or snapshot.get("key_ref") != key_ref
            or snapshot.get("key_provider") != key_provider
            or snapshot.get("key_version") != key_version
            or type(hydrated_revision) is not int
            or hydrated_revision < 1
            or type(reservation_protocol_version) is not int
            or reservation_protocol_version not in {1, 2, 3}
            or checkpoint_durability_mode not in {"local_ephemeral", "remote_registered"}
        ):
            raise ContextFault("context_hydrate_grant_invalid")
        if checkpoint_durability_mode != "remote_registered":
            raise ContextFault("context_hydrate_durability_denied")
        (
            durability_command_digest,
            restored_durability_receipt,
            durability_legacy,
            source_durability_receipt_digest,
        ) = self._restore_checkpoint_durability_registration(
            snapshot,
            grant["namespace"],
            checkpoint_durability_mode,
            receipt_keys,
            hydrated_revision,
        )
        if (
            snapshot["state"] not in {"ACTIVE", "SEALING"}
            or digest(snapshot["authority"]) != binding.authorization_digest
            or digest(snapshot["project_snapshot"]) != binding.graph_checksum
        ):
            raise ContextFault("context_integrity_failure")
        previous, records = ZERO, snapshot["records"]
        if len(records) != snapshot["cursor"] or len(records) > self.profile.max_records:
            raise ContextFault("context_integrity_failure")
        for sequence, record in enumerate(records, 1):
            meta, text = record["metadata"], filter_text(record["text"])
            if (
                meta["namespace"] != grant["namespace"]
                or meta["previous_digest"] != previous
                or meta["sequence"] != sequence
                or digest(meta) != record["root"]
                or hashlib.sha256(text.encode()).hexdigest() != meta["content_digest"]
            ):
                raise ContextFault("context_integrity_failure")
            previous = record["root"]
        if previous != snapshot["root"]:
            raise ContextFault("context_chain_corrupt")
        ns, cycle = grant["namespace"], binding.namespace.context_cycle_id
        hydrate_fields = dict(
            namespace=binding.namespace,
            context_cycle_id=cycle,
            manifest_checksum=checksum,
            checkpoint_number=claim["checkpoint_number"],
            cursor=claim["cursor"],
            root=previous,
            key_ref=key_ref,
            key_provider=key_provider,
            key_version=key_version,
            replay_digest=digest(snapshot["operations"]),
            fence=grant["fence"],
        )
        def hydrate_result_payload(durability_receipt: dict | None) -> dict:
            if hydrate_command_digest is not None:
                if expected_source_revision is None:
                    raise ContextFault("context_hydrate_source_mismatch")
                return ContextHydrateReceiptV2(
                    **hydrate_fields,
                    hydrate_command_digest=hydrate_command_digest,
                    expected_source_revision=expected_source_revision,
                    checkpoint_durability_registration_receipt=(
                        SignedV1.model_validate(durability_receipt)
                        if durability_receipt is not None
                        else None
                    ),
                    checkpoint_durability_registration_receipt_digest=(
                        digest(durability_receipt)
                        if durability_receipt is not None
                        else None
                    ),
                ).model_dump(mode="json")
            return ContextHydrateReceiptV1(**hydrate_fields).model_dump(
                mode="json"
            )

        hydrate_payload = hydrate_result_payload(restored_durability_receipt)

        with self.store.transaction() as db:
            existing = db.execute("SELECT * FROM cycles WHERE id=?", (cycle,)).fetchone()
            if existing:
                try:
                    exact_replay = self._hydrate_replay_exact(
                        db,
                        existing,
                        snapshot,
                        ns,
                        cipher,
                        checksum=checksum,
                        fence=grant["fence"],
                        binding_digest=digest(binding_payload),
                        durability_mode=checkpoint_durability_mode,
                        revision=hydrated_revision,
                        reservation_protocol_version=(
                            reservation_protocol_version
                        ),
                        durability_command_digest=durability_command_digest,
                        durability_legacy=durability_legacy,
                        source_durability_receipt_digest=(
                            source_durability_receipt_digest
                        ),
                        hydrate_command_digest=hydrate_command_digest,
                        hydrate_request_digest=hydrate_request_digest,
                    )
                except (ContextFault, TypeError, ValueError):
                    exact_replay = False
                if exact_replay:
                    if hydrate_command_digest is not None:
                        try:
                            persisted_result = json.loads(existing["hydrate_receipt"])
                            persisted_payload = verify(
                                persisted_result,
                                {self.signer.key_id: self.signer.key.public_key()},
                            )
                            persisted_claim = ContextHydrateReceiptV2.model_validate(
                                persisted_payload
                            )
                        except (ContextFault, TypeError, ValueError) as exc:
                            raise ContextFault("context_integrity_failure") from exc
                        if (
                            persisted_payload
                            != persisted_claim.model_dump(mode="json")
                            or persisted_payload
                            != hydrate_result_payload(
                                json.loads(
                                    existing["checkpoint_durability_receipt"]
                                )
                                if not durability_legacy
                                else None
                            )
                        ):
                            raise ContextFault("context_integrity_failure")
                        return persisted_result
                    replay_durability_receipt = (
                        None
                        if durability_legacy
                        else json.loads(existing["checkpoint_durability_receipt"])
                    )
                    return self.signer.sign(
                        hydrate_result_payload(replay_durability_receipt)
                    )
                raise ContextFault("context_hydrate_conflict")
            now = self.now()
            if (
                hydrate_command_digest is not None
                and (
                    type(grant.get("expires_at")) is not int
                    or not grant["issued_at"] <= now < grant["expires_at"] <= now + 900
                )
            ):
                raise ContextFault("context_capability_denied")
            if hydrate_command_digest is not None:
                runtime_health = self.health_v2()
                if (
                    not runtime_health["runtime_build_ready"]
                    or expected_source_revision
                    != self.build_qualification.source_revision
                ):
                    raise ContextFault("context_hydrate_source_mismatch")
            hydrate_result = self.signer.sign(hydrate_payload)
            db.execute(
                "INSERT INTO cycles(id,owner,project,request_key,request_digest,profile,state,"
                "revision,fence,"
                "binding,binding_digest,authority,snapshot,cursor,root,bytes,expires,checkpoint,"
                "key_ref,key_version,retention_class,checkpoint_manifest_checksum,"
                "checkpoint_durability_mode,checkpoint_durability_legacy,"
                "checkpoint_durability_command_digest,checkpoint_durability_receipt,"
                "hydrate_command_digest,hydrate_request_digest,hydrate_receipt,"
                "reservation_protocol_version) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cycle,
                    ns["owner_id"],
                    ns["project_id"],
                    snapshot["request_key"],
                    snapshot["request_digest"],
                    digest(self.profile),
                    snapshot["state"],
                    hydrated_revision,
                    grant["fence"],
                    canonical(binding_payload).decode(),
                    digest(binding_payload),
                    cipher.encrypt(snapshot["authority"], ns),
                    cipher.encrypt(
                        snapshot["project_snapshot"], {"namespace": ns, "plane": "P1"}
                    ),
                    snapshot["cursor"],
                    previous,
                    sum(len(x["text"].encode()) for x in records),
                    binding.expires_at,
                    snapshot["checkpoint"],
                    key_ref,
                    key_version,
                    snapshot.get("retention_class", binding.retention_class),
                    checksum,
                    checkpoint_durability_mode,
                    durability_legacy,
                    durability_command_digest,
                    (
                        canonical(restored_durability_receipt).decode()
                        if restored_durability_receipt is not None
                        else None
                    ),
                    hydrate_command_digest,
                    hydrate_request_digest if hydrate_command_digest is not None else None,
                    (
                        canonical(hydrate_result).decode()
                        if hydrate_command_digest is not None
                        else None
                    ),
                    reservation_protocol_version,
                ),
            )
            for record in records:
                m = record["metadata"]
                blob = cipher.encrypt(
                    {"text": record["text"]},
                    {
                        "namespace": ns,
                        "record_id": m["record_id"],
                        "metadata_digest": record["root"],
                    },
                )
                db.execute(
                    "INSERT INTO segments VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        cycle,
                        m["sequence"],
                        m["record_id"],
                        m["plane"],
                        m["lane_id"],
                        canonical(m).decode(),
                        blob,
                        m["content_digest"],
                        m["previous_digest"],
                        record["root"],
                        m["token_estimate"],
                        m["expires_at"],
                        record["superseded"],
                        record["quarantined"],
                    ),
                )
                for word in words(record["text"]):
                    db.execute(
                        "INSERT INTO postings VALUES(?,?,?)",
                        (cycle, cipher.token(ns, word), m["sequence"]),
                    )
            for op in snapshot["operations"]:
                if op["cycle"] != cycle:
                    raise ContextFault("context_namespace_denied")
                db.execute(
                    "INSERT INTO operations VALUES(?,?,?,?,?)",
                    (
                        cycle,
                        op["operation"],
                        op["key"],
                        op["request_digest"],
                        base64.b64decode(op["result"], validate=True),
                    ),
                )
            self._event(db, cycle, "context.recovered", root=previous, fence=grant["fence"])
            return hydrate_result

    def _hydrate_replay_exact(
        self,
        db: sqlite3.Connection,
        row: sqlite3.Row,
        snapshot: dict,
        namespace: dict,
        cipher: EnvelopeCipher,
        *,
        checksum: str,
        fence: int,
        binding_digest: str,
        durability_mode: str,
        revision: int,
        reservation_protocol_version: int,
        durability_command_digest: str | None,
        durability_legacy: int,
        source_durability_receipt_digest: str | None,
        hydrate_command_digest: str | None,
        hydrate_request_digest: str,
    ) -> bool:
        records = snapshot["records"]
        if (
            row["owner"] != namespace["owner_id"]
            or row["project"] != namespace["project_id"]
            or row["request_key"] != snapshot["request_key"]
            or row["request_digest"] != snapshot["request_digest"]
            or row["profile"] != digest(self.profile)
            or row["state"] != snapshot["state"]
            or row["state"] not in {"ACTIVE", "SEALING"}
            or row["revision"] != revision
            or row["reservation_protocol_version"]
            != reservation_protocol_version
            or row["fence"] != fence
            or row["binding_digest"] != binding_digest
            or row["cursor"] != snapshot["cursor"]
            or row["root"] != snapshot["root"]
            or row["bytes"] != sum(len(item["text"].encode()) for item in records)
            or row["expires"] != json.loads(row["binding"])["expires_at"]
            or row["checkpoint"] != snapshot["checkpoint"]
            or row["key_ref"] != snapshot.get("key_ref")
            or row["key_version"] != snapshot.get("key_version")
            or row["retention_class"]
            != snapshot.get("retention_class", "ephemeral")
            or row["checkpoint_manifest_checksum"] != checksum
            or row["checkpoint_durability_mode"] != durability_mode
            or row["checkpoint_durability_legacy"] != durability_legacy
            or row["checkpoint_durability_command_digest"]
            != durability_command_digest
        ):
            return False
        if hydrate_command_digest is not None:
            if (
                row["hydrate_command_digest"] != hydrate_command_digest
                or row["hydrate_request_digest"] != hydrate_request_digest
                or row["hydrate_receipt"] is None
            ):
                return False
        if durability_legacy:
            if row["checkpoint_durability_receipt"] is not None:
                return False
        else:
            try:
                stored_envelope = json.loads(row["checkpoint_durability_receipt"])
                stored_receipt, _ = self._validated_durability_receipt(
                    stored_envelope,
                    {self.signer.key_id: self.signer.key.public_key()},
                )
            except (ContextFault, TypeError, ValueError):
                return False
            if (
                not isinstance(stored_receipt, ContextCheckpointDurabilityReceiptV2)
                or stored_receipt.source_registration_receipt_digest
                != source_durability_receipt_digest
            ):
                return False
        if (
            cipher.decrypt(row["authority"], namespace, self.profile.max_record_bytes)
            != snapshot["authority"]
            or self._snapshot(row, namespace) != snapshot["project_snapshot"]
        ):
            return False
        stored_records = db.execute(
            "SELECT * FROM segments WHERE cycle=? ORDER BY seq", (row["id"],)
        ).fetchall()
        if len(stored_records) != len(records):
            return False
        expected_postings: set[tuple[str, int]] = set()
        for stored, expected in zip(stored_records, records):
            readback = self._record(stored, namespace, cipher)
            if (
                readback["metadata"] != expected["metadata"]
                or readback["text"] != expected["text"]
                or stored["root"] != expected["root"]
                or stored["superseded"] != expected["superseded"]
                or stored["quarantined"] != expected["quarantined"]
            ):
                return False
            expected_postings.update(
                (cipher.token(namespace, word), stored["seq"])
                for word in words(expected["text"])
            )
        stored_postings = {
            (item["term"], item["seq"])
            for item in db.execute(
                "SELECT term,seq FROM postings WHERE cycle=?", (row["id"],)
            )
        }
        if stored_postings != expected_postings:
            return False
        stored_operations = [
            {
                **dict(item),
                "result": base64.b64encode(item["result"]).decode(),
            }
            for item in db.execute(
                "SELECT * FROM operations WHERE cycle=? AND operation!='checkpoint' "
                "ORDER BY operation,key",
                (row["id"],),
            )
        ]
        expected_operations = sorted(
            snapshot["operations"], key=lambda item: (item["operation"], item["key"])
        )
        return stored_operations == expected_operations

    def _promotion_entry(self, row, namespace: dict, cipher: EnvelopeCipher) -> dict | None:
        metadata = self._record(row, namespace, cipher)["metadata"]
        if not metadata["evidence_refs"] or metadata["source_class"] in {
            "model_note",
            "legacy_untrusted",
        }:
            return None
        return {
            "record_id": row["id"],
            "content_digest": metadata["content_digest"],
            "evidence_refs": metadata["evidence_refs"],
        }

    def _promotion_candidates(self, db, cycle, namespace, cipher) -> list[dict]:
        values = []
        for row in db.execute(
            "SELECT * FROM segments WHERE cycle=? AND plane='P4' AND superseded=0 "
            "AND quarantined=0 AND expires>? ORDER BY id",
            (cycle["id"], self.now()),
        ):
            entry = self._promotion_entry(row, namespace, cipher)
            if entry is not None:
                values.append(entry)
        return values

    def freeze(self, capability: dict) -> dict:
        with self.store.transaction() as db:
            cap, cycle = self._authorize(db, capability, "seal")
            if cap.role != "verifier" or cycle["state"] not in {"ACTIVE", "SEALING"}:
                raise ContextFault("context_freeze_denied")
            namespace = cap.namespace.model_dump()
            cipher = self._cipher(cycle, namespace)
            if cycle["state"] == "ACTIVE":
                db.execute(
                    "UPDATE cycles SET state='SEALING',revision=revision+1 WHERE id=?",
                    (cycle["id"],),
                )
                self._event(db, cycle["id"], "context.sealing")
            candidates = self._promotion_candidates(db, cycle, namespace, cipher)
            if len(candidates) > 1000:
                raise ContextFault("context_promotion_quota")
            return self.signer.sign(
                ContextFreezeReceiptV1(
                    context_cycle_id=cycle["id"],
                    cursor=cycle["cursor"],
                    root=cycle["root"],
                    binding_digest=cycle["binding_digest"],
                    promotion_candidates=[
                        PromotionCandidateV1.model_validate(item) for item in candidates
                    ],
                    promotion_candidates_root=digest(candidates),
                ).model_dump()
            )

    def seal(self, capability: dict, signed_proof: dict, checkpoint_receipt: dict) -> dict:
        proof = ClosureProofV1.model_validate(verify(signed_proof, self.proof_keys))
        with self._transaction(capability) as db:
            cap, cycle = self._authorize(db, capability, "seal")
            if cap.role != "verifier":
                raise ContextFault("context_role_denied")
            ns = cap.namespace.model_dump()
            cipher = self._cipher(cycle, ns)
            req_digest = digest(proof)
            replay = self._replay(
                db, cycle["id"], "seal", "terminal", req_digest, cipher
            )
            if replay:
                return replay
            if cycle["state"] not in {"ACTIVE", "SEALING"} or proof.expires_at <= self.now():
                raise ContextFault("context_not_active")
            binding = json.loads(cycle["binding"])
            if (
                proof.namespace != cap.namespace
                or proof.binding_digest != cycle["binding_digest"]
                or proof.final_root != cycle["root"]
                or proof.final_cursor != cycle["cursor"]
                or proof.repo_main_sha != binding["repo_main_sha"]
                or proof.plan_ir_digest != binding["shared_ir_digest"]
            ):
                raise ContextFault("context_proof_mismatch")
            try:
                checkpoint = ContextCheckpointReceiptV1.model_validate(
                    verify(
                        checkpoint_receipt,
                        {self.signer.key_id: self.signer.key.public_key()},
                    )
                ).model_dump()
            except (ValueError, TypeError) as exc:
                raise ContextFault("context_checkpoint_receipt_invalid") from exc
            if (
                checkpoint.get("schema_version") != "ContextCheckpointReceiptV1"
                or checkpoint["namespace"] != cap.namespace.model_dump()
                or checkpoint["cursor"] != cycle["cursor"]
                or checkpoint["segment_chain_root"] != cycle["root"]
                or checkpoint["manifest_checksum"] != cycle["checkpoint_manifest_checksum"]
                or checkpoint.get("key_ref") != cycle["key_ref"]
                or checkpoint.get("key_provider")
                != (
                    getattr(self.cycle_keys, "provider", None)
                    if cycle["key_ref"] is not None
                    else None
                )
                or checkpoint.get("key_version") != cycle["key_version"]
            ):
                raise ContextFault("context_checkpoint_stale")
            promoted = []
            for record_id in proof.accepted_record_ids:
                row = db.execute(
                    "SELECT * FROM segments WHERE cycle=? AND id=? AND plane='P4' AND superseded=0 AND quarantined=0 AND expires>?",
                    (cycle["id"], record_id, self.now()),
                ).fetchone()
                if row is None:
                    raise ContextFault("context_promotion_invalid")
                entry = self._promotion_entry(row, ns, cipher)
                if entry is None:
                    raise ContextFault("context_independent_evidence_required")
                promoted.append(entry)
            if digest(sorted(promoted, key=lambda x: x["record_id"])) != proof.promotion_set_root:
                raise ContextFault("context_promotion_root_mismatch")
            db.execute(
                "UPDATE cycles SET state='SEALING',revision=revision+1 WHERE id=?", (cycle["id"],)
            )
            self._event(db, cycle["id"], "context.sealing")
            expiry = self.now() + self.profile.retention_seconds
            result = self.signer.sign(
                {
                    "schema_version": "ContextSealReceiptV1",
                    "context_cycle_id": cycle["id"],
                    "final_state": "SEALED",
                    "binding_digest": cycle["binding_digest"],
                    "segment_chain_root": cycle["root"],
                    "checkpoint_receipt": checkpoint_receipt,
                    "proof": proof.model_dump(),
                    "promotion_set_root": proof.promotion_set_root,
                    "accepted_count": len(promoted),
                    "rejected_count": len(proof.rejected_record_ids),
                    "graph_id": binding["project_graph_id"],
                    "graph_revision": binding["graph_revision"],
                    "graph_checksum": binding["graph_checksum"],
                    "sealed_at": self.now(),
                    "expires_at": expiry,
                    "retention_class": cycle["retention_class"],
                    "retention_seconds": self.profile.retention_seconds,
                }
            )
            seal_digest = digest(result)
            db.execute(
                "UPDATE cycles SET state='SEALED',expires=?,seal_digest=?,"
                "revision=revision+1 WHERE id=?",
                (expiry, seal_digest, cycle["id"]),
            )
            self._event(
                db,
                cycle["id"],
                "context.sealed",
                seal_digest=seal_digest,
                accepted_count=len(promoted),
            )
            return self._remember(
                db, cycle["id"], "seal", "terminal", req_digest, result, cipher
            )

    def control(self, signed_command: dict) -> dict:
        command = verify(signed_command, self.authority_keys)
        if set(command) != {
            "operation",
            "context_cycle_id",
            "binding_digest",
            "expected_revision",
            "fence",
            "expires_at",
        }:
            raise ContextFault("context_control_schema")
        if command["expires_at"] <= self.now():
            raise ContextFault("context_capability_denied")
        with self.store.transaction() as db:
            row = self._cycle(db, command["context_cycle_id"])
            if (
                row["binding_digest"] != command["binding_digest"]
                or row["revision"] != command["expected_revision"]
            ):
                raise ContextFault("context_stale_state")
            if (
                row["binding"] is None
                or row["binding_digest"] is None
                or row["state"] == "RESERVED"
            ):
                raise ContextFault("context_transition_denied")
            transitions = {
                "degrade": ({"ACTIVE"}, "DEGRADED"),
                "recover": ({"DEGRADED"}, "ACTIVE"),
                "abort": ({"BOUND", "ACTIVE", "DEGRADED"}, "ABORTED"),
                "quarantine": ({"ACTIVE", "DEGRADED", "SEALING"}, "QUARANTINED"),
            }
            op = command["operation"]
            if op == "fence":
                if (
                    command["fence"] <= row["fence"]
                    or row["state"] not in {"BOUND", "ACTIVE", "DEGRADED", "SEALING"}
                ):
                    raise ContextFault("context_stale_fence")
                db.execute(
                    "UPDATE cycles SET fence=?,revision=revision+1 WHERE id=?",
                    (command["fence"], row["id"]),
                )
            elif op in transitions and row["state"] in transitions[op][0]:
                if op == "abort":
                    db.execute(
                        "UPDATE cycles SET state=?,expires=?,revision=revision+1 WHERE id=?",
                        (
                            transitions[op][1],
                            self.now() + self.profile.retention_seconds,
                            row["id"],
                        ),
                    )
                else:
                    db.execute(
                        "UPDATE cycles SET state=?,revision=revision+1 WHERE id=?",
                        (transitions[op][1], row["id"]),
                    )
            else:
                raise ContextFault("context_transition_denied")
            self._event(db, row["id"], "context." + op)
            return self.signer.sign(
                {"operation": op, "context_cycle_id": row["id"], "revision": row["revision"] + 1}
            )

    def status(self, capability: dict, after: int = 0) -> dict:
        if type(after) is not int or after < 0:
            raise ContextFault("context_stale_cursor")
        self._space()
        reach_claim_enabled = self._hot_reach_claim_enabled
        with self.store.connection() as db:
            _, row = self._authorize(db, capability, "status")
            events = [
                json.loads(x[0])
                for x in db.execute(
                    "SELECT body FROM events WHERE cycle=? AND cursor>? ORDER BY cursor LIMIT 100",
                    (row["id"], after),
                )
            ]
            return self.signer.sign(
                {
                    "schema_version": "ContextCycleV1",
                    "context_cycle_id": row["id"],
                    "state": row["state"],
                    "revision": row["revision"],
                    "cursor": row["cursor"],
                    "root": row["root"],
                    "fence": row["fence"],
                    "indexed_bytes": row["bytes"],
                    "records": row["cursor"],
                    "configured_max_records": self.profile.max_records,
                    "configured_max_indexed_bytes": self.profile.max_indexed_bytes,
                    "configured_reachable_tokens_upper_bound": (
                        self.profile.max_indexed_bytes // 4
                    ),
                    "checkpoint_number": row["checkpoint"],
                    "expires_at": row["expires"],
                    "retention_class": row["retention_class"],
                    "retention_hold": bool(row["hold"]),
                    "retention_hold_reason": row["hold_reason"],
                    "cleanup_state": row["cleanup_state"],
                    "deletion_receipt": (
                        json.loads(row["deletion_receipt"])
                        if row["deletion_receipt"] is not None
                        else None
                    ),
                    "estimated_reachable_tokens": self.scale_qualification.reachable_tokens,
                    "reach_claim_enabled": reach_claim_enabled,
                    "benchmark_receipt_digest": self.scale_qualification.receipt_digest,
                    "benchmark_failures": list(self.scale_qualification.failures),
                    "native_model_window": None,
                    "retrieved_tokens": None,
                    "injected_tokens": None,
                    "resident_process_observed_peak_bytes": (
                        self.scale_qualification.resident_peak_bytes
                    ),
                    "events": events,
                }
            )

    def retention(self, capability: dict, signed_command: dict) -> dict:
        return self.retention_manager.apply(capability, signed_command)

    def expire(self, capability: dict) -> dict:
        # Kept only as a rolling wire boundary. Every deletion, including an
        # unkeyed legacy checkpoint, must pass through signed retention CAS and
        # independently signed remote-object deletion evidence.
        raise ContextFault("context_signed_retention_required")
