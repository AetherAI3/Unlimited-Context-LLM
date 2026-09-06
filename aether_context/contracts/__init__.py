"""Closed SC-CONTEXT-01 wire contracts. Requires the optional hosted extra."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Ident = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,199}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PrefixedDigest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Sha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
# The copy-heavy JSON/base64 IPC path is deliberately narrower than the V1
# checkpoint profile. A canonical JSON snapshot may be up to 16 MB, but zlib's
# entropy ceiling for JSON-safe text makes the largest emitted encrypted pack
# about 13 MB. Advertising that reachable pack boundary keeps the measured
# near-limit gate honest until the transport is streamed.
LIVE_IPC_MAX_CHECKPOINT_BYTES = 13_000_000
HOSTED_CONTEXT_CAPABILITIES = (
    "reserve_v2",
    "bind_v2",
    "checkpoint_durability",
    "release_reservation",
    "abort_reservation",
    "expire_aborted_reservation",
    "expire_reservation",
    "reserve_v3",
    "bind_v3",
    "hydrate_v2",
    "deletion_v2",
)


def canonical(value: object) -> bytes:
    """Canonical JSON subset: UTF-8, sorted keys, integers, no floats or NaN.

    Wire timestamps are integer Unix seconds. This is deliberately a smaller
    domain than arbitrary JSON so JS and Python produce the same digest.
    """
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")

    def validate(item: object) -> None:
        if item is None or isinstance(item, (bool, str)):
            return
        if isinstance(item, int) and abs(item) <= 9007199254740991:
            return
        if isinstance(item, list):
            for child in item:
                validate(child)
            return
        if isinstance(item, dict) and all(isinstance(k, str) and k.isascii() for k in item):
            for child in item.values():
                validate(child)
            return
        raise ValueError("canonical JSON requires safe integers and string keys")

    validate(value)
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class NamespaceV1(Closed):
    owner_id: Ident
    project_id: Ident
    objective_id: Ident
    context_cycle_id: Ident


class ContextRequestV1(Closed):
    schema_version: Literal["ContextRequestV1"] = "ContextRequestV1"
    mode: Literal["required", "preferred", "disabled"] = "required"
    pool_gb: int = Field(default=1, ge=1, le=32)
    captain_mode: Literal["durable_handoff", "interactive_checkpoint", "interactive_required"] = (
        "durable_handoff"
    )
    retention_class: Literal["ephemeral", "audit"] = "ephemeral"
    resume_context_cycle_id: Ident | None = None
    promotion_mode: Literal["verified_only", "disabled"] = "verified_only"


class SignedV1(Closed):
    payload: dict
    key_id: Ident
    signature: str = Field(min_length=64, max_length=128)


class ContextReservationReceiptV1(Closed):
    """Rolling-compatible service evidence for one exact reservation."""

    schema_version: Literal["ContextReservationReceiptV1"] = "ContextReservationReceiptV1"
    context_cycle_id: Ident
    context_bucket_id: Ident
    request_digest: Digest
    profile_digest: Digest


class ContextReservationCommandV2(Closed):
    """Revision-aware reservation and explicit post-release revival."""

    schema_version: Literal["ContextReservationCommandV2"] = (
        "ContextReservationCommandV2"
    )
    operation: Literal["reserve"] = "reserve"
    owner_id: Ident
    project_id: Ident
    idempotency_key: Ident
    request_digest: Digest
    profile_digest: Digest
    expires_at: int = Field(ge=0)
    released_revision: int | None = Field(default=None, ge=2)


class ContextReservationReceiptV2(Closed):
    schema_version: Literal["ContextReservationReceiptV2"] = "ContextReservationReceiptV2"
    context_cycle_id: Ident
    context_bucket_id: Ident
    request_digest: Digest
    profile_digest: Digest
    revision: int = Field(ge=1)


class ContextReservationReceiptV3(Closed):
    schema_version: Literal["ContextReservationReceiptV3"] = "ContextReservationReceiptV3"
    context_cycle_id: Ident
    context_bucket_id: Ident
    request_digest: Digest
    profile_digest: Digest
    revision: int = Field(ge=3)
    prior_expiry_receipt_digest: Digest
    prior_expiry_revision: int = Field(ge=2)
    invocation_no_dispatch_receipt_digest: Digest
    revival_command_digest: Digest

    @model_validator(mode="after")
    def exact_revival_revision(self) -> "ContextReservationReceiptV3":
        if self.revision != self.prior_expiry_revision + 1:
            raise ValueError("reservation revival revision must follow expiry")
        return self


class ContextReservationCommandV3(Closed):
    """Reservation revival authorized by an exact no-dispatch expiry receipt."""

    schema_version: Literal["ContextReservationCommandV3"] = (
        "ContextReservationCommandV3"
    )
    operation: Literal["reserve"] = "reserve"
    owner_id: Ident
    project_id: Ident
    idempotency_key: Ident
    request_digest: Digest
    profile_digest: Digest
    expires_at: int = Field(ge=0)
    expired_reservation_receipt: SignedV1


class ContextReservationReleaseCommandV1(Closed):
    """Gateway proof that authorization failed before objective dispatch."""

    schema_version: Literal["ContextReservationReleaseCommandV1"] = (
        "ContextReservationReleaseCommandV1"
    )
    operation: Literal["release_reservation"] = "release_reservation"
    context_cycle_id: Ident
    owner_id: Ident
    project_id: Ident
    idempotency_key: Ident
    request_digest: Digest
    profile_digest: Digest
    expected_revision: int = Field(ge=1)
    reason_code: Literal["pre_dispatch_authorization_failed"]
    dispatch_started: Literal[False]
    authorization_failure_receipt_digest: Digest
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def short_lived(self) -> "ContextReservationReleaseCommandV1":
        if self.expires_at <= self.issued_at or self.expires_at - self.issued_at > 900:
            raise ValueError("reservation release authority must be short-lived")
        return self


class ContextReservationReleaseReceiptV1(Closed):
    schema_version: Literal["ContextReservationReleaseReceiptV1"] = (
        "ContextReservationReleaseReceiptV1"
    )
    operation: Literal["release_reservation"] = "release_reservation"
    context_cycle_id: Ident
    owner_id: Ident
    project_id: Ident
    idempotency_key: Ident
    request_digest: Digest
    profile_digest: Digest
    authorization_failure_receipt_digest: Digest
    release_command_digest: Digest
    final_state: Literal["EXPIRED"] = "EXPIRED"
    released: Literal[True] = True
    revision: int = Field(ge=2)


class ContextReservationAbortCommandV1(Closed):
    """Gateway proof that a dispatched objective stalled before context bind."""

    schema_version: Literal["ContextReservationAbortCommandV1"] = (
        "ContextReservationAbortCommandV1"
    )
    operation: Literal["abort_reservation"] = "abort_reservation"
    context_cycle_id: Ident
    owner_id: Ident
    project_id: Ident
    objective_id: Ident
    idempotency_key: Ident
    request_digest: Digest
    profile_digest: Digest
    expected_revision: int = Field(ge=1)
    reason_code: Literal["timeout", "cancel", "reconciliation_failed"]
    dispatch_started: Literal[True]
    objective_dispatch_receipt_digest: Digest
    reservation_receipt: SignedV1
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def short_lived(self) -> "ContextReservationAbortCommandV1":
        if self.expires_at <= self.issued_at or self.expires_at - self.issued_at > 900:
            raise ValueError("reservation abort authority must be short-lived")
        return self


class ContextReservationAbortReceiptV1(Closed):
    schema_version: Literal["ContextReservationAbortReceiptV1"] = (
        "ContextReservationAbortReceiptV1"
    )
    operation: Literal["abort_reservation"] = "abort_reservation"
    context_cycle_id: Ident
    owner_id: Ident
    project_id: Ident
    objective_id: Ident
    idempotency_key: Ident
    request_digest: Digest
    profile_digest: Digest
    reason_code: Literal["timeout", "cancel", "reconciliation_failed"]
    objective_dispatch_receipt_digest: Digest
    reservation_receipt_digest: Digest
    abort_command_digest: Digest
    final_state: Literal["ABORTED"] = "ABORTED"
    aborted: Literal[True] = True
    aborted_at: int = Field(ge=0)
    retention_expires_at: int = Field(ge=0)
    revision: int = Field(ge=2)

    @model_validator(mode="after")
    def retention_follows_abort(self) -> "ContextReservationAbortReceiptV1":
        if self.retention_expires_at <= self.aborted_at:
            raise ValueError("reservation abort retention expiry must follow abort")
        return self


class ContextReservationAbortExpiryCommandV1(Closed):
    """Authority to clean one terminal, never-bound reservation after retention."""

    schema_version: Literal["ContextReservationAbortExpiryCommandV1"] = (
        "ContextReservationAbortExpiryCommandV1"
    )
    operation: Literal["expire_aborted_reservation"] = (
        "expire_aborted_reservation"
    )
    namespace: NamespaceV1
    idempotency_key: Ident
    expected_revision: int = Field(ge=2)
    reason_code: Literal["policy_expiry", "operator_expiry"]
    abort_receipt: SignedV1
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def short_lived(self) -> "ContextReservationAbortExpiryCommandV1":
        if self.expires_at <= self.issued_at or self.expires_at - self.issued_at > 900:
            raise ValueError("reservation abort expiry authority must be short-lived")
        return self


class ContextInvocationNoDispatchSnapshotV1(Closed):
    """Exact atomically locked Cloud dispatch-fence projection."""

    owner_user_id: Ident
    objective_id: Ident
    idempotency_key: Ident
    request_digest: PrefixedDigest
    context_cycle_id: Ident
    reservation_receipt_digest: Digest
    reservation_revision: int = Field(ge=1)
    state: Literal["closed_without_dispatch"]
    revision: int = Field(ge=1)
    canonical_result_state: Literal[
        "absent", "planned", "blocked_without_authority"
    ]
    canonical_result_digest: Digest | None = None

    @model_validator(mode="after")
    def result_is_resolvable(self) -> "ContextInvocationNoDispatchSnapshotV1":
        if (self.canonical_result_state == "absent") != (
            self.canonical_result_digest is None
        ):
            raise ValueError("canonical result state and digest mismatch")
        return self


class ContextInvocationNoDispatchReceiptV1(Closed):
    """Signed exact snapshot of a terminal Cloud dispatch fence."""

    schema_version: Literal["ContextInvocationNoDispatchReceiptV1"] = (
        "ContextInvocationNoDispatchReceiptV1"
    )
    operation: Literal["close_without_dispatch"] = "close_without_dispatch"
    ledger_snapshot: ContextInvocationNoDispatchSnapshotV1
    ledger_snapshot_digest: Digest
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def short_lived(self) -> "ContextInvocationNoDispatchReceiptV1":
        if self.expires_at <= self.issued_at or self.expires_at - self.issued_at > 900:
            raise ValueError("no-dispatch closure proof must be short-lived")
        if digest(self.ledger_snapshot) != self.ledger_snapshot_digest:
            raise ValueError("ledger snapshot digest mismatch")
        return self


class ContextReservationExpiryCommandV1(Closed):
    """Expire a clean reservation after canonical dispatch has been fenced out."""

    schema_version: Literal["ContextReservationExpiryCommandV1"] = (
        "ContextReservationExpiryCommandV1"
    )
    operation: Literal["expire_reservation"] = "expire_reservation"
    context_cycle_id: Ident
    context_bucket_id: Ident
    objective_id: Ident
    owner_id: Ident
    project_id: Ident
    idempotency_key: Ident
    request_digest: Digest
    profile_digest: Digest
    expected_revision: int = Field(ge=1)
    reason_code: Literal["reservation_timeout", "client_cancel", "reconciliation_failed"]
    reservation_receipt: SignedV1
    invocation_no_dispatch_receipt: SignedV1
    invocation_no_dispatch_receipt_digest: Digest
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def closed_short_lived_proof(self) -> "ContextReservationExpiryCommandV1":
        if self.expires_at <= self.issued_at or self.expires_at - self.issued_at > 900:
            raise ValueError("reservation expiry authority must be short-lived")
        if digest(self.invocation_no_dispatch_receipt) != self.invocation_no_dispatch_receipt_digest:
            raise ValueError("no-dispatch receipt digest mismatch")
        return self


class ContextReservationExpiryReceiptV1(Closed):
    schema_version: Literal["ContextReservationExpiryReceiptV1"] = (
        "ContextReservationExpiryReceiptV1"
    )
    operation: Literal["expire_reservation"] = "expire_reservation"
    context_cycle_id: Ident
    context_bucket_id: Ident
    objective_id: Ident
    owner_id: Ident
    project_id: Ident
    idempotency_key: Ident
    request_digest: Digest
    profile_digest: Digest
    reason_code: Literal["reservation_timeout", "client_cancel", "reconciliation_failed"]
    reservation_receipt_digest: Digest
    invocation_no_dispatch_receipt_digest: Digest
    expiry_command_digest: Digest
    final_state: Literal["EXPIRED"] = "EXPIRED"
    expired: Literal[True] = True
    revivable: Literal[True] = True
    expired_at: int = Field(ge=0)
    revision: int = Field(ge=2)


class ContextCheckpointDurabilityCommandV1(Closed):
    """Immutable per-cycle choice between local-only and remote checkpoints."""

    schema_version: Literal["ContextCheckpointDurabilityCommandV1"] = (
        "ContextCheckpointDurabilityCommandV1"
    )
    operation: Literal["register_checkpoint_durability"] = (
        "register_checkpoint_durability"
    )
    namespace: NamespaceV1
    idempotency_key: Ident
    expected_revision: int = Field(ge=1)
    mode: Literal["local_ephemeral", "remote_registered"]
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def short_lived(self) -> "ContextCheckpointDurabilityCommandV1":
        if self.expires_at <= self.issued_at or self.expires_at - self.issued_at > 900:
            raise ValueError("checkpoint durability authority must be short-lived")
        return self


class ContextCheckpointDurabilityReceiptV1(Closed):
    schema_version: Literal["ContextCheckpointDurabilityReceiptV1"] = (
        "ContextCheckpointDurabilityReceiptV1"
    )
    operation: Literal["register_checkpoint_durability"] = (
        "register_checkpoint_durability"
    )
    namespace: NamespaceV1
    idempotency_key: Ident
    mode: Literal["local_ephemeral", "remote_registered"]
    command_digest: Digest
    effective_at: int = Field(ge=0)
    revision: int = Field(ge=2)


class ContextProfileV1(Closed):
    schema_version: Literal["ContextProfileV1"] = "ContextProfileV1"
    name: Literal["bounded-canary/v1", "hosted-v1"] = "bounded-canary/v1"
    index_implementation: Literal["sqlite-hmac-inverted/v1"] = "sqlite-hmac-inverted/v1"
    index_version: str = Field(min_length=3, max_length=40)
    embedding_version: Literal["none-lexical/v1"] = "none-lexical/v1"
    storage_version: Literal["sqlite-wal-aes256gcm/v2"] = "sqlite-wal-aes256gcm/v2"
    max_records: int = Field(default=10000, ge=1, le=1000000)
    max_indexed_bytes: int = Field(default=64_000_000, ge=1024, le=32_000_000_000)
    max_record_bytes: int = Field(default=32768, ge=256, le=262144)
    max_capsule_tokens: int = Field(default=4096, ge=128, le=65536)
    max_candidates: int = Field(default=256, ge=1, le=4096)
    resident_cache_bytes: int = Field(default=8_388_608, ge=1_048_576, le=268_435_456)
    disk_free_floor: int = Field(default=268_435_456, ge=0)
    retention_seconds: int = Field(default=86400, ge=60, le=604800)
    max_checkpoint_bytes: int = Field(default=64_000_000, ge=1024, le=128_000_000)
    max_concurrent_cycles: int = Field(default=4, ge=1, le=64)
    max_snapshot_bytes: int = Field(default=8_000_000, ge=1024, le=16_000_000)
    max_operations_per_cycle: int = Field(default=20000, ge=16, le=2000000)
    max_calls_per_minute: int = Field(default=600, ge=10, le=6000)
    search_slo_ms: int = Field(default=500, ge=1, le=10000)
    rebuild_slo_seconds: int = Field(default=300, ge=1, le=86400)
    benchmark_receipt: Digest | None = None

    @model_validator(mode="after")
    def production_evidence(self) -> "ContextProfileV1":
        if self.name == "hosted-v1" and self.benchmark_receipt is None:
            raise ValueError("hosted-v1 requires a measured benchmark receipt")
        return self


class ContextBindingV1(Closed):
    schema_version: Literal["ContextBindingV1"] = "ContextBindingV1"
    namespace: NamespaceV1
    context_bucket_id: Ident
    repository_id: Ident
    repo_main_sha: Sha
    project_graph_id: Ident
    graph_revision: int = Field(ge=1)
    graph_checksum: Digest
    plan_digest: Digest
    execution_profile_digest: Digest
    shared_ir_digest: Digest
    authorization_receipt_ref: Ident
    authorization_digest: Digest
    policy_digest: Digest
    redaction_digest: Digest
    profile_digest: Digest
    captain_binding_id: Ident
    retention_class: Literal["ephemeral", "audit"] = "ephemeral"
    created_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)


class CapabilityV1(Closed):
    schema_version: Literal["ContextCapabilityV1"] = "ContextCapabilityV1"
    namespace: NamespaceV1
    principal_id: Ident
    lane_id: Ident
    task_id: Ident
    role: Literal["worker", "reviewer", "coordinator", "verifier", "captain", "durability"]
    operations: list[
        Literal[
            "append",
            "retrieve",
            "checkpoint",
            "seal",
            "status",
            "hydrate",
            "expire",
            "retention",
        ]
    ] = Field(min_length=1, max_length=8)
    source_class: Literal[
        "user_intent",
        "project_memory_verified",
        "repository_verified",
        "tool_observation",
        "ci_verified",
        "worker_summary",
        "model_note",
        "legacy_untrusted",
    ]
    binding_digest: Digest
    policy_digest: Digest
    profile_digest: Digest
    fence: int = Field(ge=1)
    expires_at: int = Field(ge=0)
    max_bytes: int = Field(ge=1, le=262144)
    max_tokens: int = Field(ge=1, le=65536)


class ContextCheckpointDurabilityReceiptV2(ContextCheckpointDurabilityReceiptV1):
    """Replacement-host attestation over exact source registration evidence."""

    schema_version: Literal["ContextCheckpointDurabilityReceiptV2"] = "ContextCheckpointDurabilityReceiptV2"  # type: ignore[assignment]
    source_registration_receipt: SignedV1
    source_registration_receipt_digest: Digest
    reattested_at: int = Field(ge=0)

    @model_validator(mode="after")
    def source_receipt_digest_matches(self) -> "ContextCheckpointDurabilityReceiptV2":
        if digest(self.source_registration_receipt) != self.source_registration_receipt_digest:
            raise ValueError("source durability registration digest mismatch")
        try:
            source_claim = ContextCheckpointDurabilityReceiptV1.model_validate(
                self.source_registration_receipt.payload
            )
        except ValueError as exc:
            raise ValueError(
                "source durability registration must be an exact V1 receipt"
            ) from exc
        source = source_claim.model_dump(mode="json")
        if source != self.source_registration_receipt.payload:
            raise ValueError("source durability registration must be closed")
        expected = {
            "namespace": self.namespace.model_dump(mode="json"),
            "idempotency_key": self.idempotency_key,
            "mode": self.mode,
            "command_digest": self.command_digest,
            "effective_at": self.effective_at,
            "revision": self.revision,
        }
        if any(source.get(field) != value for field, value in expected.items()):
            raise ValueError("source durability registration claim mismatch")
        return self


class ContextBindingCommandV2(Closed):
    """Revision-CAS wrapper that keeps the persisted binding itself at V1."""

    schema_version: Literal["ContextBindingCommandV2"] = "ContextBindingCommandV2"
    operation: Literal["bind"] = "bind"
    binding: ContextBindingV1
    reservation_receipt: SignedV1
    expected_reservation_revision: int = Field(ge=1)


class ContextBindingCommandV3(Closed):
    """Bind an expiry-revived reservation to its exact V3 lineage."""

    schema_version: Literal["ContextBindingCommandV3"] = "ContextBindingCommandV3"
    operation: Literal["bind"] = "bind"
    binding: ContextBindingV1
    reservation_receipt: SignedV1
    expected_reservation_revision: int = Field(ge=3)


class PromotionCandidateV1(Closed):
    record_id: Ident
    content_digest: Digest
    evidence_refs: list[Digest] = Field(min_length=1, max_length=32)


class ContextCheckpointReceiptV1(Closed):
    schema_version: Literal["ContextCheckpointReceiptV1"] = "ContextCheckpointReceiptV1"
    namespace: NamespaceV1
    checkpoint_number: int = Field(ge=1)
    cursor: int = Field(ge=0)
    segment_chain_root: Digest
    manifest_checksum: Digest
    durability: Literal["local"]
    bytes: int = Field(ge=1)
    records: int = Field(ge=0)
    policy_digest: Digest
    key_ref: Ident | None = None
    key_provider: str | None = Field(default=None, min_length=1, max_length=100)
    key_version: Ident | None = None

    @model_validator(mode="after")
    def key_reference_pair(self) -> "ContextCheckpointReceiptV1":
        if len({item is None for item in (self.key_ref, self.key_provider, self.key_version)}) != 1:
            raise ValueError("checkpoint key reference, provider and version must be paired")
        return self


class ContextHydrateCommandV2(Closed):
    """Short-lived recovery authority for one checkpoint and runtime source."""

    schema_version: Literal["ContextHydrateCommandV2"] = "ContextHydrateCommandV2"
    operation: Literal["hydrate"] = "hydrate"
    namespace: NamespaceV1
    manifest_checksum: Digest
    key_ref: Ident | None = None
    key_provider: str | None = Field(default=None, min_length=1, max_length=100)
    key_version: Ident | None = None
    fence: int = Field(ge=1)
    expected_source_revision: Sha
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def bounded_recovery(self) -> "ContextHydrateCommandV2":
        if len({item is None for item in (self.key_ref, self.key_provider, self.key_version)}) != 1:
            raise ValueError("hydrate key reference, provider and version must be paired")
        if self.expires_at <= self.issued_at or self.expires_at - self.issued_at > 900:
            raise ValueError("hydrate authority must be short-lived")
        return self


class ContextHealthV1(Closed):
    """Returned daemon health identity and measured readiness evidence."""

    schema_version: Literal["ContextHealthV1"] = "ContextHealthV1"
    ready: bool
    profile_digest: Digest
    index_implementation: str = Field(min_length=1, max_length=100)
    index_version: str = Field(min_length=1, max_length=100)
    disk_free_bytes: int = Field(ge=0)
    resident_cache_limit_bytes: int = Field(ge=0)
    resident_process_limit_bytes: int = Field(ge=0)
    resident_process_observed_peak_bytes: int | None = Field(default=None, ge=0)
    configured_max_records: int = Field(ge=1)
    configured_max_indexed_bytes: int = Field(ge=1)
    configured_reachable_tokens_upper_bound: int = Field(ge=0)
    measured_indexed_records: int | None = Field(default=None, ge=0)
    measured_indexed_bytes: int | None = Field(default=None, ge=0)
    estimated_reachable_tokens: int | None = Field(default=None, ge=0)
    reach_claim_enabled: bool
    benchmark_failures: list[str]
    benchmark_source_revision: Sha | None = None
    benchmark_receipt_digest: Digest | None = None
    benchmark_recall_dataset_digest: Digest | None = None
    runtime_build_ready: bool
    runtime_build_failures: list[str]
    runtime_build_manifest_digest: Digest | None = None
    runtime_source_revision: Sha | None = None
    runtime_data_plane_case_generator_digest: Digest | None = None
    cycle_key_provider: str | None = Field(default=None, min_length=1, max_length=100)
    managed_cycle_keys_ready: bool
    managed_cycle_key_failures: list[str]
    managed_key_provider_receipt_digests: dict[str, Digest]
    managed_key_provider_receipt_digest: Digest | None = None


class ContextHealthV2(ContextHealthV1):
    """Health V1 plus the measured executable runtime identity."""

    schema_version: Literal["ContextHealthV2"] = "ContextHealthV2"  # type: ignore[assignment]
    runtime_package_tree_digest: Digest | None = None
    runtime_locked_environment_digest: Digest | None = None
    runtime_python_implementation: str | None = Field(default=None, min_length=1, max_length=40)
    runtime_python_version: str | None = Field(default=None, min_length=1, max_length=40)
    runtime_python_abi: str | None = Field(default=None, min_length=1, max_length=200)
    runtime_python_executable_digest: Digest | None = None
    runtime_dependency_versions: dict[str, str]
    runtime_dependency_artifact_digests: dict[str, Digest]
    runtime_sqlite_version: str | None = Field(default=None, min_length=1, max_length=40)
    runtime_package_import_root_digest: Digest | None = None
    runtime_bytecode_policy: Literal["source-only/no-bytecode/v1"] | None = None


class ContextHealthV3(ContextHealthV2):
    """Health V2 plus the guarded complete dependency import closure."""

    schema_version: Literal["ContextHealthV3"] = "ContextHealthV3"  # type: ignore[assignment]
    runtime_dependency_import_closure: list[str] = Field(default_factory=list, max_length=32)
    runtime_bytecode_policy: Literal["preimport-guard/source-only/v2"] | None = None  # type: ignore[assignment]
    context_protocol_version: Literal[3] = 3
    capabilities: list[
        Literal[
            "reserve_v2",
            "bind_v2",
            "checkpoint_durability",
            "release_reservation",
            "abort_reservation",
            "expire_aborted_reservation",
            "expire_reservation",
            "reserve_v3",
            "bind_v3",
            "hydrate_v2",
            "deletion_v2",
        ]
    ]
    live_ipc_checkpoint_bytes: int = Field(ge=1024, le=LIVE_IPC_MAX_CHECKPOINT_BYTES)

    @model_validator(mode="after")
    def exact_capabilities(self) -> "ContextHealthV3":
        if self.capabilities != list(HOSTED_CONTEXT_CAPABILITIES):
            raise ValueError("hosted context capabilities must be exact and ordered")
        return self


class ContextHydrateReceiptV1(Closed):
    """Host-replacement evidence bound to one exact checkpoint replay."""

    schema_version: Literal["ContextHydrateReceiptV1"] = "ContextHydrateReceiptV1"
    operation: Literal["hydrate"] = "hydrate"
    namespace: NamespaceV1
    context_cycle_id: Ident
    manifest_checksum: Digest
    checkpoint_number: int = Field(ge=1)
    cursor: int = Field(ge=0)
    root: Digest
    key_ref: Ident | None = None
    key_provider: str | None = Field(default=None, min_length=1, max_length=100)
    key_version: Ident | None = None
    replay_digest: Digest
    fence: int = Field(ge=1)

    @model_validator(mode="after")
    def key_reference_pair(self) -> "ContextHydrateReceiptV1":
        if len({item is None for item in (self.key_ref, self.key_provider, self.key_version)}) != 1:
            raise ValueError("hydrate key reference, provider and version must be paired")
        if self.context_cycle_id != self.namespace.context_cycle_id:
            raise ValueError("hydrate cycle does not match namespace")
        return self


class ContextFreezeReceiptV1(Closed):
    schema_version: Literal["ContextFreezeReceiptV1"] = "ContextFreezeReceiptV1"
    context_cycle_id: Ident
    cursor: int = Field(ge=0)
    root: Digest
    binding_digest: Digest
    promotion_candidates: list[PromotionCandidateV1] = Field(
        default_factory=list, max_length=1000
    )
    promotion_candidates_root: Digest

    @model_validator(mode="after")
    def candidate_root(self) -> "ContextFreezeReceiptV1":
        if digest([item.model_dump() for item in self.promotion_candidates]) != (
            self.promotion_candidates_root
        ):
            raise ValueError("promotion candidate root mismatch")
        return self


class AppendRequestV1(Closed):
    idempotency_key: Ident
    text: str = Field(min_length=1, max_length=262144)
    plane: Literal["P2", "P3", "P4"] = "P3"
    content_type: Literal[
        "summary", "decision", "artifact_ref", "tool_evidence", "candidate_fact"
    ] = "summary"
    action_ref: Ident
    evidence_refs: list[Digest] = Field(default_factory=list, max_length=32)
    supersedes: Ident | None = None


class RetrieveRequestV1(Closed):
    turn_id: Ident
    query: str = Field(max_length=8192)
    native_window: int = Field(ge=256, le=10000000)
    remaining_prompt_budget: int = Field(ge=0, le=10000000)
    used_prompt_tokens: int = Field(default=0, ge=0, le=10000000)
    event_cursor: int | None = Field(default=None, ge=0)


class RemoteDeletionReceiptV1(Closed):
    """Independent object-store evidence for the last committed checkpoint."""

    schema_version: Literal["ContextRemoteDeletionReceiptV1"] = (
        "ContextRemoteDeletionReceiptV1"
    )
    namespace: NamespaceV1
    manifest_checksum: Digest
    object_ref_digest: Digest
    deleted: Literal[True]
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def short_lived(self) -> "RemoteDeletionReceiptV1":
        if self.expires_at <= self.issued_at or self.expires_at - self.issued_at > 900:
            raise ValueError("remote deletion evidence must be short-lived")
        return self


class RetentionCommandV1(Closed):
    """Short-lived authority for an idempotent hold, release, or expiry."""

    schema_version: Literal["ContextRetentionCommandV1"] = "ContextRetentionCommandV1"
    operation: Literal["hold", "release", "expire"]
    namespace: NamespaceV1
    idempotency_key: Ident
    expected_revision: int = Field(ge=1)
    reason_code: Literal[
        "audit_hold",
        "legal_hold",
        "incident_hold",
        "hold_released",
        "policy_expiry",
        "operator_expiry",
    ]
    remote_deletion_receipt: SignedV1 | None = None
    checkpoint_object_ref_digest: Digest | None = None
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def operation_reason(self) -> "RetentionCommandV1":
        allowed = {
            "hold": {"audit_hold", "legal_hold", "incident_hold"},
            "release": {"hold_released"},
            "expire": {"policy_expiry", "operator_expiry"},
        }
        if self.reason_code not in allowed[self.operation]:
            raise ValueError("retention reason does not match operation")
        if self.operation == "expire":
            if (self.remote_deletion_receipt is None) != (
                self.checkpoint_object_ref_digest is None
            ):
                raise ValueError("remote deletion receipt and object reference must be paired")
        elif (
            self.remote_deletion_receipt is not None
            or self.checkpoint_object_ref_digest is not None
        ):
            raise ValueError("remote deletion evidence applies only to expiry")
        if self.expires_at <= self.issued_at or self.expires_at - self.issued_at > 900:
            raise ValueError("retention authority must be short-lived")
        return self


class ManagedKeyProviderReceiptV1(Closed):
    """Independent qualification for a shared managed key provider."""

    schema_version: Literal["ContextManagedKeyProviderReceiptV1"] = (
        "ContextManagedKeyProviderReceiptV1"
    )
    provider: str = Field(min_length=1, max_length=100)
    key_version: Ident
    shared_hydrate_supported: Literal[True]
    verified_destruction_supported: Literal[True]
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def bounded_qualification(self) -> "ManagedKeyProviderReceiptV1":
        if self.expires_at <= self.issued_at or self.expires_at - self.issued_at > 604800:
            raise ValueError("managed key qualification may be valid for at most seven days")
        return self


class KeyDestructionReceiptV1(Closed):
    """Independent managed-KMS evidence that an exact cycle key was destroyed."""

    schema_version: Literal["ContextKeyDestructionReceiptV1"] = (
        "ContextKeyDestructionReceiptV1"
    )
    provider: str = Field(min_length=1, max_length=100)
    key_ref: Ident
    namespace_digest: Digest
    key_version: Ident
    destroyed: Literal[True]
    destroyed_at: int = Field(ge=0)
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def short_lived(self) -> "KeyDestructionReceiptV1":
        if not self.destroyed_at <= self.issued_at < self.expires_at:
            raise ValueError("managed key destruction time ordering is invalid")
        if self.expires_at - self.issued_at > 900:
            raise ValueError("managed key destruction evidence must be short-lived")
        return self


class ContextRetentionReceiptV1(Closed):
    schema_version: Literal["ContextRetentionReceiptV1"] = "ContextRetentionReceiptV1"
    operation: Literal["hold", "release"]
    context_cycle_id: Ident
    reason_code: Literal[
        "audit_hold", "legal_hold", "incident_hold", "hold_released"
    ]
    effective_at: int = Field(ge=0)
    revision: int = Field(ge=2)

    @model_validator(mode="after")
    def operation_reason(self) -> "ContextRetentionReceiptV1":
        if self.operation == "release" and self.reason_code != "hold_released":
            raise ValueError("release receipt reason mismatch")
        if self.operation == "hold" and self.reason_code == "hold_released":
            raise ValueError("hold receipt reason mismatch")
        return self


class ContextHydrateReceiptV2(ContextHydrateReceiptV1):
    schema_version: Literal["ContextHydrateReceiptV2"] = "ContextHydrateReceiptV2"  # type: ignore[assignment]
    hydrate_command_digest: Digest
    expected_source_revision: Sha
    checkpoint_durability_registration_receipt: SignedV1 | None = None
    checkpoint_durability_registration_receipt_digest: Digest | None = None

    @model_validator(mode="after")
    def durability_registration_is_resolvable(self) -> "ContextHydrateReceiptV2":
        if (
            self.checkpoint_durability_registration_receipt is None
        ) != (self.checkpoint_durability_registration_receipt_digest is None):
            raise ValueError("hydrate durability receipt and digest must be paired")
        if (
            self.checkpoint_durability_registration_receipt is not None
            and digest(self.checkpoint_durability_registration_receipt)
            != self.checkpoint_durability_registration_receipt_digest
        ):
            raise ValueError("hydrate durability receipt digest mismatch")
        return self


class ContextDeletionReceiptV1(Closed):
    """Closed rolling deletion evidence emitted for legacy durability rows."""

    schema_version: Literal["ContextDeletionReceiptV1"] = "ContextDeletionReceiptV1"
    context_cycle_id: Ident
    segment_chain_root: Digest
    context_seal_digest: Digest | None = None
    retention_class: Literal["ephemeral", "audit"]
    retention_expired_at: int = Field(ge=0)
    deleted_at: int = Field(ge=0)
    reason_code: Literal["policy_expiry", "operator_expiry"]
    retention_operation_key: Ident
    retention_request_digest: Digest
    logical_deletion: Literal[True]
    cryptographic_erasure: bool
    local_key_destruction: dict | None = None
    key_destruction_receipt: SignedV1 | None = None
    key_destruction_receipt_digest: Digest | None = None
    managed_key_provider_receipt_digest: Digest | None = None
    remote_deletion_receipt_digest: Digest | None = None
    checkpoint_manifest_checksum: Digest | None = None
    checkpoint_object_ref_digest: Digest | None = None

    @model_validator(mode="after")
    def key_erasure_evidence(self) -> "ContextDeletionReceiptV1":
        managed_parts = (
            self.key_destruction_receipt,
            self.key_destruction_receipt_digest,
            self.managed_key_provider_receipt_digest,
        )
        managed_erasure = all(item is not None for item in managed_parts)
        if any(item is not None for item in managed_parts) != managed_erasure:
            raise ValueError("managed key destruction evidence must be complete")
        if (
            self.key_destruction_receipt is not None
            and digest(self.key_destruction_receipt)
            != self.key_destruction_receipt_digest
        ):
            raise ValueError("managed key destruction receipt digest mismatch")
        if self.local_key_destruction is not None and managed_erasure:
            raise ValueError("local and managed key destruction evidence are exclusive")
        if self.cryptographic_erasure != managed_erasure:
            raise ValueError("cryptographic erasure requires managed destruction evidence")
        return self


class ContextLocalKeyDestructionV1(Closed):
    """Closed local key tombstone; it is canary evidence, not KMS erasure."""

    schema_version: Literal["ContextLocalKeyDestructionV1"] = (
        "ContextLocalKeyDestructionV1"
    )
    provider: Literal["file-wrapped-dek/v1"] = "file-wrapped-dek/v1"
    key_ref: Ident
    key_version: Ident
    namespace_digest: Digest
    wrapped_key_digest: Digest
    destroyed_at: int = Field(ge=0)
    verified: Literal[True]


class ContextDeletionReceiptV2(Closed):
    """Deletion evidence that binds the registered checkpoint durability mode."""

    schema_version: Literal["ContextDeletionReceiptV2"] = "ContextDeletionReceiptV2"
    context_cycle_id: Ident
    segment_chain_root: Digest
    context_seal_digest: Digest | None = None
    retention_class: Literal["ephemeral", "audit"]
    retention_expired_at: int = Field(ge=0)
    deleted_at: int = Field(ge=0)
    reason_code: Literal["policy_expiry", "operator_expiry"]
    retention_operation_key: Ident
    retention_request_digest: Digest
    logical_deletion: Literal[True]
    cryptographic_erasure: bool
    local_key_destruction: ContextLocalKeyDestructionV1 | None = None
    key_destruction_receipt: SignedV1 | None = None
    key_destruction_receipt_digest: Digest | None = None
    managed_key_provider_receipt_digest: Digest | None = None
    remote_deletion_receipt_digest: Digest | None = None
    checkpoint_manifest_checksum: Digest | None = None
    checkpoint_object_ref_digest: Digest | None = None
    checkpoint_durability_mode: Literal["unregistered", "local_ephemeral", "remote_registered"]
    checkpoint_durability_registration_receipt_digest: Digest | None = None

    @model_validator(mode="after")
    def durability_evidence(self) -> "ContextDeletionReceiptV2":
        managed_parts = (
            self.key_destruction_receipt,
            self.key_destruction_receipt_digest,
            self.managed_key_provider_receipt_digest,
        )
        managed_erasure = all(item is not None for item in managed_parts)
        if any(item is not None for item in managed_parts) != managed_erasure:
            raise ValueError("managed key destruction evidence must be complete")
        if (
            self.key_destruction_receipt is not None
            and digest(self.key_destruction_receipt)
            != self.key_destruction_receipt_digest
        ):
            raise ValueError("managed key destruction receipt digest mismatch")
        if self.local_key_destruction is not None and managed_erasure:
            raise ValueError("local and managed key destruction evidence are exclusive")
        if self.cryptographic_erasure != managed_erasure:
            raise ValueError("cryptographic erasure requires managed destruction evidence")
        if self.checkpoint_durability_mode == "unregistered":
            if (
                self.checkpoint_manifest_checksum is not None
                or self.checkpoint_durability_registration_receipt_digest is not None
                or self.remote_deletion_receipt_digest is not None
                or self.checkpoint_object_ref_digest is not None
            ):
                raise ValueError("unregistered cleanup cannot contain checkpoint evidence")
        elif self.checkpoint_durability_mode == "local_ephemeral":
            if self.checkpoint_durability_registration_receipt_digest is None:
                raise ValueError("local cleanup requires signed durability registration")
            if (
                self.remote_deletion_receipt_digest is not None
                or self.checkpoint_object_ref_digest is not None
            ):
                raise ValueError("local cleanup cannot contain remote evidence")
        else:
            if self.checkpoint_durability_registration_receipt_digest is None:
                raise ValueError("remote cleanup requires signed durability registration")
            remote_evidence = (
                self.remote_deletion_receipt_digest,
                self.checkpoint_object_ref_digest,
            )
            if self.checkpoint_manifest_checksum is None:
                if any(item is not None for item in remote_evidence):
                    raise ValueError(
                        "remote cleanup without a checkpoint cannot contain deletion evidence"
                    )
            elif any(item is None for item in remote_evidence):
                raise ValueError(
                    "remote checkpoint cleanup requires exact remote deletion evidence"
                )
        return self


class RuntimeBuildManifestV1(Closed):
    """Signed build identity for the exact installed runtime package bytes."""

    schema_version: Literal["ContextRuntimeBuildManifestV1"] = (
        "ContextRuntimeBuildManifestV1"
    )
    distribution: Literal["aether-context"] = "aether-context"
    version: str = Field(min_length=1, max_length=40)
    source_revision: Sha
    package_tree_digest: Digest
    wheel_digest: Digest
    recall_dataset_digest: Digest
    data_plane_case_generator_digest: Digest | None = None
    built_at: int = Field(ge=0)


class RuntimeEnvironmentV1(Closed):
    """Live-measurable identity of the interpreter and hosted dependencies."""

    schema_version: Literal["ContextRuntimeEnvironmentV1"] = (
        "ContextRuntimeEnvironmentV1"
    )
    python_implementation: str = Field(min_length=1, max_length=40)
    python_version: str = Field(min_length=1, max_length=40)
    python_abi: str = Field(min_length=1, max_length=200)
    python_executable_digest: Digest
    dependency_versions: dict[str, str]
    dependency_artifact_digests: dict[str, Digest]
    sqlite_version: str = Field(min_length=1, max_length=40)
    package_import_root_digest: Digest
    bytecode_policy: Literal["source-only/no-bytecode/v1"] = (
        "source-only/no-bytecode/v1"
    )

    @model_validator(mode="after")
    def exact_hosted_dependencies(self) -> "RuntimeEnvironmentV1":
        expected = {"cryptography", "numpy", "pydantic"}
        if (
            set(self.dependency_versions) != expected
            or set(self.dependency_artifact_digests) != expected
        ):
            raise ValueError("runtime environment requires exact hosted dependency versions")
        if any(
            not isinstance(version, str) or not 1 <= len(version) <= 100
            for version in self.dependency_versions.values()
        ):
            raise ValueError("runtime dependency versions are invalid")
        return self


class RuntimeBuildManifestV2(RuntimeBuildManifestV1):
    """Build V1 plus a lock over the live executable runtime environment."""

    schema_version: Literal["ContextRuntimeBuildManifestV2"] = "ContextRuntimeBuildManifestV2"  # type: ignore[assignment]
    runtime_environment: RuntimeEnvironmentV1
    locked_environment_digest: Digest

    @model_validator(mode="after")
    def environment_digest_matches(self) -> "RuntimeBuildManifestV2":
        if digest(self.runtime_environment) != self.locked_environment_digest:
            raise ValueError("runtime environment digest mismatch")
        return self


class RuntimeEnvironmentV2(Closed):
    """Complete audited import closure for the hosted daemon."""

    schema_version: Literal["ContextRuntimeEnvironmentV2"] = (
        "ContextRuntimeEnvironmentV2"
    )
    python_implementation: str = Field(min_length=1, max_length=40)
    python_version: str = Field(min_length=1, max_length=40)
    python_abi: str = Field(min_length=1, max_length=200)
    python_executable_digest: Digest
    dependency_import_closure: list[str] = Field(min_length=6, max_length=32)
    dependency_versions: dict[str, str]
    dependency_artifact_digests: dict[str, Digest]
    sqlite_version: str = Field(min_length=1, max_length=40)
    package_import_root_digest: Digest
    bytecode_policy: Literal["preimport-guard/source-only/v2"] = (
        "preimport-guard/source-only/v2"
    )

    @model_validator(mode="after")
    def complete_hosted_import_closure(self) -> "RuntimeEnvironmentV2":
        required = {
            "annotated-types",
            "cffi",
            "cryptography",
            "numpy",
            "pydantic",
            "pydantic_core",
            "pycparser",
            "typing-inspection",
            "typing_extensions",
        }
        closure = self.dependency_import_closure
        if (
            closure != sorted(set(closure))
            or not required.issubset(closure)
            or set(self.dependency_versions) != set(closure)
            or set(self.dependency_artifact_digests) != set(closure)
        ):
            raise ValueError("runtime environment requires the audited import closure")
        if any(
            not isinstance(version, str) or not 1 <= len(version) <= 100
            for version in self.dependency_versions.values()
        ):
            raise ValueError("runtime dependency versions are invalid")
        return self


class RuntimeBuildManifestV3(RuntimeBuildManifestV1):
    """Build identity bound to the complete guarded executable environment."""

    schema_version: Literal["ContextRuntimeBuildManifestV3"] = "ContextRuntimeBuildManifestV3"  # type: ignore[assignment]
    runtime_environment: RuntimeEnvironmentV2
    locked_environment_digest: Digest

    @model_validator(mode="after")
    def environment_digest_matches(self) -> "RuntimeBuildManifestV3":
        if digest(self.runtime_environment) != self.locked_environment_digest:
            raise ValueError("runtime environment digest mismatch")
        return self


class ExecutorStabilityReceiptV1(Closed):
    """Independent CI/executor observation bound to one measured build."""

    schema_version: Literal["ContextExecutorStabilityReceiptV1"] = (
        "ContextExecutorStabilityReceiptV1"
    )
    source_revision: Sha
    profile_benchmark_digest: Digest
    target_concurrency: int = Field(ge=1)
    ci_receipt_digest: Digest
    executor_stable: Literal[True]
    started_at: int = Field(ge=0)
    finished_at: int = Field(ge=0)
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered_observation(self) -> "ExecutorStabilityReceiptV1":
        if not self.started_at <= self.finished_at <= self.issued_at < self.expires_at:
            raise ValueError("executor observation ordering is invalid")
        if self.expires_at - self.issued_at > 604800:
            raise ValueError("executor evidence may be valid for at most seven days")
        return self


class DataPlaneIsolationReceiptV1(Closed):
    """Independent randomized isolation evidence from the hosted request boundary."""

    schema_version: Literal["ContextDataPlaneIsolationReceiptV1"] = (
        "ContextDataPlaneIsolationReceiptV1"
    )
    source_revision: Sha
    profile_benchmark_digest: Digest
    boundary: Literal["hosted-data-plane/v1"] = "hosted-data-plane/v1"
    case_generator_digest: Digest
    cases: int = Field(ge=1_000_000)
    unauthorized_records: int = Field(ge=0)
    result_digest: Digest
    started_at: int = Field(ge=0)
    finished_at: int = Field(ge=0)
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered_observation(self) -> "DataPlaneIsolationReceiptV1":
        if not self.started_at <= self.finished_at <= self.issued_at < self.expires_at:
            raise ValueError("data-plane isolation observation ordering is invalid")
        if self.expires_at - self.issued_at > 604800:
            raise ValueError("data-plane isolation evidence may be valid for at most seven days")
        return self


class HostedScaleReceiptV1(Closed):
    """Independent, measured evidence for one exact hosted profile."""

    schema_version: Literal["HostedScaleReceiptV1"] = "HostedScaleReceiptV1"
    profile_benchmark_digest: Digest
    source_revision: Sha
    source_tree_clean: Literal[True]
    index_implementation: str = Field(min_length=1, max_length=100)
    index_version: str = Field(min_length=1, max_length=100)
    embedding_version: str = Field(min_length=1, max_length=100)
    storage_version: str = Field(min_length=1, max_length=100)
    configured_max_records: int = Field(ge=1)
    configured_max_indexed_bytes: int = Field(ge=1024)
    indexed_records: int = Field(ge=1)
    indexed_bytes: int = Field(ge=1)
    reachable_tokens: int = Field(ge=1)
    resident_peak_bytes: int = Field(ge=1)
    retrieval_samples: int = Field(ge=20)
    target_concurrency: int = Field(ge=1)
    retrieval_p50_ms: int = Field(ge=0)
    retrieval_p95_ms: int = Field(ge=0)
    retrieval_p99_ms: int = Field(ge=0)
    checkpoint_ms: int = Field(ge=0)
    checkpoint_receipt_digest: Digest
    hydrate_ms: int = Field(ge=0)
    hydrate_receipt_digest: Digest
    rebuild_ms: int = Field(ge=0)
    rebuild_receipt_digest: Digest
    recall_dataset_name: str = Field(min_length=1, max_length=100)
    recall_dataset_digest: Digest
    recall_basis_points: int = Field(ge=0, le=10000)
    isolation_cases: int = Field(ge=1_000_000)
    unauthorized_records: int = Field(ge=0)
    isolation_result_digest: Digest
    isolation_evidence_class: Literal["predicate", "data_plane"]
    data_plane_isolation_receipt: SignedV1 | None = None
    executor_stable: bool
    executor_stability_receipt: SignedV1 | None = None
    started_at: int = Field(ge=0)
    finished_at: int = Field(ge=0)
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered_measurements(self) -> "HostedScaleReceiptV1":
        if not (
            self.retrieval_p50_ms <= self.retrieval_p95_ms <= self.retrieval_p99_ms
            and self.started_at <= self.finished_at <= self.issued_at < self.expires_at
        ):
            raise ValueError("scale measurement ordering is invalid")
        if self.expires_at - self.issued_at > 604800:
            raise ValueError("scale evidence may be valid for at most seven days")
        if self.executor_stable != (self.executor_stability_receipt is not None):
            raise ValueError("executor stability claim requires independent evidence")
        if (self.isolation_evidence_class == "data_plane") != (
            self.data_plane_isolation_receipt is not None
        ):
            raise ValueError("data-plane isolation claim requires independent evidence")
        return self


class ClosureProofV1(Closed):
    schema_version: Literal["ContextClosureProofV1"] = "ContextClosureProofV1"
    namespace: NamespaceV1
    binding_digest: Digest
    final_root: Digest
    final_cursor: int = Field(ge=0)
    execution_dag_digest: Digest
    plan_ir_digest: Digest
    shared_ir_digest: Digest
    repo_main_sha: Sha
    git_head_sha: Sha
    pr_head_sha: Sha
    pr_url: str = Field(pattern=r"^https://[^/?#]+/[^/?#]+/[^/?#]+/pull/[1-9][0-9]*$")
    ci_receipt: Digest
    nano_receipt: Digest
    proof_receipt: Digest
    memory_candidate_digest: Digest
    promotion_set_root: Digest
    accepted_record_ids: list[Ident] = Field(default_factory=list, max_length=1000)
    rejected_record_ids: list[Ident] = Field(default_factory=list, max_length=1000)
    expires_at: int = Field(ge=0)

    @model_validator(mode="after")
    def exact_heads(self) -> "ClosureProofV1":
        if self.git_head_sha != self.pr_head_sha:
            raise ValueError("PR head must equal the independently verified Git head")
        if len(set(self.accepted_record_ids)) != len(self.accepted_record_ids):
            raise ValueError("duplicate promoted record")
        if set(self.accepted_record_ids) & set(self.rejected_record_ids):
            raise ValueError("promotion sets overlap")
        return self
