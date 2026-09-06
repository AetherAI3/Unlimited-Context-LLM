"""Closed SC-CONTEXT-01 wire contracts. Requires the optional hosted extra."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Ident = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,199}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Sha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


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


class SignedV1(Closed):
    payload: dict
    key_id: Ident
    signature: str = Field(min_length=64, max_length=128)


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
