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
        Literal["append", "retrieve", "checkpoint", "seal", "status", "hydrate", "expire"]
    ] = Field(min_length=1, max_length=7)
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
