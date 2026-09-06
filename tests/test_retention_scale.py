import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing
import os
import py_compile
import shutil
import sqlite3
import sys
import time
import traceback
from collections import namedtuple
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from aether_context.contracts import (
    AppendRequestV1,
    CapabilityV1,
    ContextBindingCommandV2,
    ContextBindingV1,
    ContextCheckpointDurabilityCommandV1,
    ContextCheckpointDurabilityReceiptV1,
    ContextCheckpointDurabilityReceiptV2,
    ContextDeletionReceiptV2,
    ContextHealthV1,
    ContextHealthV2,
    ContextHydrateCommandV2,
    ContextHydrateReceiptV2,
    ContextProfileV1,
    ContextReservationCommandV2,
    ContextReservationReceiptV2,
    HostedScaleReceiptV1,
    NamespaceV1,
    RetentionCommandV1,
    RetrieveRequestV1,
    RuntimeEnvironmentV2,
    canonical,
    digest,
)
from aether_context import __version__
from aether_context.crypto import ContextFault, EnvelopeCipher, ReceiptSigner, verify
from aether_context.engine import ContextEngine
from aether_context.retention import FileCycleKeyProvider, RetentionManager
from aether_context.service import ContextService
from aether_context.storage_v2 import SegmentStoreV2
from aether_context.scale import (
    profile_benchmark_digest,
    qualify_scale_receipt,
    run_namespace_isolation,
    runtime_environment_identity,
    runtime_package_tree_digest,
)


def _live_ipc_scale_worker(root: str, output) -> None:
    """Isolated Linux worker so VmHWM covers only the live service scenario."""

    try:
        import aether_context.engine as engine_module
        from bench.hosted_scale import _UnixServiceClient

        dependencies = [
            "annotated-types",
            "cffi",
            "cryptography",
            "numpy",
            "pycparser",
            "pydantic",
            "pydantic_core",
            "typing-inspection",
            "typing_extensions",
        ]
        environment = RuntimeEnvironmentV2(
            python_implementation="cpython",
            python_version="3.13.0",
            python_abi="cp313-live-ipc-test",
            python_executable_digest=digest("live-ipc-python"),
            dependency_import_closure=dependencies,
            dependency_versions={name: "1.0" for name in dependencies},
            dependency_artifact_digests={
                name: digest(name) for name in dependencies
            },
            sqlite_version=sqlite3.sqlite_version,
            package_import_root_digest=digest("live-ipc-site-packages"),
        )
        package_digest = digest("live-ipc-package-tree")
        engine_module.runtime_environment_identity = lambda **_: environment
        engine_module.runtime_package_tree_digest = lambda **_: package_digest
        build = ReceiptSigner("live-ipc-build", Ed25519PrivateKey.generate())
        source_revision = "d" * 40
        build_manifest = build.sign(
            {
                "schema_version": "ContextRuntimeBuildManifestV3",
                "distribution": "aether-context",
                "version": __version__,
                "source_revision": source_revision,
                "package_tree_digest": package_digest,
                "wheel_digest": digest("live-ipc-wheel"),
                "recall_dataset_digest": digest("live-ipc-recall"),
                "data_plane_case_generator_digest": digest("live-ipc-cases"),
                "built_at": 900,
                "runtime_environment": environment.model_dump(mode="json"),
                "locked_environment_digest": digest(environment),
            }
        )
        item = _keyed_cycle(
            Path(root),
            reservation_protocol_version=2,
            runtime_build_manifest=build_manifest,
            runtime_build_keys={build.key_id: build.key.public_key()},
            profile_overrides={
                "max_records": 70,
                "max_indexed_bytes": 15_500_000,
                "max_record_bytes": 250_000,
                "max_checkpoint_bytes": 16_000_000,
                "max_operations_per_cycle": 256,
                "max_concurrent_cycles": 4,
                "max_calls_per_minute": 6000,
            },
        )
        alphabet = bytes(
            value for value in range(33, 127) if value not in b":=-"
        )
        for index in range(61):
            source = hashlib.shake_256(f"live-ipc-{index}".encode()).digest(
                248_000
            )
            text = bytes(alphabet[value % len(alphabet)] for value in source).decode()
            item["engine"].append(
                item["cap"](),
                AppendRequestV1(
                    idempotency_key=f"live-ipc-fill-{index}",
                    text=text,
                    action_ref=f"live-ipc-fill-{index}",
                ),
            )
        service = ContextService(item["engine"])
        checkpoint_request = {
            "operation": "checkpoint",
            "capability": item["cap"]("durability"),
            "idempotency_key": "live-ipc-near-bound",
        }
        with _UnixServiceClient(service) as client:
            with ThreadPoolExecutor(max_workers=5) as executor:
                checkpoint_future = executor.submit(
                    client.dispatch,
                    checkpoint_request,
                    read_chunk_size=16_384,
                    read_delay_seconds=0.001,
                )
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    with item["engine"].store.connection() as db:
                        if db.execute("SELECT checkpoint FROM cycles").fetchone()[0]:
                            break
                    time.sleep(0.01)
                else:
                    raise AssertionError("live checkpoint did not commit")
                small_requests = [
                    {
                        "operation": "status",
                        "capability": item["cap"]("durability"),
                        "after": 0,
                    },
                    {
                        "operation": "status",
                        "capability": item["cap"]("durability"),
                        "after": 0,
                    },
                    {
                        "operation": "retrieve",
                        "capability": item["cap"](),
                        "request": RetrieveRequestV1(
                            turn_id="live-ipc-concurrent-retrieve",
                            query="unlikely-query",
                            native_window=32768,
                            remaining_prompt_budget=4096,
                        ).model_dump(mode="json"),
                    },
                    {
                        "operation": "append",
                        "capability": item["cap"](),
                        "request": AppendRequestV1(
                            idempotency_key="live-ipc-concurrent-append",
                            text="concurrent small request",
                            action_ref="live-ipc-concurrent-append",
                        ).model_dump(mode="json"),
                    },
                ]
                small = list(executor.map(client.dispatch, small_requests))
                checkpoint = checkpoint_future.result(timeout=180)
        assert len(small) == item["profile"].max_concurrent_cycles
        raw_pack = base64.b64decode(checkpoint["pack"], validate=True)
        assert (
            len(raw_pack) * 10
            >= service.max_live_hydrate_pack_bytes * 9
        )
        before = item["engine"].status(item["cap"]("durability"))["payload"]
        with pytest.raises(ContextFault, match="checkpoint_transport_limit"):
            item["engine"].checkpoint(
                item["cap"]("durability"),
                "live-ipc-near-bound",
                transport_max_bytes=len(raw_pack) - 1,
            )
        after = item["engine"].status(item["cap"]("durability"))["payload"]
        assert (after["revision"], after["checkpoint_number"]) == (
            before["revision"],
            before["checkpoint_number"],
        )

        claim = checkpoint["receipt"]["payload"]
        hydrate_command = item["authority"].sign(
            ContextHydrateCommandV2(
                namespace=item["namespace"],
                manifest_checksum=claim["manifest_checksum"],
                key_ref=claim["key_ref"],
                key_provider=claim["key_provider"],
                key_version=claim["key_version"],
                fence=2,
                expected_source_revision=source_revision,
                issued_at=item["clock"][0],
                expires_at=item["clock"][0] + 900,
            ).model_dump(mode="json")
        )
        replacement = item["make"](Path(root) / "live-ipc-replacement.sqlite3")
        replacement_service = ContextService(
            replacement,
            checkpoint_receipt_keys={
                item["service"].key_id: item["service"].key.public_key()
            },
        )
        hydrate_request = {
            "operation": "hydrate",
            "grant": hydrate_command,
            "checkpoint_receipt": checkpoint["receipt"],
            "pack": checkpoint["pack"],
        }
        with _UnixServiceClient(replacement_service) as client:
            hydrated = client.dispatch(hydrate_request)
            assert client.dispatch(hydrate_request) == hydrated
        assert hydrated["payload"]["schema_version"] == "ContextHydrateReceiptV2"
        peak = 0
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                peak = int(line.split()[1]) * 1024
                break
        assert 0 < peak <= 480 * 1024 * 1024
        output.put(
            {
                "pack_bytes": len(raw_pack),
                "live_pack_limit_bytes": service.max_live_hydrate_pack_bytes,
                "peak_rss_bytes": peak,
                "small_calls": len(small),
            }
        )
    except BaseException:
        output.put({"error": traceback.format_exc()})


@pytest.mark.skipif(
    os.name == "nt" or os.environ.get("AETHER_LIVE_IPC_SCALE") != "1",
    reason="dedicated Linux live-IPC memory gate",
)
def test_live_unix_ipc_near_cap_memory_and_concurrency_gate(tmp_path):
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=_live_ipc_scale_worker,
        args=(str(tmp_path / "live-worker"), output),
    )
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        pytest.fail("live IPC memory worker timed out")
    assert process.exitcode == 0
    result = output.get(timeout=10)
    print(f"live IPC evidence: {result}")
    assert "error" not in result, result.get("error")
    assert (
        result["pack_bytes"] * 10
        >= result["live_pack_limit_bytes"] * 9
    )
    assert result["peak_rss_bytes"] <= 480 * 1024 * 1024
    assert result["small_calls"] == 4


class ManagedTestCycleKeys(FileCycleKeyProvider):
    provider = "managed-test-kms/v1"
    key_version = "managed-test-key/v1"

    def __init__(self, root, wrapper, destruction_signer, clock):
        super().__init__(root, wrapper)
        self.destruction_signer = destruction_signer
        self.clock = clock
        self.receipts = {}
        self.active_key_version = self.key_version
        self.key_versions = {}

    def current_key_version(self):
        return self.active_key_version

    def ensure(self, namespace):
        key_ref = super().ensure(namespace)
        self.key_versions.setdefault(key_ref, self.active_key_version)
        return key_ref

    def version_for(self, key_ref):
        try:
            return self.key_versions[key_ref]
        except KeyError as exc:
            raise ContextFault("context_key_version_unavailable") from exc

    def destroy(self, key_ref, namespace, destroyed_at):
        local = super().destroy(key_ref, namespace, destroyed_at)
        if key_ref not in self.receipts:
            self.receipts[key_ref] = self.destruction_signer.sign(
                {
                    "schema_version": "ContextKeyDestructionReceiptV1",
                    "provider": self.provider,
                    "key_ref": key_ref,
                    "namespace_digest": digest(namespace),
                    "key_version": self.version_for(key_ref),
                    "destroyed": True,
                    "destroyed_at": local["destroyed_at"],
                    "issued_at": self.clock[0],
                    "expires_at": self.clock[0] + 100,
                }
            )
        return self.receipts[key_ref]


def _crash_uncommitted_write(path: str) -> None:
    db = sqlite3.connect(path)
    db.execute("BEGIN IMMEDIATE")
    db.execute("INSERT INTO chaos_probe VALUES(1)")
    os._exit(17)


def _keyed_cycle(
    tmp_path,
    *,
    managed=True,
    keyed=True,
    legacy_binding=False,
    durability_mode="remote_registered",
    register_durability=True,
    reservation_protocol_version=1,
    runtime_build_manifest=None,
    runtime_build_keys=None,
    profile_overrides=None,
):
    authority = ReceiptSigner("gateway-retention", Ed25519PrivateKey.generate())
    proof = ReceiptSigner("proof-retention", Ed25519PrivateKey.generate())
    deletion = ReceiptSigner("object-store-deletion", Ed25519PrivateKey.generate())
    provider_authority = ReceiptSigner("key-provider-authority", Ed25519PrivateKey.generate())
    destruction = ReceiptSigner("kms-destruction", Ed25519PrivateKey.generate())
    service = ReceiptSigner("contextd-retention", Ed25519PrivateKey.generate())
    profile_values = dict(
        index_version=sqlite3.sqlite_version,
        disk_free_floor=0,
        max_records=100,
        max_indexed_bytes=1_000_000,
        max_checkpoint_bytes=16_000_000,
        retention_seconds=60,
    )
    profile_values.update(profile_overrides or {})
    profile = ContextProfileV1(**profile_values)
    clock = [1000]
    global_cipher = EnvelopeCipher(os.urandom(32))
    provider = None
    if keyed:
        if managed:
            provider = ManagedTestCycleKeys(
                tmp_path / "cycle-keys",
                EnvelopeCipher(os.urandom(32), domain=b"test-cycle-wrapper/v1"),
                destruction,
                clock,
            )
        else:
            provider = FileCycleKeyProvider(
                tmp_path / "cycle-keys",
                EnvelopeCipher(os.urandom(32), domain=b"test-cycle-wrapper/v1"),
            )
    provider_receipt = (
        provider_authority.sign(
            {
                "schema_version": "ContextManagedKeyProviderReceiptV1",
                "provider": provider.provider,
                "key_version": provider.key_version,
                "shared_hydrate_supported": True,
                "verified_destruction_supported": True,
                "issued_at": 900,
                "expires_at": 2000,
            }
        )
        if managed and provider is not None
        else None
    )
    provider_receipts = [provider_receipt] if provider_receipt is not None else []

    def make(path):
        return ContextEngine(
            path,
            profile,
            global_cipher,
            service,
            {authority.key_id: authority.key.public_key()},
            {proof.key_id: proof.key.public_key()},
            clock=lambda: clock[0],
            cycle_keys=provider,
            remote_deletion_keys={deletion.key_id: deletion.key.public_key()},
            managed_key_provider_receipts=(provider_receipts or None),
            key_provider_keys=(
                {provider_authority.key_id: provider_authority.key.public_key()}
                if provider_receipts
                else None
            ),
            key_destruction_keys={destruction.key_id: destruction.key.public_key()},
            runtime_build_manifest=runtime_build_manifest,
            runtime_build_keys=runtime_build_keys,
        )

    engine = make(tmp_path / "context.sqlite3")
    reservation_payload = {
        "operation": "reserve",
        "owner_id": "owner-retention",
        "project_id": "project-retention",
        "idempotency_key": "request-retention",
        "request_digest": digest("request-retention"),
        "profile_digest": digest(profile),
        "expires_at": 1500,
    }
    if reservation_protocol_version == 2:
        reservation_payload = ContextReservationCommandV2(
            **reservation_payload
        ).model_dump(mode="json")
        reservation = engine.reserve_v2(authority.sign(reservation_payload))
    elif reservation_protocol_version == 1:
        reservation = engine.reserve(authority.sign(reservation_payload))
    else:
        raise ValueError("unsupported test reservation protocol")
    namespace = NamespaceV1(
        owner_id="owner-retention",
        project_id="project-retention",
        objective_id="objective-retention",
        context_cycle_id=reservation["payload"]["context_cycle_id"],
    )
    authority_payload = {"objective": "retention test", "paths": ["src"]}
    snapshot = {
        "project_id": namespace.project_id,
        "graph_id": "graph-retention",
        "policy_digest": digest("policy-retention"),
        "nodes": [],
    }
    binding = ContextBindingV1(
        namespace=namespace,
        context_bucket_id=reservation["payload"]["context_bucket_id"],
        repository_id="repository-retention",
        repo_main_sha="a" * 40,
        project_graph_id="graph-retention",
        graph_revision=1,
        graph_checksum=digest(snapshot),
        plan_digest=digest("plan-retention"),
        execution_profile_digest=digest("execution-retention"),
        shared_ir_digest=digest("ir-retention"),
        authorization_receipt_ref="authority-retention",
        authorization_digest=digest(authority_payload),
        policy_digest=digest("policy-retention"),
        redaction_digest=digest("redaction-retention"),
        profile_digest=digest(profile),
        captain_binding_id="captain-retention",
        created_at=1000,
        expires_at=100000,
    )
    binding_payload = binding.model_dump()
    if legacy_binding:
        binding_payload.pop("retention_class")
    binding_digest = digest(binding_payload)
    if reservation_protocol_version == 2:
        reservation_claim = ContextReservationReceiptV2.model_validate(
            verify(reservation, {service.key_id: service.key.public_key()})
        )
        engine.bind_v2(
            authority.sign(
                ContextBindingCommandV2(
                    binding=ContextBindingV1.model_validate(binding_payload),
                    reservation_receipt=reservation,
                    expected_reservation_revision=reservation_claim.revision,
                ).model_dump(mode="json")
            ),
            authority_payload,
            snapshot,
        )
    else:
        engine.bind(authority.sign(binding_payload), authority_payload, snapshot)
    durability_command = None
    durability_receipt = None
    if register_durability:
        with engine.store.connection() as db:
            durability_revision = db.execute(
                "SELECT revision FROM cycles WHERE id=?", (namespace.context_cycle_id,)
            ).fetchone()[0]
        durability_payload = ContextCheckpointDurabilityCommandV1(
            namespace=namespace,
            idempotency_key="checkpoint-durability",
            expected_revision=durability_revision,
            mode=durability_mode,
            issued_at=clock[0],
            expires_at=clock[0] + 100,
        ).model_dump(mode="json")
        durability_command = authority.sign(durability_payload)
        durability_receipt = engine.register_checkpoint_durability(
            durability_command
        )

    def cap(role="worker", fence=1, source="tool_observation"):
        return authority.sign(
            CapabilityV1(
                namespace=namespace,
                principal_id="principal-retention",
                lane_id="lane-retention",
                task_id="task-retention",
                role=role,
                operations=[
                    "append",
                    "retrieve",
                    "checkpoint",
                    "seal",
                    "status",
                    "expire",
                    "retention",
                ],
                source_class=source,
                binding_digest=binding_digest,
                policy_digest=binding.policy_digest,
                profile_digest=digest(profile),
                fence=fence,
                expires_at=90000,
                max_bytes=profile.max_record_bytes,
                max_tokens=12000,
            ).model_dump()
        )

    def seal():
        checkpoint = engine.checkpoint(cap("durability"), "terminal-checkpoint")
        closure = proof.sign(
            {
                "schema_version": "ContextClosureProofV1",
                "namespace": namespace.model_dump(),
                "binding_digest": binding_digest,
                "final_root": checkpoint["receipt"]["payload"]["segment_chain_root"],
                "final_cursor": checkpoint["receipt"]["payload"]["cursor"],
                "execution_dag_digest": digest("dag-retention"),
                "plan_ir_digest": binding.shared_ir_digest,
                "shared_ir_digest": digest("final-ir-retention"),
                "repo_main_sha": binding.repo_main_sha,
                "git_head_sha": "b" * 40,
                "pr_head_sha": "b" * 40,
                "pr_url": "https://github.com/org/repo/pull/1",
                "ci_receipt": digest("ci-retention"),
                "nano_receipt": digest("nano-retention"),
                "proof_receipt": digest("proof-retention"),
                "memory_candidate_digest": digest("memory-retention"),
                "promotion_set_root": digest([]),
                "accepted_record_ids": [],
                "rejected_record_ids": [],
                "expires_at": 2000,
            }
        )
        sealed = engine.seal(cap("verifier"), closure, checkpoint["receipt"])
        return checkpoint, sealed

    def expire_command(checkpoint, *, manifest=None, object_ref=None):
        object_ref = object_ref or digest("spaces://context/checkpoint-object")
        remote = deletion.sign(
            {
                "schema_version": "ContextRemoteDeletionReceiptV1",
                "namespace": namespace.model_dump(),
                "manifest_checksum": manifest
                or checkpoint["receipt"]["payload"]["manifest_checksum"],
                "object_ref_digest": object_ref,
                "deleted": True,
                "issued_at": clock[0],
                "expires_at": clock[0] + 100,
            }
        )
        command = authority.sign(
            {
                "schema_version": "ContextRetentionCommandV1",
                "operation": "expire",
                "namespace": namespace.model_dump(),
                "idempotency_key": "expire-retention",
                "expected_revision": engine.status(cap("durability"))["payload"]["revision"],
                "reason_code": "policy_expiry",
                "remote_deletion_receipt": remote,
                "checkpoint_object_ref_digest": object_ref,
                "issued_at": clock[0],
                "expires_at": clock[0] + 100,
            }
        )
        return command

    return {
        "engine": engine,
        "make": make,
        "cap": cap,
        "seal": seal,
        "expire_command": expire_command,
        "authority": authority,
        "proof": proof,
        "deletion": deletion,
        "provider_authority": provider_authority,
        "destruction": destruction,
        "service": service,
        "global_cipher": global_cipher,
        "provider": provider,
        "provider_receipts": provider_receipts,
        "namespace": namespace,
        "binding": binding,
        "binding_payload": binding_payload,
        "binding_digest": binding_digest,
        "clock": clock,
        "profile": profile,
        "durability_command": durability_command,
        "durability_receipt": durability_receipt,
    }


def test_retention_default_and_hard_ceiling():
    omitted = ContextProfileV1(index_version=sqlite3.sqlite_version)
    explicit = ContextProfileV1(
        index_version=sqlite3.sqlite_version,
        max_checkpoint_bytes=64_000_000,
    )
    assert omitted.retention_seconds == 86400
    assert omitted.max_checkpoint_bytes == 64_000_000
    assert digest(omitted) == digest(explicit)
    with pytest.raises(ValueError):
        ContextProfileV1(index_version=sqlite3.sqlite_version, retention_seconds=604801)


def test_storage_v2_additively_migrates_legacy_cycles(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE cycles (id TEXT PRIMARY KEY,owner TEXT NOT NULL,project TEXT NOT NULL,"
            "request_key TEXT NOT NULL,request_digest TEXT NOT NULL,profile TEXT NOT NULL,"
            "state TEXT NOT NULL,revision INTEGER NOT NULL DEFAULT 1,fence INTEGER NOT NULL "
            "DEFAULT 1,binding TEXT,binding_digest TEXT,authority BLOB,cursor INTEGER NOT NULL "
            "DEFAULT 0,root TEXT NOT NULL,bytes INTEGER NOT NULL DEFAULT 0,expires INTEGER NOT "
            "NULL,hold INTEGER NOT NULL DEFAULT 0,checkpoint INTEGER NOT NULL DEFAULT 0,"
            "UNIQUE(owner,project,request_key))"
        )
        db.execute(
            "INSERT INTO cycles(id,owner,project,request_key,request_digest,profile,state,root,"
            "expires) VALUES('legacy','owner','project','key','digest','profile','EXPIRED',?,1)",
            ("0" * 64,),
        )
    store = SegmentStoreV2(path, 1_048_576)
    with store.connection() as db:
        row = db.execute("SELECT * FROM cycles WHERE id='legacy'").fetchone()
        assert row["retention_class"] == "ephemeral"
        assert row["cleanup_state"] == "NONE"
        assert row["cleanup_operation_key"] is None
        assert row["cleanup_request_digest"] is None
        assert row["key_ref"] is None
        assert row["snapshot"] is None
        assert row["checkpoint_durability_mode"] == "remote_registered"
        assert row["checkpoint_durability_legacy"] == 1
        assert row["checkpoint_durability_command_digest"] is None
        assert row["reservation_release_digest"] is None
        db.execute(
            "INSERT INTO cycles(id,owner,project,request_key,request_digest,profile,state,root,"
            "expires) VALUES('old-writer-new','owner','project','key-2','digest','profile',"
            "'RESERVED',?,2)",
            ("0" * 64,),
        )
        inserted = db.execute(
            "SELECT checkpoint_durability_mode,checkpoint_durability_legacy FROM cycles "
            "WHERE id='old-writer-new'"
        ).fetchone()
        # A writer that omits reservation_protocol_version is a V1 writer.
        # The explicit drain path preserves its legacy remote-checkpoint behavior.
        assert tuple(inserted) == ("remote_registered", 1)


@pytest.mark.parametrize("interrupt_after", ["mode", "legacy", "backfill"])
def test_durability_migration_rolls_back_every_partial_semantic_step(
    tmp_path, monkeypatch, interrupt_after
):
    path = tmp_path / f"legacy-crash-{interrupt_after}.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE cycles (id TEXT PRIMARY KEY,owner TEXT NOT NULL,project TEXT NOT NULL,"
            "request_key TEXT NOT NULL,request_digest TEXT NOT NULL,profile TEXT NOT NULL,"
            "state TEXT NOT NULL,revision INTEGER NOT NULL DEFAULT 1,fence INTEGER NOT NULL "
            "DEFAULT 1,binding TEXT,binding_digest TEXT,authority BLOB,cursor INTEGER NOT NULL "
            "DEFAULT 0,root TEXT NOT NULL,bytes INTEGER NOT NULL DEFAULT 0,expires INTEGER NOT "
            "NULL,hold INTEGER NOT NULL DEFAULT 0,checkpoint INTEGER NOT NULL DEFAULT 0,"
            "UNIQUE(owner,project,request_key))"
        )
        db.execute(
            "INSERT INTO cycles(id,owner,project,request_key,request_digest,profile,state,root,"
            "expires) VALUES('legacy','owner','project','key','digest','profile','EXPIRED',?,1)",
            ("0" * 64,),
        )

    original = SegmentStoreV2._migrate_cycles

    def interrupted(db):
        db.execute("ALTER TABLE cycles ADD COLUMN snapshot BLOB")
        db.execute(
            "ALTER TABLE cycles ADD COLUMN checkpoint_durability_mode "
            "TEXT NOT NULL DEFAULT 'unregistered'"
        )
        if interrupt_after == "mode":
            raise RuntimeError("simulated migration loss")
        db.execute(
            "ALTER TABLE cycles ADD COLUMN checkpoint_durability_legacy "
            "INTEGER NOT NULL DEFAULT 0"
        )
        if interrupt_after == "legacy":
            raise RuntimeError("simulated migration loss")
        db.execute(
            "UPDATE cycles SET checkpoint_durability_mode='remote_registered',"
            "checkpoint_durability_legacy=1"
        )
        raise RuntimeError("simulated migration loss")

    monkeypatch.setattr(SegmentStoreV2, "_migrate_cycles", staticmethod(interrupted))
    with pytest.raises(RuntimeError, match="simulated migration loss"):
        SegmentStoreV2(path, 1_048_576)
    with sqlite3.connect(path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(cycles)")}
        assert "snapshot" not in columns
        assert "checkpoint_durability_mode" not in columns
        assert "checkpoint_durability_legacy" not in columns

    monkeypatch.setattr(SegmentStoreV2, "_migrate_cycles", staticmethod(original))
    store = SegmentStoreV2(path, 1_048_576)
    with store.connection() as db:
        row = db.execute(
            "SELECT checkpoint_durability_mode,checkpoint_durability_legacy "
            "FROM cycles WHERE id='legacy'"
        ).fetchone()
        assert tuple(row) == ("remote_registered", 1)


def test_checkpoint_durability_registration_is_closed_and_lost_ack_safe(tmp_path):
    item = _keyed_cycle(tmp_path)
    receipt = item["durability_receipt"]
    claim = ContextCheckpointDurabilityReceiptV1.model_validate(
        verify(
            receipt,
            {item["engine"].signer.key_id: item["engine"].signer.key.public_key()},
        )
    )
    assert claim.mode == "remote_registered"
    checkpoint = item["engine"].checkpoint(
        item["cap"]("durability"), "durability-registration-checkpoint"
    )
    assert "durability_mode" not in checkpoint["receipt"]["payload"]
    item["clock"][0] += 101
    assert ContextService(item["engine"]).dispatch(
        {
            "operation": "checkpoint_durability",
            "command": item["durability_command"],
        }
    ) == receipt

    conflicting = dict(item["durability_command"]["payload"])
    conflicting["mode"] = "local_ephemeral"
    with pytest.raises(ContextFault, match="checkpoint_durability_conflict"):
        item["engine"].register_checkpoint_durability(
            item["authority"].sign(conflicting)
        )
    extra = dict(item["durability_command"]["payload"])
    extra["unexpected"] = True
    with pytest.raises(ContextFault, match="checkpoint_durability_schema"):
        item["engine"].register_checkpoint_durability(item["authority"].sign(extra))


def test_fresh_cycle_cannot_checkpoint_or_claim_remote_before_registration(tmp_path):
    item = _keyed_cycle(
        tmp_path, register_durability=False, reservation_protocol_version=2
    )
    with item["engine"].store.connection() as db:
        row = db.execute(
            "SELECT checkpoint_durability_mode,checkpoint_durability_receipt "
            "FROM cycles WHERE id=?",
            (item["namespace"].context_cycle_id,),
        ).fetchone()
    assert tuple(row) == ("unregistered", None)
    with pytest.raises(ContextFault, match="checkpoint_durability_unregistered"):
        item["engine"].checkpoint(
            item["cap"]("durability"), "unregistered-checkpoint"
        )


def test_local_ephemeral_checkpoint_expires_without_remote_deletion_proof(tmp_path):
    item = _keyed_cycle(tmp_path, durability_mode="local_ephemeral")
    checkpoint, _ = item["seal"]()
    assert "durability_mode" not in checkpoint["receipt"]["payload"]
    with item["engine"].store.connection() as db:
        assert db.execute(
            "SELECT checkpoint_durability_mode FROM cycles WHERE id=?",
            (item["namespace"].context_cycle_id,),
        ).fetchone()[0] == "local_ephemeral"
    item["clock"][0] = 1061
    with pytest.raises(ContextFault, match="remote_deletion_proof_mismatch"):
        item["engine"].retention(
            item["cap"]("durability"), item["expire_command"](checkpoint)
        )
    key_ref = checkpoint["receipt"]["payload"]["key_ref"]
    item["provider"].cipher(key_ref, item["namespace"].model_dump())

    status = item["engine"].status(item["cap"]("durability"))["payload"]
    command = item["authority"].sign(
        RetentionCommandV1(
            operation="expire",
            namespace=item["namespace"],
            idempotency_key="expire-local-ephemeral",
            expected_revision=status["revision"],
            reason_code="policy_expiry",
            remote_deletion_receipt=None,
            checkpoint_object_ref_digest=None,
            issued_at=item["clock"][0],
            expires_at=item["clock"][0] + 100,
        ).model_dump(mode="json")
    )
    deleted = item["engine"].retention(item["cap"]("durability"), command)
    assert deleted["payload"]["cryptographic_erasure"] is True
    assert deleted["payload"]["remote_deletion_receipt_digest"] is None
    assert deleted["payload"]["checkpoint_durability_mode"] == "local_ephemeral"
    assert (
        deleted["payload"]["checkpoint_manifest_checksum"]
        == checkpoint["receipt"]["payload"]["manifest_checksum"]
    )


def test_remote_registered_checkpoint_still_requires_deletion_before_key_destroy(tmp_path):
    item = _keyed_cycle(tmp_path, durability_mode="remote_registered")
    checkpoint, _ = item["seal"]()
    item["clock"][0] = 1061
    status = item["engine"].status(item["cap"]("durability"))["payload"]
    missing = item["authority"].sign(
        RetentionCommandV1(
            operation="expire",
            namespace=item["namespace"],
            idempotency_key="expire-without-remote-proof",
            expected_revision=status["revision"],
            reason_code="policy_expiry",
            remote_deletion_receipt=None,
            checkpoint_object_ref_digest=None,
            issued_at=item["clock"][0],
            expires_at=item["clock"][0] + 100,
        ).model_dump(mode="json")
    )
    with pytest.raises(ContextFault, match="remote_deletion_proof_required"):
        item["engine"].retention(item["cap"]("durability"), missing)
    item["provider"].cipher(
        checkpoint["receipt"]["payload"]["key_ref"],
        item["namespace"].model_dump(),
    )


def test_v1_legacy_binding_and_checkpoint_round_trip_without_default_digest_rewrite(
    tmp_path,
):
    item = _keyed_cycle(tmp_path, keyed=False, legacy_binding=True)
    engine, cap = item["engine"], item["cap"]
    engine.append(
        cap(),
        AppendRequestV1(
            idempotency_key="legacy-record",
            text="needle legacy checkpoint",
            action_ref="legacy-test",
        ),
    )
    checkpoint = engine.checkpoint(cap("durability"), "legacy-checkpoint")
    namespace = item["namespace"].model_dump()
    pack = base64.b64decode(checkpoint["pack"], validate=True)
    snapshot = item["global_cipher"].decrypt(
        pack,
        {"namespace": namespace, "domain": "checkpoint/v1"},
        item["profile"].max_checkpoint_bytes,
    )
    snapshot.pop("key_ref")
    snapshot.pop("key_provider")
    snapshot.pop("key_version")
    snapshot.pop("retention_class")
    snapshot.pop("checkpoint_durability_mode")
    snapshot.pop("checkpoint_durability_command_digest")
    snapshot.pop("checkpoint_durability_receipt")
    legacy_pack = item["global_cipher"].encrypt(
        snapshot, {"namespace": namespace, "domain": "checkpoint/v1"}
    )
    legacy_claim = checkpoint["receipt"]["payload"].copy()
    legacy_claim.pop("key_ref")
    legacy_claim.pop("key_provider")
    legacy_claim.pop("key_version")
    legacy_claim["manifest_checksum"] = hashlib.sha256(legacy_pack).hexdigest()
    legacy_claim["bytes"] = len(legacy_pack)
    legacy_receipt = item["service"].sign(legacy_claim)
    replacement = item["make"](tmp_path / "legacy-replacement.sqlite3")
    replacement.hydrate(
        item["authority"].sign(
            {
                "operation": "hydrate",
                "namespace": namespace,
                "manifest_checksum": legacy_claim["manifest_checksum"],
                "fence": 2,
                "expires_at": 1900,
            }
        ),
        legacy_receipt,
        legacy_pack,
        {item["service"].key_id: item["service"].key.public_key()},
    )
    recalled = replacement.retrieve(
        cap(fence=2),
        RetrieveRequestV1(
            turn_id="legacy-recall",
            query="needle",
            native_window=64000,
            remaining_prompt_budget=12000,
        ),
    )
    assert recalled["payload"]["entries"][0]["text"] == "needle legacy checkpoint"
    with pytest.raises(ContextFault, match="profile_mismatch"):
        ContextEngine(
            engine.store.path,
            item["profile"].model_copy(update={"max_records": 101}),
            item["global_cipher"],
            item["service"],
            {item["authority"].key_id: item["authority"].key.public_key()},
            {item["proof"].key_id: item["proof"].key.public_key()},
            clock=lambda: item["clock"][0],
        )


def test_signed_hold_release_and_due_selection(tmp_path):
    item = _keyed_cycle(tmp_path)
    checkpoint, _ = item["seal"]()
    engine, cap, authority, namespace, clock = (
        item["engine"],
        item["cap"],
        item["authority"],
        item["namespace"],
        item["clock"],
    )

    def command(operation, reason, key):
        revision = engine.status(cap("durability"))["payload"]["revision"]
        return authority.sign(
            {
                "schema_version": "ContextRetentionCommandV1",
                "operation": operation,
                "namespace": namespace.model_dump(),
                "idempotency_key": key,
                "expected_revision": revision,
                "reason_code": reason,
                "remote_deletion_receipt": None,
                "checkpoint_object_ref_digest": None,
                "issued_at": clock[0],
                "expires_at": clock[0] + 100,
            }
        )

    held = ContextService(engine).dispatch(
        {
            "operation": "retention",
            "capability": cap("durability"),
            "command": command("hold", "legal_hold", "hold-1"),
        }
    )
    assert held["payload"]["operation"] == "hold"
    assert engine.retention_manager.due(before=2000) == []
    clock[0] = 1061
    released = engine.retention(
        cap("durability"), command("release", "hold_released", "release-1")
    )
    assert released["payload"]["operation"] == "release"
    assert engine.retention_manager.due()[0]["context_cycle_id"] == namespace.context_cycle_id
    assert checkpoint["receipt"]["payload"]["key_ref"]


def test_hold_and_release_lost_ack_replay_survives_later_state_and_restart(tmp_path):
    item = _keyed_cycle(tmp_path)
    item["seal"]()
    engine, cap, authority, namespace, clock = (
        item["engine"],
        item["cap"],
        item["authority"],
        item["namespace"],
        item["clock"],
    )

    def command(operation, reason, key, revision):
        return authority.sign(
            RetentionCommandV1(
                operation=operation,
                namespace=namespace,
                idempotency_key=key,
                expected_revision=revision,
                reason_code=reason,
                remote_deletion_receipt=None,
                checkpoint_object_ref_digest=None,
                issued_at=clock[0],
                expires_at=clock[0] + 100,
            ).model_dump(mode="json")
        )

    revision = engine.status(cap("durability"))["payload"]["revision"]
    hold_command = command("hold", "legal_hold", "hold-lost-ack", revision)
    held = engine.retention(cap("durability"), hold_command)
    release_command = command(
        "release", "hold_released", "release-lost-ack", held["payload"]["revision"]
    )
    released = engine.retention(cap("durability"), release_command)
    newer_hold_command = command(
        "hold", "incident_hold", "hold-after-release", released["payload"]["revision"]
    )
    engine.retention(cap("durability"), newer_hold_command)

    # Reconciliation happens after the original authority windows close and
    # after the current hold state/revision have moved on.
    clock[0] += 101
    assert engine.retention(cap("durability"), hold_command) == held
    assert engine.retention(cap("durability"), release_command) == released
    reopened = item["make"](engine.store.path)
    assert reopened.retention(cap("durability"), hold_command) == held
    assert reopened.retention(cap("durability"), release_command) == released


@pytest.mark.parametrize("operation,reason", [("hold", "legal_hold"), ("release", "hold_released")])
def test_hold_release_replay_rejects_tampered_persisted_result(
    tmp_path, operation, reason
):
    item = _keyed_cycle(tmp_path)
    item["seal"]()
    engine, cap, authority, namespace = (
        item["engine"],
        item["cap"],
        item["authority"],
        item["namespace"],
    )
    revision = engine.status(cap("durability"))["payload"]["revision"]
    if operation == "release":
        held = engine.retention(
            cap("durability"),
            authority.sign(
                RetentionCommandV1(
                    operation="hold",
                    namespace=namespace,
                    idempotency_key="hold-before-tamper",
                    expected_revision=revision,
                    reason_code="legal_hold",
                    remote_deletion_receipt=None,
                    checkpoint_object_ref_digest=None,
                    issued_at=item["clock"][0],
                    expires_at=item["clock"][0] + 100,
                ).model_dump(mode="json")
            ),
        )
        revision = held["payload"]["revision"]
    command = authority.sign(
        RetentionCommandV1(
            operation=operation,
            namespace=namespace,
            idempotency_key=f"{operation}-tampered-result",
            expected_revision=revision,
            reason_code=reason,
            remote_deletion_receipt=None,
            checkpoint_object_ref_digest=None,
            issued_at=item["clock"][0],
            expires_at=item["clock"][0] + 100,
        ).model_dump(mode="json")
    )
    engine.retention(cap("durability"), command)
    with engine.store.transaction() as db:
        stored = json.loads(
            db.execute(
                "SELECT result FROM retention_operations WHERE cycle=? AND operation=? "
                "AND key=?",
                (namespace.context_cycle_id, operation, f"{operation}-tampered-result"),
            ).fetchone()[0]
        )
        stored["signature"] = "A" * len(stored["signature"])
        db.execute(
            "UPDATE retention_operations SET result=? WHERE cycle=? AND operation=? "
            "AND key=?",
            (
                canonical(stored).decode(),
                namespace.context_cycle_id,
                operation,
                f"{operation}-tampered-result",
            ),
        )
    reopened = item["make"](engine.store.path)
    with pytest.raises(ContextFault, match="integrity_failure"):
        reopened.retention(cap("durability"), command)


def test_stale_release_cannot_clear_a_newer_incident_hold(tmp_path):
    item = _keyed_cycle(tmp_path)
    item["seal"]()
    engine, cap, authority, namespace = (
        item["engine"],
        item["cap"],
        item["authority"],
        item["namespace"],
    )

    def command(operation, reason, key, revision):
        return authority.sign(
            {
                "schema_version": "ContextRetentionCommandV1",
                "operation": operation,
                "namespace": namespace.model_dump(),
                "idempotency_key": key,
                "expected_revision": revision,
                "reason_code": reason,
                "remote_deletion_receipt": None,
                "checkpoint_object_ref_digest": None,
                "issued_at": item["clock"][0],
                "expires_at": item["clock"][0] + 100,
            }
        )

    original = engine.status(cap("durability"))["payload"]["revision"]
    held = engine.retention(
        cap("durability"), command("hold", "legal_hold", "hold-legal", original)
    )
    stale_release = command(
        "release", "hold_released", "release-stale", held["payload"]["revision"]
    )
    engine.retention(
        cap("durability"),
        command(
            "hold",
            "incident_hold",
            "hold-incident",
            held["payload"]["revision"],
        ),
    )
    with pytest.raises(ContextFault, match="stale_state"):
        engine.retention(cap("durability"), stale_release)
    status = engine.status(cap("durability"))["payload"]
    assert status["retention_hold"] is True
    assert status["retention_hold_reason"] == "incident_hold"


@pytest.mark.parametrize("managed,expected", [(True, True), (False, False)])
def test_secure_expiry_is_idempotent_and_file_provider_never_claims_production_erasure(
    tmp_path, managed, expected
):
    item = _keyed_cycle(tmp_path, managed=managed)
    engine, cap, clock = item["engine"], item["cap"], item["clock"]
    engine.append(
        cap(),
        AppendRequestV1(
            idempotency_key="retained-record",
            text="retained encrypted evidence",
            action_ref="retention-test",
        ),
    )
    checkpoint, _ = item["seal"]()
    clock[0] = 1061
    command = item["expire_command"](checkpoint)
    receipt = engine.retention(cap("durability"), command)
    assert receipt["payload"]["cryptographic_erasure"] is expected
    assert receipt["payload"]["remote_deletion_receipt_digest"] == digest(
        command["payload"]["remote_deletion_receipt"]
    )
    assert engine.retention(cap("durability"), command) == receipt
    if managed:
        engine.managed_key_provider_receipts = ()
        assert engine.retention(cap("durability"), command) == receipt
        item["provider_receipts"].clear()
        reopened = item["make"](engine.store.path)
        assert reopened.retention(cap("durability"), command) == receipt
    with pytest.raises(ContextFault, match="key_destroyed"):
        item["provider"].cipher(
            checkpoint["receipt"]["payload"]["key_ref"],
            item["namespace"].model_dump(),
        )
    status = engine.status(cap())
    assert status["payload"]["state"] == "EXPIRED"
    assert status["payload"]["cleanup_state"] == "COMPLETE"
    with engine.store.connection() as db:
        assert db.execute("SELECT count(*) FROM segments").fetchone()[0] == 0


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("tamper_target", ["operation", "cycle"])
def test_deletion_replay_rejects_tampered_v1_v2_persisted_evidence_after_restart(
    tmp_path, modern, tamper_target
):
    item = _keyed_cycle(
        tmp_path,
        managed=modern,
        register_durability=modern,
    )
    checkpoint, _ = item["seal"]()
    item["clock"][0] = 1061
    command = item["expire_command"](checkpoint)
    receipt = item["engine"].retention(item["cap"]("durability"), command)
    assert receipt["payload"]["schema_version"] == (
        "ContextDeletionReceiptV2" if modern else "ContextDeletionReceiptV1"
    )
    with item["engine"].store.transaction() as db:
        stored = json.loads(
            db.execute(
                "SELECT result FROM retention_operations WHERE cycle=? "
                "AND operation='expire' AND key=?",
                (
                    item["namespace"].context_cycle_id,
                    command["payload"]["idempotency_key"],
                ),
            ).fetchone()[0]
        )
        stored["signature"] = "A" * len(stored["signature"])
        if tamper_target == "operation":
            db.execute(
                "UPDATE retention_operations SET result=? WHERE cycle=? "
                "AND operation='expire' AND key=?",
                (
                    canonical(stored).decode(),
                    item["namespace"].context_cycle_id,
                    command["payload"]["idempotency_key"],
                ),
            )
        else:
            db.execute(
                "UPDATE cycles SET deletion_receipt=? WHERE id=?",
                (
                    canonical(stored).decode(),
                    item["namespace"].context_cycle_id,
                ),
            )
    reopened = item["make"](item["engine"].store.path)
    with pytest.raises(ContextFault, match="integrity_failure"):
        reopened.retention(item["cap"]("durability"), command)


def test_deletion_v2_rejects_incomplete_or_contradictory_erasure_claims(tmp_path):
    item = _keyed_cycle(tmp_path)
    checkpoint, _ = item["seal"]()
    item["clock"][0] = 1061
    valid = item["engine"].retention(
        item["cap"]("durability"), item["expire_command"](checkpoint)
    )["payload"]
    assert ContextDeletionReceiptV2.model_validate(valid)

    without_managed_proof = {
        **valid,
        "key_destruction_receipt": None,
        "key_destruction_receipt_digest": None,
        "managed_key_provider_receipt_digest": None,
    }
    with pytest.raises(ValueError, match="cryptographic erasure"):
        ContextDeletionReceiptV2.model_validate(without_managed_proof)

    incomplete = {**valid, "key_destruction_receipt_digest": None}
    with pytest.raises(ValueError, match="must be complete"):
        ContextDeletionReceiptV2.model_validate(incomplete)

    mismatched = {**valid, "key_destruction_receipt_digest": digest("wrong")}
    with pytest.raises(ValueError, match="receipt digest mismatch"):
        ContextDeletionReceiptV2.model_validate(mismatched)

    contradictory = {
        **valid,
        "local_key_destruction": {
            "schema_version": "ContextLocalKeyDestructionV1",
            "provider": "file-wrapped-dek/v1",
            "key_ref": valid["key_destruction_receipt"]["payload"]["key_ref"],
            "key_version": valid["key_destruction_receipt"]["payload"]["key_version"],
            "namespace_digest": digest(item["namespace"]),
            "wrapped_key_digest": digest("local tombstone"),
            "destroyed_at": 1061,
            "verified": True,
        },
    }
    with pytest.raises(ValueError, match="exclusive"):
        ContextDeletionReceiptV2.model_validate(contradictory)

    false_with_managed_proof = {**valid, "cryptographic_erasure": False}
    with pytest.raises(ValueError, match="cryptographic erasure"):
        ContextDeletionReceiptV2.model_validate(false_with_managed_proof)


def test_remote_deletion_must_match_before_key_destruction(tmp_path):
    item = _keyed_cycle(tmp_path)
    checkpoint, _ = item["seal"]()
    item["clock"][0] = 1061
    command = item["expire_command"](checkpoint, manifest="f" * 64)
    with pytest.raises(ContextFault, match="remote_deletion_proof_mismatch"):
        item["engine"].retention(item["cap"]("durability"), command)
    key_ref = checkpoint["receipt"]["payload"]["key_ref"]
    item["provider"].cipher(key_ref, item["namespace"].model_dump())
    with item["engine"].store.connection() as db:
        assert db.execute("SELECT cleanup_state FROM cycles").fetchone()[0] == "NONE"


def test_managed_key_self_assertion_cannot_claim_erasure(tmp_path):
    item = _keyed_cycle(tmp_path)
    checkpoint, _ = item["seal"]()
    item["clock"][0] = 1061
    item["provider"].destroy = lambda *_: {"verified": True, "destroyed": True}
    with pytest.raises(ContextFault, match="key_destruction_unverified"):
        item["engine"].retention(
            item["cap"]("durability"), item["expire_command"](checkpoint)
        )
    key_ref = checkpoint["receipt"]["payload"]["key_ref"]
    item["provider"].cipher(key_ref, item["namespace"].model_dump())


def test_managed_key_destruction_receipt_binds_exact_key_version(tmp_path):
    item = _keyed_cycle(tmp_path)
    checkpoint, _ = item["seal"]()
    item["clock"][0] = 1061
    key_ref = checkpoint["receipt"]["payload"]["key_ref"]
    item["provider"].destroy = lambda *_: item["destruction"].sign(
        {
            "schema_version": "ContextKeyDestructionReceiptV1",
            "provider": item["provider"].provider,
            "key_ref": key_ref,
            "namespace_digest": digest(item["namespace"].model_dump()),
            "key_version": "wrong-key-version/v1",
            "destroyed": True,
            "destroyed_at": 1061,
            "issued_at": 1061,
            "expires_at": 1161,
        }
    )
    with pytest.raises(ContextFault, match="key_destruction_unverified"):
        item["engine"].retention(
            item["cap"]("durability"), item["expire_command"](checkpoint)
        )


def test_cleanup_recovers_after_crash_between_key_destroy_and_sql_cleanup(tmp_path):
    item = _keyed_cycle(tmp_path)
    checkpoint, _ = item["seal"]()
    item["clock"][0] = 1061
    command = item["expire_command"](checkpoint)

    def crash():
        raise RuntimeError("simulated process loss")

    item["engine"].retention_manager = RetentionManager(
        item["engine"], after_key_destroy=crash
    )
    with pytest.raises(RuntimeError, match="simulated process loss"):
        item["engine"].retention(item["cap"]("durability"), command)
    with item["engine"].store.connection() as db:
        pending = db.execute(
            "SELECT cleanup_state,cleanup_operation_key,cleanup_request_digest FROM cycles"
        ).fetchone()
        assert tuple(pending) == (
            "PENDING",
            command["payload"]["idempotency_key"],
            digest(command["payload"]),
        )
    item["clock"][0] = 1200  # Both original signed proofs have expired on this retry.
    item["engine"].retention_manager = RetentionManager(item["engine"])
    current_revision = item["engine"].status(item["cap"]("durability"))["payload"][
        "revision"
    ]
    competing_payload = dict(command["payload"])
    competing_payload.update(
        idempotency_key="expire-retention-competing",
        expected_revision=current_revision,
        issued_at=1200,
        expires_at=1300,
    )
    competing = item["authority"].sign(competing_payload)
    with pytest.raises(ContextFault, match="cleanup_in_progress"):
        item["engine"].retention(item["cap"]("durability"), competing)
    with item["engine"].store.connection() as db:
        assert (
            db.execute(
                "SELECT count(*) FROM retention_operations WHERE operation='expire'"
            ).fetchone()[0]
            == 1
        )
    receipt = item["engine"].retention(item["cap"]("durability"), command)
    assert receipt["payload"]["cryptographic_erasure"] is True
    assert item["engine"].retention(item["cap"]("durability"), command) == receipt


def test_aborted_cycle_gets_post_terminal_retention_without_remote_proof_if_never_checkpointed(
    tmp_path,
):
    item = _keyed_cycle(tmp_path)
    status = item["engine"].status(item["cap"]())
    item["engine"].control(
        item["authority"].sign(
            {
                "operation": "abort",
                "context_cycle_id": item["namespace"].context_cycle_id,
                "binding_digest": item["cap"]()["payload"]["binding_digest"],
                "expected_revision": status["payload"]["revision"],
                "fence": status["payload"]["fence"],
                "expires_at": 1100,
            }
        )
    )
    aborted = item["engine"].status(item["cap"]())["payload"]
    assert aborted["state"] == "ABORTED"
    assert aborted["expires_at"] == 1060
    item["clock"][0] = 1061
    command = item["authority"].sign(
        {
            "schema_version": "ContextRetentionCommandV1",
            "operation": "expire",
            "namespace": item["namespace"].model_dump(),
            "idempotency_key": "expire-aborted",
            "expected_revision": aborted["revision"],
            "reason_code": "policy_expiry",
            "remote_deletion_receipt": None,
            "checkpoint_object_ref_digest": None,
            "issued_at": 1061,
            "expires_at": 1100,
        }
    )
    deleted = item["engine"].retention(item["cap"]("durability"), command)
    assert deleted["payload"]["cryptographic_erasure"] is True
    assert deleted["payload"]["remote_deletion_receipt_digest"] is None


def test_unregistered_aborted_cycle_without_checkpoint_can_be_cleaned_up(tmp_path):
    item = _keyed_cycle(
        tmp_path, register_durability=False, reservation_protocol_version=2
    )
    status = item["engine"].status(item["cap"]())["payload"]
    item["engine"].control(
        item["authority"].sign(
            {
                "operation": "abort",
                "context_cycle_id": item["namespace"].context_cycle_id,
                "binding_digest": item["binding_digest"],
                "expected_revision": status["revision"],
                "fence": status["fence"],
                "expires_at": 1100,
            }
        )
    )
    aborted = item["engine"].status(item["cap"]())["payload"]
    item["clock"][0] = aborted["expires_at"] + 1
    command = item["authority"].sign(
        RetentionCommandV1(
            operation="expire",
            namespace=item["namespace"],
            idempotency_key="expire-unregistered-abort",
            expected_revision=aborted["revision"],
            reason_code="policy_expiry",
            remote_deletion_receipt=None,
            checkpoint_object_ref_digest=None,
            issued_at=item["clock"][0],
            expires_at=item["clock"][0] + 100,
        ).model_dump(mode="json")
    )
    deleted = item["engine"].retention(item["cap"]("durability"), command)
    assert deleted["payload"]["schema_version"] == "ContextDeletionReceiptV2"
    assert deleted["payload"]["checkpoint_durability_mode"] == "unregistered"
    assert deleted["payload"]["checkpoint_manifest_checksum"] is None
    assert deleted["payload"]["remote_deletion_receipt_digest"] is None
    inconsistent = {
        **deleted["payload"],
        "checkpoint_durability_mode": "remote_registered",
        "checkpoint_durability_registration_receipt_digest": digest("registration"),
        "remote_deletion_receipt_digest": digest("remote deletion"),
        "checkpoint_object_ref_digest": digest("remote object"),
    }
    with pytest.raises(
        ValueError, match="without a checkpoint cannot contain deletion evidence"
    ):
        ContextDeletionReceiptV2.model_validate(inconsistent)


def test_keyed_checkpoint_hydrates_on_replacement_using_opaque_reference(tmp_path):
    item = _keyed_cycle(tmp_path)
    engine, cap, namespace = item["engine"], item["cap"], item["namespace"]
    engine.append(
        cap(),
        AppendRequestV1(
            idempotency_key="host-replacement",
            text="needle replacement evidence",
            action_ref="replacement-test",
        ),
    )
    checkpoint = engine.checkpoint(cap("durability"), "host-replacement-checkpoint")
    claim = checkpoint["receipt"]["payload"]
    replacement_signer = ReceiptSigner(
        "contextd-replacement", Ed25519PrivateKey.generate()
    )
    replacement = ContextEngine(
        tmp_path / "replacement.sqlite3",
        item["profile"],
        item["global_cipher"],
        replacement_signer,
        {item["authority"].key_id: item["authority"].key.public_key()},
        {item["proof"].key_id: item["proof"].key.public_key()},
        clock=lambda: item["clock"][0],
        cycle_keys=item["provider"],
        remote_deletion_keys={
            item["deletion"].key_id: item["deletion"].key.public_key()
        },
        managed_key_provider_receipts=item["provider_receipts"],
        key_provider_keys={
            item["provider_authority"].key_id: item[
                "provider_authority"
            ].key.public_key()
        },
        key_destruction_keys={
            item["destruction"].key_id: item["destruction"].key.public_key()
        },
    )
    grant = item["authority"].sign(
        {
            "operation": "hydrate",
            "namespace": namespace.model_dump(),
            "manifest_checksum": claim["manifest_checksum"],
            "key_ref": claim["key_ref"],
            "key_provider": claim["key_provider"],
            "key_version": claim["key_version"],
            "fence": 2,
            "expires_at": 1900,
        }
    )
    hydrate = replacement.hydrate(
        grant,
        checkpoint["receipt"],
        base64.b64decode(checkpoint["pack"], validate=True),
        {engine.signer.key_id: engine.signer.key.public_key()},
    )
    hydrate_payload = hydrate["payload"]
    assert hydrate_payload["schema_version"] == "ContextHydrateReceiptV1"
    assert hydrate_payload["manifest_checksum"] == claim["manifest_checksum"]
    assert hydrate_payload["checkpoint_number"] == claim["checkpoint_number"]
    assert hydrate_payload["cursor"] == claim["cursor"]
    assert hydrate_payload["key_ref"] == claim["key_ref"]
    assert hydrate_payload["key_version"] == claim["key_version"]
    assert hydrate_payload["replay_digest"]
    with replacement.store.connection() as db:
        durability = db.execute(
            "SELECT checkpoint_durability_mode,checkpoint_durability_legacy,"
            "checkpoint_durability_command_digest,checkpoint_durability_receipt "
            "FROM cycles"
        ).fetchone()
    assert durability["checkpoint_durability_mode"] == "remote_registered"
    assert durability["checkpoint_durability_legacy"] == 0
    assert durability["checkpoint_durability_command_digest"] == digest(
        item["durability_command"]["payload"]
    )
    durability_envelope = json.loads(durability["checkpoint_durability_receipt"])
    durability_payload = verify(
        durability_envelope,
        {replacement.signer.key_id: replacement.signer.key.public_key()},
    )
    durability_claim = ContextCheckpointDurabilityReceiptV2.model_validate(
        durability_payload
    )
    assert durability_claim.source_registration_receipt_digest == digest(
        item["durability_receipt"]
    )
    assert (
        replacement.hydrate(
            grant,
            checkpoint["receipt"],
            base64.b64decode(checkpoint["pack"], validate=True),
            {engine.signer.key_id: engine.signer.key.public_key()},
        )
        == hydrate
    )
    with replacement.store.transaction() as db:
        db.execute("DELETE FROM postings")
    with pytest.raises(ContextFault, match="hydrate_conflict"):
        replacement.hydrate(
            grant,
            checkpoint["receipt"],
            base64.b64decode(checkpoint["pack"], validate=True),
            {engine.signer.key_id: engine.signer.key.public_key()},
        )
    empty = replacement.retrieve(
        cap(fence=2),
        RetrieveRequestV1(
            turn_id="before-rebuild",
            query="needle",
            native_window=64000,
            remaining_prompt_budget=12000,
        ),
    )
    assert empty["payload"]["entries"] == []
    rebuilt = replacement.rebuild_index(
        cap("durability", fence=2), "replacement-rebuild"
    )
    assert rebuilt["payload"]["records"] == 1
    capsule = replacement.retrieve(
        cap(fence=2),
        RetrieveRequestV1(
            turn_id="replacement-turn",
            query="needle",
            native_window=64000,
            remaining_prompt_budget=12000,
        ),
    )
    assert capsule["payload"]["entries"][0]["text"] == "needle replacement evidence"
    active = replacement.status(cap("durability", fence=2))["payload"]
    replacement.control(
        item["authority"].sign(
            {
                "operation": "abort",
                "context_cycle_id": namespace.context_cycle_id,
                "binding_digest": item["binding_digest"],
                "expected_revision": active["revision"],
                "fence": active["fence"],
                "expires_at": 1100,
            }
        )
    )
    aborted = replacement.status(cap("durability", fence=2))["payload"]
    item["clock"][0] = aborted["expires_at"] + 1
    object_ref = digest("spaces://restored-checkpoint")
    remote = item["deletion"].sign(
        {
            "schema_version": "ContextRemoteDeletionReceiptV1",
            "namespace": namespace.model_dump(),
            "manifest_checksum": claim["manifest_checksum"],
            "object_ref_digest": object_ref,
            "deleted": True,
            "issued_at": item["clock"][0],
            "expires_at": item["clock"][0] + 100,
        }
    )
    retention_command = item["authority"].sign(
        RetentionCommandV1(
            operation="expire",
            namespace=namespace,
            idempotency_key="expire-restored-checkpoint",
            expected_revision=aborted["revision"],
            reason_code="policy_expiry",
            remote_deletion_receipt=remote,
            checkpoint_object_ref_digest=object_ref,
            issued_at=item["clock"][0],
            expires_at=item["clock"][0] + 100,
        ).model_dump(mode="json")
    )
    deletion = replacement.retention(
        cap("durability", fence=2), retention_command
    )
    assert deletion["payload"]["schema_version"] == "ContextDeletionReceiptV2"
    assert deletion["payload"]["remote_deletion_receipt_digest"] == digest(remote)
    assert deletion["payload"]["checkpoint_manifest_checksum"] == claim[
        "manifest_checksum"
    ]
    assert deletion["payload"][
        "checkpoint_durability_registration_receipt_digest"
    ] == digest(durability_envelope)
    without_registration = {
        **deletion["payload"],
        "checkpoint_durability_registration_receipt_digest": None,
    }
    with pytest.raises(ValueError, match="requires signed durability registration"):
        ContextDeletionReceiptV2.model_validate(without_registration)


def test_v2_hydrate_command_is_source_bound_and_receipt_is_exact(
    tmp_path, monkeypatch
):
    build = ReceiptSigner("hydrate-build", Ed25519PrivateKey.generate())
    source_revision = "c" * 40
    dependencies = [
        "annotated-types",
        "cffi",
        "cryptography",
        "numpy",
        "pycparser",
        "pydantic",
        "pydantic_core",
        "typing-inspection",
        "typing_extensions",
    ]
    environment = RuntimeEnvironmentV2(
        python_implementation="cpython",
        python_version="3.13.0",
        python_abi="cp313-test",
        python_executable_digest=digest("hydrate-python"),
        dependency_import_closure=dependencies,
        dependency_versions={name: "1.0" for name in dependencies},
        dependency_artifact_digests={name: digest(name) for name in dependencies},
        sqlite_version=sqlite3.sqlite_version,
        package_import_root_digest=digest("hydrate-site-packages"),
    )
    package_digest = digest("hydrate-package-tree")
    build_manifest = build.sign(
        {
            "schema_version": "ContextRuntimeBuildManifestV3",
            "distribution": "aether-context",
            "version": __version__,
            "source_revision": source_revision,
            "package_tree_digest": package_digest,
            "wheel_digest": digest("hydrate-wheel"),
            "recall_dataset_digest": digest("hydrate-recall-dataset"),
            "data_plane_case_generator_digest": digest("hydrate-case-generator"),
            "built_at": 900,
            "runtime_environment": environment.model_dump(mode="json"),
            "locked_environment_digest": digest(environment),
        }
    )
    monkeypatch.setattr(
        "aether_context.engine.runtime_environment_identity",
        lambda **_: environment,
    )
    monkeypatch.setattr(
        "aether_context.engine.runtime_package_tree_digest",
        lambda **_: package_digest,
    )
    item = _keyed_cycle(
        tmp_path,
        runtime_build_manifest=build_manifest,
        runtime_build_keys={build.key_id: build.key.public_key()},
    )
    environment_measurements = 0
    package_measurements = 0

    def measured_environment(**_):
        nonlocal environment_measurements
        environment_measurements += 1
        return environment

    def measured_package(**_):
        nonlocal package_measurements
        package_measurements += 1
        return package_digest

    monkeypatch.setattr(
        "aether_context.engine.runtime_environment_identity", measured_environment
    )
    monkeypatch.setattr(
        "aether_context.engine.runtime_package_tree_digest", measured_package
    )
    item["engine"].append(
        item["cap"](),
        AppendRequestV1(
            idempotency_key="v2-hot-path",
            text="runtime identity remains cached on the write path",
            action_ref="v2-hot-path",
        ),
    )
    checkpoint = item["engine"].checkpoint(
        item["cap"]("durability"), "v2-hydrate-checkpoint"
    )
    item["engine"].status(item["cap"]("durability"))
    item["engine"].retrieve(
        item["cap"](),
        RetrieveRequestV1(
            turn_id="v2-hot-path-cached-retrieve",
            query="cached runtime identity",
            native_window=32_768,
            remaining_prompt_budget=4_096,
        ),
    )
    assert (environment_measurements, package_measurements) == (0, 0)
    item["clock"][0] += 61
    item["engine"].retrieve(
        item["cap"](),
        RetrieveRequestV1(
            turn_id="v2-hot-path-refreshed-retrieve",
            query="refreshed runtime identity",
            native_window=32_768,
            remaining_prompt_budget=4_096,
        ),
    )
    assert (environment_measurements, package_measurements) == (1, 1)
    assert item["engine"].health_v2()["runtime_build_ready"] is True
    assert (environment_measurements, package_measurements) == (2, 2)
    item["engine"].reserve_v2(
        item["authority"].sign(
            ContextReservationCommandV2(
                owner_id="hot-path-owner",
                project_id="hot-path-project",
                idempotency_key="hot-path-admission",
                request_digest=digest("hot-path-admission"),
                profile_digest=digest(item["profile"]),
                expires_at=item["clock"][0] + 100,
            ).model_dump(mode="json")
        )
    )
    assert (environment_measurements, package_measurements) == (3, 3)
    claim = checkpoint["receipt"]["payload"]
    command_payload = ContextHydrateCommandV2(
        namespace=item["namespace"],
        manifest_checksum=claim["manifest_checksum"],
        key_ref=claim["key_ref"],
        key_provider=claim["key_provider"],
        key_version=claim["key_version"],
        fence=2,
        expected_source_revision=source_revision,
        issued_at=item["clock"][0],
        expires_at=item["clock"][0] + 100,
    ).model_dump(mode="json")
    command = item["authority"].sign(command_payload)
    replacement = item["make"](tmp_path / "v2-hydrate-replacement.sqlite3")
    checkpoint_keys = {
        item["service"].key_id: item["service"].key.public_key()
    }
    hydrate_request = {
        "operation": "hydrate",
        "grant": command,
        "checkpoint_receipt": checkpoint["receipt"],
        "pack": checkpoint["pack"],
    }
    replacement_service = ContextService(
        replacement, checkpoint_receipt_keys=checkpoint_keys
    )
    receipt = replacement_service.dispatch(hydrate_request)
    with replacement.store.connection() as db:
        restored = db.execute(
            "SELECT checkpoint_durability_mode,checkpoint_durability_legacy,"
            "checkpoint_durability_command_digest FROM cycles"
        ).fetchone()
    assert tuple(restored) == (
        "remote_registered",
        0,
        digest(item["durability_command"]["payload"]),
    )
    payload = verify(
        receipt,
        {replacement.signer.key_id: replacement.signer.key.public_key()},
    )
    assert ContextHydrateReceiptV2.model_validate(payload).model_dump(
        mode="json"
    ) == payload
    assert payload["hydrate_command_digest"] == digest(command_payload)
    assert payload["expected_source_revision"] == source_revision
    wrong_key = {
        **command_payload,
        "key_ref": "wrong-key-ref",
        "key_provider": "wrong-provider/v1",
        "key_version": "wrong-key-version",
    }
    with pytest.raises(ContextFault, match="hydrate_grant_invalid"):
        replacement.hydrate(
            item["authority"].sign(wrong_key),
            checkpoint["receipt"],
            base64.b64decode(checkpoint["pack"], validate=True),
            {item["service"].key_id: item["service"].key.public_key()},
        )
    assert replacement_service.dispatch(hydrate_request) == receipt

    # Lost acknowledgements reconcile against the persisted signed result even
    # after the original grant expires, including concurrent exact retries.
    item["clock"][0] += 101
    with ThreadPoolExecutor(max_workers=4) as executor:
        replayed = list(
            executor.map(
                lambda _: replacement_service.dispatch(hydrate_request),
                range(8),
            )
        )
    assert replayed == [receipt] * 8

    fresh_payload = {
        **command_payload,
        "issued_at": item["clock"][0],
        "expires_at": item["clock"][0] + 100,
    }
    with pytest.raises(ContextFault, match="hydrate_conflict"):
        replacement.hydrate(
            item["authority"].sign(fresh_payload),
            checkpoint["receipt"],
            base64.b64decode(checkpoint["pack"], validate=True),
            checkpoint_keys,
        )
    original_pack = base64.b64decode(checkpoint["pack"], validate=True)
    with pytest.raises(ContextFault, match="integrity_failure"):
        replacement.hydrate(
            command,
            checkpoint["receipt"],
            original_pack[:-1] + bytes([original_pack[-1] ^ 1]),
            checkpoint_keys,
        )
    empty_after_expiry = item["make"](
        tmp_path / "v2-hydrate-expired-first-attempt.sqlite3"
    )
    with pytest.raises(ContextFault, match="capability_denied"):
        empty_after_expiry.hydrate(
            command,
            checkpoint["receipt"],
            original_pack,
            checkpoint_keys,
        )

    # A replacement checkpoint embeds a V2 reattestation. The next host
    # verifies that immediate source but carries the same original signed V1
    # registration forward instead of growing a nested receipt chain.
    second_checkpoint = replacement.checkpoint(
        item["cap"]("durability", fence=2), "v2-second-hop-checkpoint"
    )
    second_claim = second_checkpoint["receipt"]["payload"]
    second_command_payload = ContextHydrateCommandV2(
        namespace=item["namespace"],
        manifest_checksum=second_claim["manifest_checksum"],
        key_ref=second_claim["key_ref"],
        key_provider=second_claim["key_provider"],
        key_version=second_claim["key_version"],
        fence=3,
        expected_source_revision=source_revision,
        issued_at=item["clock"][0],
        expires_at=item["clock"][0] + 100,
    ).model_dump(mode="json")
    second_request = {
        "operation": "hydrate",
        "grant": item["authority"].sign(second_command_payload),
        "checkpoint_receipt": second_checkpoint["receipt"],
        "pack": second_checkpoint["pack"],
    }
    second_replacement = item["make"](
        tmp_path / "v2-hydrate-second-replacement.sqlite3"
    )
    second_service = ContextService(
        second_replacement,
        checkpoint_receipt_keys={
            replacement.signer.key_id: replacement.signer.key.public_key()
        },
    )
    second_receipt = second_service.dispatch(second_request)
    second_payload = verify(
        second_receipt,
        {
            second_replacement.signer.key_id:
            second_replacement.signer.key.public_key()
        },
    )
    assert (
        second_payload["checkpoint_durability_registration_receipt"][
            "payload"
        ]["source_registration_receipt"]
        == item["durability_receipt"]
    )
    assert second_payload[
        "checkpoint_durability_registration_receipt"
    ]["payload"]["source_registration_receipt_digest"] == digest(
        item["durability_receipt"]
    )
    assert second_service.dispatch(second_request) == second_receipt

    active = second_replacement.status(
        item["cap"]("durability", fence=3)
    )["payload"]
    second_replacement.control(
        item["authority"].sign(
            {
                "operation": "abort",
                "context_cycle_id": item["namespace"].context_cycle_id,
                "binding_digest": item["binding_digest"],
                "expected_revision": active["revision"],
                "fence": active["fence"],
                "expires_at": item["clock"][0] + 100,
            }
        )
    )
    aborted = second_replacement.status(
        item["cap"]("durability", fence=3)
    )["payload"]
    item["clock"][0] = aborted["expires_at"] + 1
    object_ref = digest("spaces://context/v2-second-hop")
    remote = item["deletion"].sign(
        {
            "schema_version": "ContextRemoteDeletionReceiptV1",
            "namespace": item["namespace"].model_dump(),
            "manifest_checksum": second_claim["manifest_checksum"],
            "object_ref_digest": object_ref,
            "deleted": True,
            "issued_at": item["clock"][0],
            "expires_at": item["clock"][0] + 100,
        }
    )
    delete_command = item["authority"].sign(
        RetentionCommandV1(
            operation="expire",
            namespace=item["namespace"],
            idempotency_key="expire-v2-second-hop",
            expected_revision=aborted["revision"],
            reason_code="policy_expiry",
            remote_deletion_receipt=remote,
            checkpoint_object_ref_digest=object_ref,
            issued_at=item["clock"][0],
            expires_at=item["clock"][0] + 100,
        ).model_dump(mode="json")
    )
    deleted = second_replacement.retention(
        item["cap"]("durability", fence=3), delete_command
    )
    assert deleted["payload"][
        "checkpoint_durability_registration_receipt_digest"
    ] == digest(
        second_payload["checkpoint_durability_registration_receipt"]
    )
    assert second_replacement.retention(
        item["cap"]("durability", fence=3), delete_command
    ) == deleted

    measurements_before_expiry = (
        environment_measurements,
        package_measurements,
    )
    item["clock"][0] = 2001
    with pytest.raises(ContextFault, match="hosted_gate_unready"):
        item["engine"].retrieve(
            item["cap"](),
            RetrieveRequestV1(
                turn_id="v2-hot-path-expired-evidence",
                query="expired provider evidence",
                native_window=32_768,
                remaining_prompt_budget=4_096,
            ),
        )
    assert (environment_measurements, package_measurements) == (
        measurements_before_expiry[0] + 1,
        measurements_before_expiry[1] + 1,
    )


def test_managed_key_rotation_keeps_v1_and_v2_cycles_hydratable_and_expirable(
    tmp_path,
):
    item = _keyed_cycle(tmp_path)
    engine = item["engine"]
    engine.append(
        item["cap"](),
        AppendRequestV1(
            idempotency_key="rotation-v1-record",
            text="managed rotation version one evidence",
            action_ref="rotation-v1",
        ),
    )
    v1_checkpoint = engine.checkpoint(
        item["cap"]("durability"), "rotation-v1-checkpoint"
    )
    assert v1_checkpoint["receipt"]["payload"]["key_version"] == (
        "managed-test-key/v1"
    )

    item["provider"].active_key_version = "managed-test-key/v2"
    v2_provider_receipt = item["provider_authority"].sign(
        {
            "schema_version": "ContextManagedKeyProviderReceiptV1",
            "provider": item["provider"].provider,
            "key_version": "managed-test-key/v2",
            "shared_hydrate_supported": True,
            "verified_destruction_supported": True,
            "issued_at": 900,
            "expires_at": 2000,
        }
    )
    item["provider_receipts"].append(v2_provider_receipt)
    engine.managed_key_provider_receipts = (v2_provider_receipt,)
    missing_v1 = engine.health()
    assert missing_v1["managed_cycle_keys_ready"] is False
    assert (
        "managed_key_provider_version_unqualified:managed-test-key/v1"
        in missing_v1["managed_cycle_key_failures"]
    )
    both_provider_receipts = list(item["provider_receipts"])
    item["provider_receipts"][:] = [v2_provider_receipt]
    unqualified_replacement = item["make"](
        tmp_path / "rotation-v1-unqualified.sqlite3"
    )
    v1_claim = v1_checkpoint["receipt"]["payload"]
    with pytest.raises(ContextFault, match="managed_key_provider_unqualified"):
        unqualified_replacement.hydrate(
            item["authority"].sign(
                {
                    "operation": "hydrate",
                    "namespace": item["namespace"].model_dump(),
                    "manifest_checksum": v1_claim["manifest_checksum"],
                    "key_ref": v1_claim["key_ref"],
                    "key_provider": v1_claim["key_provider"],
                    "key_version": v1_claim["key_version"],
                    "fence": 2,
                    "expires_at": 1900,
                }
            ),
            v1_checkpoint["receipt"],
            base64.b64decode(v1_checkpoint["pack"], validate=True),
            {item["service"].key_id: item["service"].key.public_key()},
        )
    item["provider_receipts"][:] = both_provider_receipts
    engine.managed_key_provider_receipts = tuple(item["provider_receipts"])

    reservation = engine.reserve(
        item["authority"].sign(
            {
                "operation": "reserve",
                "owner_id": "owner-retention",
                "project_id": "project-retention",
                "idempotency_key": "request-retention-v2",
                "request_digest": digest("request-retention-v2"),
                "profile_digest": digest(item["profile"]),
                "expires_at": 1500,
            }
        )
    )
    namespace_v2 = NamespaceV1(
        owner_id="owner-retention",
        project_id="project-retention",
        objective_id="objective-retention-v2",
        context_cycle_id=reservation["payload"]["context_cycle_id"],
    )
    authority_v2 = {"objective": "retention rotation v2", "paths": ["src"]}
    snapshot_v2 = {
        "project_id": namespace_v2.project_id,
        "graph_id": "graph-retention-v2",
        "policy_digest": digest("policy-retention-v2"),
        "nodes": [],
    }
    binding_v2 = ContextBindingV1(
        namespace=namespace_v2,
        context_bucket_id=reservation["payload"]["context_bucket_id"],
        repository_id="repository-retention",
        repo_main_sha="a" * 40,
        project_graph_id="graph-retention-v2",
        graph_revision=2,
        graph_checksum=digest(snapshot_v2),
        plan_digest=digest("plan-retention-v2"),
        execution_profile_digest=digest("execution-retention-v2"),
        shared_ir_digest=digest("ir-retention-v2"),
        authorization_receipt_ref="authority-retention-v2",
        authorization_digest=digest(authority_v2),
        policy_digest=digest("policy-retention-v2"),
        redaction_digest=digest("redaction-retention-v2"),
        profile_digest=digest(item["profile"]),
        captain_binding_id="captain-retention-v2",
        created_at=1000,
        expires_at=100000,
    )
    binding_v2_digest = digest(binding_v2)
    engine.bind(
        item["authority"].sign(binding_v2.model_dump()), authority_v2, snapshot_v2
    )

    def cap_v2(role="worker", fence=1):
        return item["authority"].sign(
            CapabilityV1(
                namespace=namespace_v2,
                principal_id="principal-retention-v2",
                lane_id="lane-retention-v2",
                task_id="task-retention-v2",
                role=role,
                operations=[
                    "append",
                    "retrieve",
                    "checkpoint",
                    "seal",
                    "status",
                    "retention",
                ],
                source_class="tool_observation",
                binding_digest=binding_v2_digest,
                policy_digest=binding_v2.policy_digest,
                profile_digest=digest(item["profile"]),
                fence=fence,
                expires_at=90000,
                max_bytes=32768,
                max_tokens=12000,
            ).model_dump()
        )

    durability_status = engine.status(cap_v2("durability"))["payload"]
    engine.register_checkpoint_durability(
        item["authority"].sign(
            ContextCheckpointDurabilityCommandV1(
                namespace=namespace_v2,
                idempotency_key="register-rotation-v2",
                expected_revision=durability_status["revision"],
                mode="remote_registered",
                issued_at=item["clock"][0],
                expires_at=item["clock"][0] + 100,
            ).model_dump(mode="json")
        )
    )
    engine.append(
        cap_v2(),
        AppendRequestV1(
            idempotency_key="rotation-v2-record",
            text="managed rotation version two evidence",
            action_ref="rotation-v2",
        ),
    )
    v2_checkpoint = engine.checkpoint(cap_v2("durability"), "rotation-v2-checkpoint")
    assert v2_checkpoint["receipt"]["payload"]["key_version"] == (
        "managed-test-key/v2"
    )
    health = engine.health()
    assert health["managed_cycle_keys_ready"] is True
    assert set(health["managed_key_provider_receipt_digests"]) == {
        "managed-test-key/v1",
        "managed-test-key/v2",
    }

    def hydrate(checkpoint, namespace, path):
        claim = checkpoint["receipt"]["payload"]
        replacement = item["make"](path)
        return replacement.hydrate(
            item["authority"].sign(
                {
                    "operation": "hydrate",
                    "namespace": namespace.model_dump(),
                    "manifest_checksum": claim["manifest_checksum"],
                    "key_ref": claim["key_ref"],
                    "key_provider": claim["key_provider"],
                    "key_version": claim["key_version"],
                    "fence": 2,
                    "expires_at": 1900,
                }
            ),
            checkpoint["receipt"],
            base64.b64decode(checkpoint["pack"], validate=True),
            {item["service"].key_id: item["service"].key.public_key()},
        )

    assert hydrate(
        v1_checkpoint, item["namespace"], tmp_path / "rotation-v1-replacement.sqlite3"
    )["payload"]["key_version"] == "managed-test-key/v1"
    assert hydrate(
        v2_checkpoint, namespace_v2, tmp_path / "rotation-v2-replacement.sqlite3"
    )["payload"]["key_version"] == "managed-test-key/v2"

    v1_terminal, _ = item["seal"]()
    v2_terminal = engine.checkpoint(cap_v2("durability"), "rotation-v2-terminal")
    v2_closure = item["proof"].sign(
        {
            "schema_version": "ContextClosureProofV1",
            "namespace": namespace_v2.model_dump(),
            "binding_digest": binding_v2_digest,
            "final_root": v2_terminal["receipt"]["payload"]["segment_chain_root"],
            "final_cursor": v2_terminal["receipt"]["payload"]["cursor"],
            "execution_dag_digest": digest("dag-retention-v2"),
            "plan_ir_digest": binding_v2.shared_ir_digest,
            "shared_ir_digest": digest("final-ir-retention-v2"),
            "repo_main_sha": binding_v2.repo_main_sha,
            "git_head_sha": "b" * 40,
            "pr_head_sha": "b" * 40,
            "pr_url": "https://github.com/org/repo/pull/2",
            "ci_receipt": digest("ci-retention-v2"),
            "nano_receipt": digest("nano-retention-v2"),
            "proof_receipt": digest("proof-retention-v2"),
            "memory_candidate_digest": digest("memory-retention-v2"),
            "promotion_set_root": digest([]),
            "accepted_record_ids": [],
            "rejected_record_ids": [],
            "expires_at": 2000,
        }
    )
    engine.seal(cap_v2("verifier"), v2_closure, v2_terminal["receipt"])

    item["clock"][0] = 1061

    def expire(checkpoint, namespace, capability, operation_key):
        object_ref = digest(f"spaces://context/{operation_key}")
        remote = item["deletion"].sign(
            {
                "schema_version": "ContextRemoteDeletionReceiptV1",
                "namespace": namespace.model_dump(),
                "manifest_checksum": checkpoint["receipt"]["payload"][
                    "manifest_checksum"
                ],
                "object_ref_digest": object_ref,
                "deleted": True,
                "issued_at": 1061,
                "expires_at": 1161,
            }
        )
        status = engine.status(capability)["payload"]
        return engine.retention(
            capability,
            item["authority"].sign(
                {
                    "schema_version": "ContextRetentionCommandV1",
                    "operation": "expire",
                    "namespace": namespace.model_dump(),
                    "idempotency_key": operation_key,
                    "expected_revision": status["revision"],
                    "reason_code": "policy_expiry",
                    "remote_deletion_receipt": remote,
                    "checkpoint_object_ref_digest": object_ref,
                    "issued_at": 1061,
                    "expires_at": 1161,
                }
            ),
        )

    v1_deleted = expire(
        v1_terminal,
        item["namespace"],
        item["cap"]("durability"),
        "expire-retention-v1",
    )
    v2_deleted = expire(
        v2_terminal,
        namespace_v2,
        cap_v2("durability"),
        "expire-retention-v2",
    )
    assert v1_deleted["payload"]["cryptographic_erasure"] is True
    assert v2_deleted["payload"]["cryptographic_erasure"] is True
    assert v1_deleted["payload"]["key_destruction_receipt"]["payload"][
        "key_version"
    ] == "managed-test-key/v1"
    assert v2_deleted["payload"]["key_destruction_receipt"]["payload"][
        "key_version"
    ] == "managed-test-key/v2"


def test_hydrate_receipts_distinguish_checkpoints_with_the_same_root(tmp_path):
    item = _keyed_cycle(tmp_path)
    first = item["engine"].checkpoint(item["cap"]("durability"), "same-root-one")
    second = item["engine"].checkpoint(item["cap"]("durability"), "same-root-two")

    def hydrate(checkpoint, path):
        claim = checkpoint["receipt"]["payload"]
        replacement = item["make"](path)
        return replacement.hydrate(
            item["authority"].sign(
                {
                    "operation": "hydrate",
                    "namespace": item["namespace"].model_dump(),
                    "manifest_checksum": claim["manifest_checksum"],
                    "key_ref": claim["key_ref"],
                    "key_provider": claim["key_provider"],
                    "key_version": claim["key_version"],
                    "fence": 2,
                    "expires_at": 1900,
                }
            ),
            checkpoint["receipt"],
            base64.b64decode(checkpoint["pack"], validate=True),
            {item["service"].key_id: item["service"].key.public_key()},
        )

    first_receipt = hydrate(first, tmp_path / "same-root-first.sqlite3")
    second_receipt = hydrate(second, tmp_path / "same-root-second.sqlite3")
    assert first["receipt"]["payload"]["segment_chain_root"] == second["receipt"][
        "payload"
    ]["segment_chain_root"]
    assert first_receipt != second_receipt
    assert first_receipt["payload"]["checkpoint_number"] == 1
    assert second_receipt["payload"]["checkpoint_number"] == 2


def test_legacy_expire_route_cannot_bypass_signed_remote_deletion(tmp_path):
    item = _keyed_cycle(tmp_path, keyed=False, legacy_binding=True)
    item["engine"].append(
        item["cap"](),
        AppendRequestV1(
            idempotency_key="legacy-retained",
            text="legacy remote checkpoint remains decryptable",
            action_ref="legacy-expire-test",
        ),
    )
    checkpoint, _ = item["seal"]()
    item["clock"][0] = 1061
    with pytest.raises(ContextFault, match="signed_retention_required"):
        ContextService(item["engine"]).dispatch(
            {"operation": "expire", "capability": item["cap"]("durability")}
        )
    with item["engine"].store.connection() as db:
        assert db.execute("SELECT count(*) FROM segments").fetchone()[0] == 1
    deleted = item["engine"].retention(
        item["cap"]("durability"), item["expire_command"](checkpoint)
    )
    assert deleted["payload"]["logical_deletion"] is True
    assert deleted["payload"]["cryptographic_erasure"] is False


def test_freeze_receipts_publish_authenticated_promotion_metadata_without_text(tmp_path):
    item = _keyed_cycle(tmp_path)
    evidence = digest("independent-ci-evidence")
    appended = item["engine"].append(
        item["cap"]("verifier", source="ci_verified"),
        AppendRequestV1(
            idempotency_key="promotion-candidate",
            text="private candidate text must not enter freeze receipt",
            plane="P4",
            action_ref="promotion-test",
            evidence_refs=[evidence],
        ),
    )
    frozen = item["engine"].freeze(item["cap"]("verifier"))
    payload = verify(
        frozen,
        {item["engine"].signer.key_id: item["engine"].signer.key.public_key()},
    )
    expected = [
        {
            "record_id": appended["payload"]["record_id"],
            "content_digest": appended["payload"]["content_digest"],
            "evidence_refs": [evidence],
        }
    ]
    assert payload["promotion_candidates"] == expected
    assert payload["promotion_candidates_root"] == digest(expected)
    assert b"private candidate text" not in str(frozen).encode()
    checkpoint = item["engine"].checkpoint(
        item["cap"]("durability"), "promotion-checkpoint"
    )
    binding = item["binding"]
    closure = item["proof"].sign(
        {
            "schema_version": "ContextClosureProofV1",
            "namespace": item["namespace"].model_dump(),
            "binding_digest": digest(binding),
            "final_root": checkpoint["receipt"]["payload"]["segment_chain_root"],
            "final_cursor": 1,
            "execution_dag_digest": digest("promotion-dag"),
            "plan_ir_digest": binding.shared_ir_digest,
            "shared_ir_digest": digest("promotion-final-ir"),
            "repo_main_sha": binding.repo_main_sha,
            "git_head_sha": "b" * 40,
            "pr_head_sha": "b" * 40,
            "pr_url": "https://github.com/org/repo/pull/1",
            "ci_receipt": digest("promotion-ci"),
            "nano_receipt": digest("promotion-nano"),
            "proof_receipt": digest("promotion-proof"),
            "memory_candidate_digest": digest("promotion-memory"),
            "promotion_set_root": digest(expected),
            "accepted_record_ids": [appended["payload"]["record_id"]],
            "rejected_record_ids": [],
            "expires_at": 2000,
        }
    )
    sealed = item["engine"].seal(
        item["cap"]("verifier"), closure, checkpoint["receipt"]
    )
    assert sealed["payload"]["accepted_count"] == 1


def _scale_evidence():
    benchmark = ReceiptSigner("benchmark-independent", Ed25519PrivateKey.generate())
    stability_signer = ReceiptSigner("executor-independent", Ed25519PrivateKey.generate())
    isolation_signer = ReceiptSigner("isolation-independent", Ed25519PrivateKey.generate())
    provisional = ContextProfileV1(
        name="hosted-v1",
        index_version=sqlite3.sqlite_version,
        benchmark_receipt="0" * 64,
        max_records=100,
        max_indexed_bytes=10_000,
        disk_free_floor=0,
    )
    revision = "d" * 40
    dataset_digest = digest("approved-project-recall-dataset")
    generator_digest = digest("randomized-data-plane-cases/v1")
    isolation_result_digest = digest("isolation-scale-result")
    isolation = isolation_signer.sign(
        {
            "schema_version": "ContextDataPlaneIsolationReceiptV1",
            "source_revision": revision,
            "profile_benchmark_digest": profile_benchmark_digest(provisional),
            "boundary": "hosted-data-plane/v1",
            "case_generator_digest": generator_digest,
            "cases": 1_000_000,
            "unauthorized_records": 0,
            "result_digest": isolation_result_digest,
            "started_at": 700,
            "finished_at": 850,
            "issued_at": 900,
            "expires_at": 1200,
        }
    )
    stability = stability_signer.sign(
        {
            "schema_version": "ContextExecutorStabilityReceiptV1",
            "source_revision": revision,
            "profile_benchmark_digest": profile_benchmark_digest(provisional),
            "target_concurrency": provisional.max_concurrent_cycles,
            "ci_receipt_digest": digest("executor-ci-observation"),
            "executor_stable": True,
            "started_at": 800,
            "finished_at": 900,
            "issued_at": 900,
            "expires_at": 1200,
        }
    )
    payload = HostedScaleReceiptV1(
        profile_benchmark_digest=profile_benchmark_digest(provisional),
        source_revision=revision,
        source_tree_clean=True,
        index_implementation=provisional.index_implementation,
        index_version=provisional.index_version,
        embedding_version=provisional.embedding_version,
        storage_version=provisional.storage_version,
        configured_max_records=100,
        configured_max_indexed_bytes=10_000,
        indexed_records=90,
        indexed_bytes=9_000,
        reachable_tokens=2_250,
        resident_peak_bytes=100_000_000,
        retrieval_samples=20,
        target_concurrency=provisional.max_concurrent_cycles,
        retrieval_p50_ms=10,
        retrieval_p95_ms=20,
        retrieval_p99_ms=30,
        checkpoint_ms=100,
        checkpoint_receipt_digest=digest("checkpoint-scale-receipt"),
        hydrate_ms=100,
        hydrate_receipt_digest=digest("hydrate-scale-receipt"),
        rebuild_ms=100,
        rebuild_receipt_digest=digest("rebuild-scale-receipt"),
        recall_dataset_name="approved-project-recall/v1",
        recall_dataset_digest=dataset_digest,
        recall_basis_points=9000,
        isolation_cases=1_000_000,
        unauthorized_records=0,
        isolation_result_digest=isolation_result_digest,
        isolation_evidence_class="data_plane",
        data_plane_isolation_receipt=isolation,
        executor_stable=True,
        executor_stability_receipt=stability,
        started_at=900,
        finished_at=950,
        issued_at=950,
        expires_at=1200,
    )
    envelope = benchmark.sign(payload.model_dump())
    profile = provisional.model_copy(update={"benchmark_receipt": digest(envelope)})
    return (
        benchmark,
        stability_signer,
        isolation_signer,
        revision,
        dataset_digest,
        envelope,
        profile,
    )


def test_scale_receipt_enables_reach_only_for_exact_signed_build():
    (
        benchmark,
        stability,
        isolation,
        revision,
        dataset_digest,
        envelope,
        profile,
    ) = _scale_evidence()
    keys = {benchmark.key_id: benchmark.key.public_key()}
    stability_keys = {stability.key_id: stability.key.public_key()}
    isolation_keys = {isolation.key_id: isolation.key.public_key()}
    generator_digest = digest("randomized-data-plane-cases/v1")
    qualified = qualify_scale_receipt(
        profile,
        envelope,
        keys,
        now=1000,
        expected_source_revision=revision,
        expected_recall_dataset_digest=dataset_digest,
        expected_data_plane_case_generator_digest=generator_digest,
        executor_stability_keys=stability_keys,
        data_plane_isolation_keys=isolation_keys,
    )
    assert qualified.valid is True
    assert qualified.reachable_tokens == 2250
    under_target_stability = stability.sign(
        {
            **envelope["payload"]["executor_stability_receipt"]["payload"],
            "target_concurrency": 1,
        }
    )
    under_target = benchmark.sign(
        {
            **envelope["payload"],
            "target_concurrency": 1,
            "executor_stability_receipt": under_target_stability,
        }
    )
    under_target_profile = profile.model_copy(
        update={"benchmark_receipt": digest(under_target)}
    )
    with pytest.raises(ContextFault, match="benchmark_profile_mismatch"):
        qualify_scale_receipt(
            under_target_profile,
            under_target,
            keys,
            now=1000,
            expected_source_revision=revision,
            expected_recall_dataset_digest=dataset_digest,
            expected_data_plane_case_generator_digest=generator_digest,
            executor_stability_keys=stability_keys,
            data_plane_isolation_keys=isolation_keys,
        )
    wrong_build = qualify_scale_receipt(
        profile,
        envelope,
        keys,
        now=1000,
        expected_source_revision="e" * 40,
        expected_recall_dataset_digest=dataset_digest,
        expected_data_plane_case_generator_digest=generator_digest,
        executor_stability_keys=stability_keys,
        data_plane_isolation_keys=isolation_keys,
    )
    assert wrong_build.valid is False
    assert "source_revision_mismatch" in wrong_build.failures
    overstated_payload = {**envelope["payload"], "reachable_tokens": 2251}
    overstated = benchmark.sign(overstated_payload)
    overstated_profile = profile.model_copy(update={"benchmark_receipt": digest(overstated)})
    overstated_result = qualify_scale_receipt(
        overstated_profile,
        overstated,
        keys,
        now=1000,
        expected_source_revision=revision,
        expected_recall_dataset_digest=dataset_digest,
        expected_data_plane_case_generator_digest=generator_digest,
        executor_stability_keys=stability_keys,
        data_plane_isolation_keys=isolation_keys,
    )
    assert "reach_not_conservative" in overstated_result.failures
    predicate_payload = {
        **envelope["payload"],
        "isolation_evidence_class": "predicate",
        "data_plane_isolation_receipt": None,
    }
    predicate = benchmark.sign(predicate_payload)
    predicate_profile = profile.model_copy(update={"benchmark_receipt": digest(predicate)})
    predicate_result = qualify_scale_receipt(
        predicate_profile,
        predicate,
        keys,
        now=1000,
        expected_source_revision=revision,
        expected_recall_dataset_digest=dataset_digest,
        expected_data_plane_case_generator_digest=generator_digest,
        executor_stability_keys=stability_keys,
        data_plane_isolation_keys=isolation_keys,
    )
    assert predicate_result.valid is False
    assert "namespace_data_plane_unmeasured" in predicate_result.failures
    changed_generator_receipt = isolation.sign(
        {
            **envelope["payload"]["data_plane_isolation_receipt"]["payload"],
            "case_generator_digest": digest("weak-or-different-generator"),
        }
    )
    changed_generator = benchmark.sign(
        {
            **envelope["payload"],
            "data_plane_isolation_receipt": changed_generator_receipt,
        }
    )
    changed_generator_profile = profile.model_copy(
        update={"benchmark_receipt": digest(changed_generator)}
    )
    changed_generator_result = qualify_scale_receipt(
        changed_generator_profile,
        changed_generator,
        keys,
        now=1000,
        expected_source_revision=revision,
        expected_recall_dataset_digest=dataset_digest,
        expected_data_plane_case_generator_digest=generator_digest,
        executor_stability_keys=stability_keys,
        data_plane_isolation_keys=isolation_keys,
    )
    assert changed_generator_result.valid is False
    assert changed_generator_result.reachable_tokens is None
    assert (
        "data_plane_case_generator_mismatch"
        in changed_generator_result.failures
    )


def test_benchmark_signer_must_be_independent(tmp_path):
    authority = ReceiptSigner("gateway", Ed25519PrivateKey.generate())
    proof = ReceiptSigner("proof", Ed25519PrivateKey.generate())
    service = ReceiptSigner("contextd", Ed25519PrivateKey.generate())
    _, _, _, _, _, envelope, profile = _scale_evidence()
    with pytest.raises(ContextFault, match="verification_key_overlap"):
        ContextEngine(
            tmp_path / "overlap.sqlite3",
            profile,
            EnvelopeCipher(os.urandom(32)),
            service,
            {authority.key_id: authority.key.public_key()},
            {proof.key_id: proof.key.public_key()},
            benchmark_receipt=envelope,
            benchmark_keys={"benchmark-alias": authority.key.public_key()},
        )


def test_hosted_reach_claim_stays_off_when_managed_retention_is_not_ready(tmp_path):
    (
        benchmark,
        stability,
        isolation,
        revision,
        dataset_digest,
        envelope,
        profile,
    ) = _scale_evidence()
    authority = ReceiptSigner("health-gateway", Ed25519PrivateKey.generate())
    proof = ReceiptSigner("health-proof", Ed25519PrivateKey.generate())
    service = ReceiptSigner("health-contextd", Ed25519PrivateKey.generate())
    deletion = ReceiptSigner("health-deletion", Ed25519PrivateKey.generate())
    destruction = ReceiptSigner("health-kms-deletion", Ed25519PrivateKey.generate())
    provider_authority = ReceiptSigner("health-kms-authority", Ed25519PrivateKey.generate())
    build = ReceiptSigner("health-build", Ed25519PrivateKey.generate())
    clock = [1000]
    runtime_environment = runtime_environment_identity()
    build_manifest = build.sign(
        {
            "schema_version": "ContextRuntimeBuildManifestV3",
            "distribution": "aether-context",
            "version": __version__,
            "source_revision": revision,
            "package_tree_digest": runtime_package_tree_digest(
                require_source_only=True
            ),
            "wheel_digest": digest("exact-built-wheel"),
            "recall_dataset_digest": dataset_digest,
            "data_plane_case_generator_digest": digest(
                "randomized-data-plane-cases/v1"
            ),
            "built_at": 800,
            "runtime_environment": runtime_environment.model_dump(mode="json"),
            "locked_environment_digest": digest(runtime_environment),
        }
    )
    common = {
        "benchmark_receipt": envelope,
        "benchmark_keys": {benchmark.key_id: benchmark.key.public_key()},
        "remote_deletion_keys": {deletion.key_id: deletion.key.public_key()},
        "executor_stability_keys": {stability.key_id: stability.key.public_key()},
        "data_plane_isolation_keys": {isolation.key_id: isolation.key.public_key()},
        "key_destruction_keys": {destruction.key_id: destruction.key.public_key()},
        "runtime_build_manifest": build_manifest,
        "runtime_build_keys": {build.key_id: build.key.public_key()},
    }

    def make(path, provider, provider_receipt):
        return ContextEngine(
            path,
            profile,
            EnvelopeCipher(os.urandom(32)),
            service,
            {authority.key_id: authority.key.public_key()},
            {proof.key_id: proof.key.public_key()},
            clock=lambda: clock[0],
            cycle_keys=provider,
            managed_key_provider_receipt=provider_receipt,
            key_provider_keys={
                provider_authority.key_id: provider_authority.key.public_key()
            },
            **common,
        )

    managed = ManagedTestCycleKeys(
        tmp_path / "managed-keys",
        EnvelopeCipher(os.urandom(32)),
        destruction,
        clock,
    )
    provider_receipt = provider_authority.sign(
        {
            "schema_version": "ContextManagedKeyProviderReceiptV1",
            "provider": managed.provider,
            "key_version": managed.key_version,
            "shared_hydrate_supported": True,
            "verified_destruction_supported": True,
            "issued_at": 900,
            "expires_at": 1300,
        }
    )
    healthy_engine = make(tmp_path / "healthy.sqlite3", managed, provider_receipt)
    healthy = healthy_engine.health()
    assert ContextHealthV1.model_validate(healthy).model_dump(mode="json") == healthy
    assert "runtime_package_tree_digest" not in healthy
    assert healthy["ready"] is True
    assert healthy["reach_claim_enabled"] is True
    healthy_v2 = healthy_engine.health_v2()
    assert ContextHealthV2.model_validate(healthy_v2).model_dump(mode="json") == healthy_v2
    assert (
        healthy_v2["runtime_package_tree_digest"]
        == build_manifest["payload"]["package_tree_digest"]
        == runtime_package_tree_digest(require_source_only=True)
    )
    assert healthy_v2["runtime_locked_environment_digest"] == digest(
        runtime_environment
    )
    assert healthy_v2["runtime_python_abi"] == runtime_environment.python_abi
    assert (
        healthy_v2["runtime_dependency_versions"]
        == runtime_environment.dependency_versions
    )
    assert (
        healthy_v2["runtime_dependency_artifact_digests"]
        == runtime_environment.dependency_artifact_digests
    )
    legacy_build_payload = build_manifest["payload"].copy()
    legacy_build_payload["schema_version"] = "ContextRuntimeBuildManifestV1"
    legacy_build_payload.pop("runtime_environment")
    legacy_build_payload.pop("locked_environment_digest")
    legacy_build_payload.pop("data_plane_case_generator_digest")
    legacy_build_common = {
        **common,
        "runtime_build_manifest": build.sign(legacy_build_payload),
    }
    legacy_build_health = ContextEngine(
        tmp_path / "legacy-build.sqlite3",
        profile,
        EnvelopeCipher(os.urandom(32)),
        service,
        {authority.key_id: authority.key.public_key()},
        {proof.key_id: proof.key.public_key()},
        clock=lambda: clock[0],
        cycle_keys=managed,
        managed_key_provider_receipt=provider_receipt,
        key_provider_keys={
            provider_authority.key_id: provider_authority.key.public_key()
        },
        **legacy_build_common,
    ).health()
    assert legacy_build_health["ready"] is False
    assert legacy_build_health["reach_claim_enabled"] is False
    assert (
        "data_plane_case_generator_unbound"
        in legacy_build_health["benchmark_failures"]
    )
    assert "runtime_environment_unbound" in legacy_build_health[
        "runtime_build_failures"
    ]
    clock[0] = 1200
    expired = healthy_engine.health()
    assert expired["ready"] is False
    assert expired["reach_claim_enabled"] is False
    assert "benchmark_receipt_expired" in expired["benchmark_failures"]
    clock[0] = 1000
    canary = FileCycleKeyProvider(
        tmp_path / "canary-keys", EnvelopeCipher(os.urandom(32))
    )
    blocked = make(tmp_path / "blocked.sqlite3", canary, provider_receipt).health()
    assert blocked["ready"] is False
    assert blocked["reach_claim_enabled"] is False
    wrong_build = build.sign(
        {
            **build_manifest["payload"],
            "package_tree_digest": digest("different-installed-package-bytes"),
        }
    )
    invalid_build_common = {**common, "runtime_build_manifest": wrong_build}
    build_blocked = ContextEngine(
        tmp_path / "wrong-build.sqlite3",
        profile,
        EnvelopeCipher(os.urandom(32)),
        service,
        {authority.key_id: authority.key.public_key()},
        {proof.key_id: proof.key.public_key()},
        clock=lambda: clock[0],
        cycle_keys=managed,
        managed_key_provider_receipt=provider_receipt,
        key_provider_keys={
            provider_authority.key_id: provider_authority.key.public_key()
        },
        **invalid_build_common,
    ).health()
    assert build_blocked["ready"] is False
    assert build_blocked["reach_claim_enabled"] is False
    assert "runtime_package_tree_mismatch" in build_blocked["runtime_build_failures"]

    changed_environment = runtime_environment.model_copy(
        update={
            "dependency_versions": {
                **runtime_environment.dependency_versions,
                "pydantic": "0.0.0-mismatch",
            }
        }
    )
    wrong_environment_manifest = build.sign(
        {
            **build_manifest["payload"],
            "runtime_environment": changed_environment.model_dump(mode="json"),
            "locked_environment_digest": digest(changed_environment),
        }
    )
    environment_blocked = ContextEngine(
        tmp_path / "wrong-environment.sqlite3",
        profile,
        EnvelopeCipher(os.urandom(32)),
        service,
        {authority.key_id: authority.key.public_key()},
        {proof.key_id: proof.key.public_key()},
        clock=lambda: clock[0],
        cycle_keys=managed,
        managed_key_provider_receipt=provider_receipt,
        key_provider_keys={
            provider_authority.key_id: provider_authority.key.public_key()
        },
        **{**common, "runtime_build_manifest": wrong_environment_manifest},
    ).health_v2()
    assert environment_blocked["ready"] is False
    assert "runtime_environment_mismatch" in environment_blocked[
        "runtime_build_failures"
    ]


def test_million_case_namespace_isolation_is_deterministic():
    result = run_namespace_isolation()
    assert result["cases"] == 1_000_000
    assert result["unauthorized_records"] == 0
    assert result["result_digest"] == (
        "dae81f9acfb516a098fa758d7d31be4390ff4cb256b0fda8efb9316906851036"
    )


def test_source_only_digest_rejects_timestamp_pyc_and_native_shadow_artifacts(
    tmp_path, monkeypatch
):
    package = tmp_path / "aether_context"
    package.mkdir()
    source = package / "engine.py"
    benign = "VALUE = 'SAFE'\n"
    malicious = "VALUE = 'PWN!'\n"
    assert len(benign) == len(malicious)
    source.write_text(benign)
    baseline = runtime_package_tree_digest(package)
    source.write_text(malicious)
    timestamp = 1_700_000_000
    os.utime(source, (timestamp, timestamp))
    py_compile.compile(
        str(source),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    source.write_text(benign)
    os.utime(source, (timestamp, timestamp))
    assert runtime_package_tree_digest(package) == baseline
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    with pytest.raises(ContextFault, match="runtime_import_artifact"):
        runtime_package_tree_digest(package, require_source_only=True)

    shutil.rmtree(package / "__pycache__")
    for suffix in (".pyd", ".so"):
        shadow = package / ("engine" + suffix)
        shadow.write_bytes(b"native-shadow")
        with pytest.raises(ContextFault, match="runtime_import_artifact"):
            runtime_package_tree_digest(package, require_source_only=True)
        shadow.unlink()


def test_health_v2_returns_closed_unready_projection_when_rehash_fails(
    tmp_path, monkeypatch
):
    item = _keyed_cycle(tmp_path)

    def unavailable(*_args, **_kwargs):
        raise ContextFault("context_runtime_build_invalid")

    monkeypatch.setattr("aether_context.engine.runtime_package_tree_digest", unavailable)
    health = item["engine"].health_v2()
    assert ContextHealthV2.model_validate(health).model_dump(mode="json") == health
    assert health["ready"] is False
    assert health["runtime_package_tree_digest"] is None
    assert "runtime_package_tree_unavailable" in health["runtime_build_failures"]


def test_sqlite_wal_discards_uncommitted_process_crash(tmp_path):
    path = tmp_path / "chaos.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE chaos_probe(value INTEGER)")
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_uncommitted_write, args=(str(path),)
    )
    process.start()
    process.join(20)
    assert process.exitcode == 17
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert db.execute("SELECT count(*) FROM chaos_probe").fetchone()[0] == 0


def test_disk_watermark_stops_new_writes_without_deleting_cycle(tmp_path, monkeypatch):
    item = _keyed_cycle(tmp_path)
    item["engine"].profile = item["profile"].model_copy(update={"disk_free_floor": 2})
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr("aether_context.engine.shutil.disk_usage", lambda _: usage(10, 10, 0))
    with pytest.raises(ContextFault, match="disk_watermark"):
        item["engine"].append(
            item["cap"](),
            AppendRequestV1(
                idempotency_key="watermark",
                text="must not persist",
                action_ref="watermark-test",
            ),
        )
    with item["engine"].store.connection() as db:
        assert db.execute("SELECT count(*) FROM segments").fetchone()[0] == 0


def test_deletion_receipt_is_service_signed(tmp_path):
    item = _keyed_cycle(tmp_path)
    checkpoint, _ = item["seal"]()
    item["clock"][0] = 1061
    receipt = item["engine"].retention(
        item["cap"]("durability"), item["expire_command"](checkpoint)
    )
    verified = verify(
        receipt,
        {item["engine"].signer.key_id: item["engine"].signer.key.public_key()},
    )
    assert verified["logical_deletion"] is True
