import base64
import hashlib
import multiprocessing
import os
import sqlite3
from collections import namedtuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from aether_context.contracts import (
    AppendRequestV1,
    CapabilityV1,
    ContextBindingV1,
    ContextProfileV1,
    HostedScaleReceiptV1,
    NamespaceV1,
    RetrieveRequestV1,
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
    runtime_package_tree_digest,
)


class ManagedTestCycleKeys(FileCycleKeyProvider):
    provider = "managed-test-kms/v1"
    key_version = "managed-test-key/v1"

    def __init__(self, root, wrapper, destruction_signer, clock):
        super().__init__(root, wrapper)
        self.destruction_signer = destruction_signer
        self.clock = clock
        self.receipts = {}

    def destroy(self, key_ref, namespace, destroyed_at):
        local = super().destroy(key_ref, namespace, destroyed_at)
        if key_ref not in self.receipts:
            self.receipts[key_ref] = self.destruction_signer.sign(
                {
                    "schema_version": "ContextKeyDestructionReceiptV1",
                    "provider": self.provider,
                    "key_ref": key_ref,
                    "namespace_digest": digest(namespace),
                    "key_version": self.key_version,
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


def _keyed_cycle(tmp_path, *, managed=True, keyed=True, legacy_binding=False):
    authority = ReceiptSigner("gateway-retention", Ed25519PrivateKey.generate())
    proof = ReceiptSigner("proof-retention", Ed25519PrivateKey.generate())
    deletion = ReceiptSigner("object-store-deletion", Ed25519PrivateKey.generate())
    provider_authority = ReceiptSigner("key-provider-authority", Ed25519PrivateKey.generate())
    destruction = ReceiptSigner("kms-destruction", Ed25519PrivateKey.generate())
    service = ReceiptSigner("contextd-retention", Ed25519PrivateKey.generate())
    profile = ContextProfileV1(
        index_version=sqlite3.sqlite_version,
        disk_free_floor=0,
        max_records=100,
        max_indexed_bytes=1_000_000,
        retention_seconds=60,
    )
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
            managed_key_provider_receipt=provider_receipt,
            key_provider_keys=(
                {provider_authority.key_id: provider_authority.key.public_key()}
                if provider_receipt is not None
                else None
            ),
            key_destruction_keys={destruction.key_id: destruction.key.public_key()},
        )

    engine = make(tmp_path / "context.sqlite3")
    reservation = engine.reserve(
        authority.sign(
            {
                "operation": "reserve",
                "owner_id": "owner-retention",
                "project_id": "project-retention",
                "idempotency_key": "request-retention",
                "request_digest": digest("request-retention"),
                "profile_digest": digest(profile),
                "expires_at": 1500,
            }
        )
    )
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
    engine.bind(authority.sign(binding_payload), authority_payload, snapshot)

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
                max_bytes=32768,
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
        "namespace": namespace,
        "binding": binding,
        "binding_payload": binding_payload,
        "binding_digest": binding_digest,
        "clock": clock,
        "profile": profile,
    }


def test_retention_default_and_hard_ceiling():
    assert ContextProfileV1(index_version=sqlite3.sqlite_version).retention_seconds == 86400
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
        assert row["key_ref"] is None
        assert row["snapshot"] is None


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
                "expires_at": 2000,
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
        assert db.execute("SELECT cleanup_state FROM cycles").fetchone()[0] == "PENDING"
    item["clock"][0] = 1200  # Both original signed proofs have expired on this retry.
    item["engine"].retention_manager = RetentionManager(item["engine"])
    receipt = item["engine"].retention(item["cap"]("durability"), command)
    assert receipt["payload"]["cryptographic_erasure"] is True


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
    replacement = item["make"](tmp_path / "replacement.sqlite3")
    grant = item["authority"].sign(
        {
            "operation": "hydrate",
            "namespace": namespace.model_dump(),
            "manifest_checksum": claim["manifest_checksum"],
            "key_ref": claim["key_ref"],
            "key_provider": claim["key_provider"],
            "key_version": claim["key_version"],
            "fence": 2,
            "expires_at": 2000,
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
                    "expires_at": 2000,
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
    isolation_result_digest = digest("isolation-scale-result")
    isolation = isolation_signer.sign(
        {
            "schema_version": "ContextDataPlaneIsolationReceiptV1",
            "source_revision": revision,
            "profile_benchmark_digest": profile_benchmark_digest(provisional),
            "boundary": "hosted-data-plane/v1",
            "case_generator_digest": digest("randomized-data-plane-cases/v1"),
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
            "target_concurrency": 8,
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
        target_concurrency=8,
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
    qualified = qualify_scale_receipt(
        profile,
        envelope,
        keys,
        now=1000,
        expected_source_revision=revision,
        expected_recall_dataset_digest=dataset_digest,
        executor_stability_keys=stability_keys,
        data_plane_isolation_keys=isolation_keys,
    )
    assert qualified.valid is True
    assert qualified.reachable_tokens == 2250
    wrong_build = qualify_scale_receipt(
        profile,
        envelope,
        keys,
        now=1000,
        expected_source_revision="e" * 40,
        expected_recall_dataset_digest=dataset_digest,
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
        executor_stability_keys=stability_keys,
        data_plane_isolation_keys=isolation_keys,
    )
    assert predicate_result.valid is False
    assert "namespace_data_plane_unmeasured" in predicate_result.failures


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
    build_manifest = build.sign(
        {
            "schema_version": "ContextRuntimeBuildManifestV1",
            "distribution": "aether-context",
            "version": __version__,
            "source_revision": revision,
            "package_tree_digest": runtime_package_tree_digest(),
            "wheel_digest": digest("exact-built-wheel"),
            "recall_dataset_digest": dataset_digest,
            "built_at": 800,
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
    assert healthy["ready"] is True
    assert healthy["reach_claim_enabled"] is True
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


def test_million_case_namespace_isolation_is_deterministic():
    result = run_namespace_isolation()
    assert result["cases"] == 1_000_000
    assert result["unauthorized_records"] == 0
    assert result["result_digest"] == (
        "dae81f9acfb516a098fa758d7d31be4390ff4cb256b0fda8efb9316906851036"
    )


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
