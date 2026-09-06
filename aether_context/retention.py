"""Receipt-backed context retention and per-cycle key destruction.

The daemon never guesses when a legal or audit hold applies. Gateway supplies
short-lived signed commands, while the manager makes every command idempotent
and keeps deletion receipts after the encrypted cycle key is gone.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from .contracts import (
    ContextDeletionReceiptV1,
    ContextDeletionReceiptV2,
    ContextRetentionReceiptV1,
    KeyDestructionReceiptV1,
    RemoteDeletionReceiptV1,
    RetentionCommandV1,
    canonical,
    digest,
)
from .crypto import ContextFault, EnvelopeCipher, verify


def _sync_directory(path: Path) -> None:
    """Best effort directory fsync; supported on the production Linux path."""

    if os.name == "nt":  # Windows has no portable directory fsync.
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class FileCycleKeyProvider:
    """Canary-only file implementation of the managed cycle-key seam.

    Every cycle gets a random DEK. Only its wrapped form is stored in this
    service-owned directory. Deletion removes that sole wrapped-key object and
    leaves a content-free tombstone that can be verified on retry.
    """

    provider = "file-wrapped-dek/v1"
    key_version = "file-local/v1"

    def __init__(self, root: str | Path, wrapper: EnvelopeCipher):
        self.root = Path(root).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name != "nt":
            os.chmod(self.root, 0o700)
        self.wrapper = wrapper

    @staticmethod
    def _identity(namespace: dict) -> tuple[str, str]:
        namespace_digest = digest(namespace)
        return "ck_" + namespace_digest, namespace_digest

    def _paths(self, key_ref: str) -> tuple[Path, Path]:
        if not key_ref.startswith("ck_") or len(key_ref) != 67:
            raise ContextFault("context_key_reference_invalid")
        return self.root / (key_ref + ".key"), self.root / (key_ref + ".destroyed.json")

    @staticmethod
    def _aad(key_ref: str, namespace_digest: str) -> dict:
        return {
            "domain": "context-cycle-key/v1",
            "key_ref": key_ref,
            "namespace_digest": namespace_digest,
        }

    def ensure(self, namespace: dict) -> str:
        key_ref, namespace_digest = self._identity(namespace)
        key_path, tombstone_path = self._paths(key_ref)
        if tombstone_path.exists():
            raise ContextFault("context_key_destroyed")
        if not key_path.exists():
            wrapped = self.wrapper.encrypt(
                {"dek": base64.b64encode(os.urandom(32)).decode("ascii")},
                self._aad(key_ref, namespace_digest),
            )
            try:
                fd = os.open(
                    key_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                    0o600,
                )
            except FileExistsError:
                pass
            else:
                try:
                    os.write(fd, wrapped)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                _sync_directory(self.root)
        # A concurrent creator is accepted only after its bytes authenticate.
        self.cipher(key_ref, namespace)
        return key_ref

    def cipher(self, key_ref: str, namespace: dict) -> EnvelopeCipher:
        expected, namespace_digest = self._identity(namespace)
        if key_ref != expected:
            raise ContextFault("context_key_reference_invalid")
        key_path, tombstone_path = self._paths(key_ref)
        if tombstone_path.exists() and not key_path.exists():
            raise ContextFault("context_key_destroyed")
        try:
            wrapped = key_path.read_bytes()
        except OSError as exc:
            raise ContextFault("context_key_unavailable") from exc
        value = self.wrapper.decrypt(
            wrapped, self._aad(key_ref, namespace_digest), max_bytes=256
        )
        try:
            dek = base64.b64decode(value["dek"], validate=True)
        except (KeyError, ValueError) as exc:
            raise ContextFault("context_key_invalid") from exc
        if len(dek) != 32:
            raise ContextFault("context_key_invalid")
        return EnvelopeCipher(dek, domain=b"context-cycle/v1")

    def version_for(self, key_ref: str) -> str:
        self._paths(key_ref)
        return self.key_version

    def destroy(self, key_ref: str, namespace: dict, destroyed_at: int) -> dict:
        expected, namespace_digest = self._identity(namespace)
        if key_ref != expected:
            raise ContextFault("context_key_reference_invalid")
        key_path, tombstone_path = self._paths(key_ref)
        key_version = self.version_for(key_ref)
        tombstone: dict[str, Any] | None = None
        if tombstone_path.exists():
            try:
                tombstone = json.loads(tombstone_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ContextFault("context_key_destruction_unverified") from exc
            if (
                set(tombstone)
                != {
                    "schema_version",
                    "provider",
                    "key_ref",
                    "key_version",
                    "namespace_digest",
                    "wrapped_key_digest",
                    "destroyed_at",
                }
                or tombstone["schema_version"] != "ContextLocalKeyDestructionV1"
                or tombstone["provider"] != self.provider
                or tombstone["key_ref"] != key_ref
                or tombstone["key_version"] != key_version
                or tombstone["namespace_digest"] != namespace_digest
            ):
                raise ContextFault("context_key_destruction_unverified")
        if key_path.exists():
            # Authenticate the object before claiming that this exact key was destroyed.
            self.cipher(key_ref, namespace)
            wrapped_digest = hashlib.sha256(key_path.read_bytes()).hexdigest()
            if tombstone is None:
                tombstone = {
                    "schema_version": "ContextLocalKeyDestructionV1",
                    "provider": self.provider,
                    "key_ref": key_ref,
                    "key_version": key_version,
                    "namespace_digest": namespace_digest,
                    "wrapped_key_digest": wrapped_digest,
                    "destroyed_at": destroyed_at,
                }
                temporary = tombstone_path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    stream.write(canonical(tombstone))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, tombstone_path)
                _sync_directory(self.root)
            elif tombstone["wrapped_key_digest"] != wrapped_digest:
                raise ContextFault("context_key_destruction_unverified")
            key_path.unlink()
            _sync_directory(self.root)
        if tombstone is None or key_path.exists():
            raise ContextFault("context_key_destruction_unverified")
        # Re-read after unlink so a receipt never depends on an in-memory assumption.
        persisted = json.loads(tombstone_path.read_text(encoding="utf-8"))
        if persisted != tombstone:
            raise ContextFault("context_key_destruction_unverified")
        return {
            **persisted,
            "verified": True,
        }


class RetentionManager:
    """CAS-serialized hold/release/expiry over one ContextEngine."""

    def __init__(self, engine: Any, *, after_key_destroy: Callable[[], None] | None = None):
        self.engine = engine
        self.after_key_destroy = after_key_destroy

    def due(self, *, before: int | None = None, limit: int = 100) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ContextFault("context_retention_limit")
        cutoff = self.engine.now() if before is None else before
        if type(cutoff) is not int or cutoff < 0:
            raise ContextFault("context_retention_limit")
        with self.engine.store.connection() as db:
            return [
                {
                    "context_cycle_id": row["id"],
                    "owner_id": row["owner"],
                    "project_id": row["project"],
                    "expires_at": row["expires"],
                    "retention_class": row["retention_class"],
                    "cleanup_state": row["cleanup_state"],
                }
                for row in db.execute(
                    "SELECT * FROM cycles WHERE state IN ('SEALED','ABORTED') "
                    "AND hold=0 AND expires<=? "
                    "ORDER BY expires,id LIMIT ?",
                    (cutoff, limit),
                )
            ]

    def _deletion_replay(
        self,
        db,
        row,
        command: RetentionCommandV1,
        request_digest: str,
        stored_result: str,
    ) -> dict:
        try:
            envelope = json.loads(stored_result)
            payload = verify(
                envelope,
                {self.engine.signer.key_id: self.engine.signer.key.public_key()},
            )
            if payload.get("schema_version") == "ContextDeletionReceiptV2":
                receipt: ContextDeletionReceiptV1 | ContextDeletionReceiptV2 = (
                    ContextDeletionReceiptV2.model_validate(payload)
                )
            else:
                receipt = ContextDeletionReceiptV1.model_validate(payload)
        except (ContextFault, TypeError, ValueError) as exc:
            raise ContextFault("context_integrity_failure") from exc
        operation = db.execute(
            "SELECT request_digest,result FROM retention_operations "
            "WHERE cycle=? AND operation='expire' AND key=?",
            (row["id"], command.idempotency_key),
        ).fetchone()
        if (
            payload != receipt.model_dump(mode="json")
            or row["state"] != "EXPIRED"
            or row["cleanup_state"] != "COMPLETE"
            or row["cleanup_operation_key"] != command.idempotency_key
            or row["cleanup_request_digest"] != request_digest
            or row["deletion_receipt"] != canonical(envelope).decode()
            or receipt.context_cycle_id != row["id"]
            or receipt.segment_chain_root != row["root"]
            or receipt.context_seal_digest != row["seal_digest"]
            or receipt.retention_class != row["retention_class"]
            or receipt.retention_expired_at != row["expires"]
            or receipt.retention_operation_key != command.idempotency_key
            or receipt.retention_request_digest != request_digest
            or operation is None
            or operation["request_digest"] != request_digest
            or operation["result"] != canonical(envelope).decode()
        ):
            raise ContextFault("context_integrity_failure")
        remote_envelope = (
            json.loads(row["cleanup_remote_receipt"])
            if row["cleanup_remote_receipt"] is not None
            else None
        )
        if receipt.remote_deletion_receipt_digest != (
            digest(remote_envelope) if remote_envelope is not None else None
        ):
            raise ContextFault("context_integrity_failure")
        try:
            namespace = json.loads(row["binding"])["namespace"]
        except (TypeError, ValueError, KeyError) as exc:
            raise ContextFault("context_integrity_failure") from exc
        if remote_envelope is not None:
            try:
                self._verify_remote_deletion(
                    remote_envelope,
                    row,
                    namespace,
                    row["checkpoint_object_ref_digest"],
                    now=self.engine.now(),
                    require_fresh=False,
                )
            except ContextFault as exc:
                raise ContextFault("context_integrity_failure") from exc
        if receipt.key_destruction_receipt is not None:
            try:
                self._verify_key_destruction(
                    receipt.key_destruction_receipt.model_dump(mode="json"),
                    row["key_ref"],
                    row["key_version"],
                    namespace,
                    self.engine.now(),
                    command_issued_at=command.issued_at,
                    require_fresh=False,
                )
            except ContextFault as exc:
                raise ContextFault("context_integrity_failure") from exc
            # The service-signed terminal receipt records the provider
            # qualification used at the effect, while the embedded KMS receipt
            # remains independently verifiable. Exact lost-ACK replay must not
            # depend on a short-lived provider qualification that is correctly
            # retired after its last non-expired cycle is deleted.
        elif receipt.local_key_destruction is not None:
            local = (
                receipt.local_key_destruction.model_dump(mode="json")
                if hasattr(receipt.local_key_destruction, "model_dump")
                else receipt.local_key_destruction
            )
            if (
                set(local)
                != {
                    "schema_version",
                    "provider",
                    "key_ref",
                    "key_version",
                    "namespace_digest",
                    "wrapped_key_digest",
                    "destroyed_at",
                    "verified",
                }
                or local["schema_version"] != "ContextLocalKeyDestructionV1"
                or local["provider"] != FileCycleKeyProvider.provider
                or local["key_ref"] != row["key_ref"]
                or local["key_version"] != row["key_version"]
                or local["namespace_digest"] != digest(namespace)
                or local["verified"] is not True
            ):
                raise ContextFault("context_integrity_failure")
        if isinstance(receipt, ContextDeletionReceiptV2):
            registration_digest = (
                digest(json.loads(row["checkpoint_durability_receipt"]))
                if row["checkpoint_durability_receipt"] is not None
                else None
            )
            if (
                row["checkpoint_durability_legacy"]
                or receipt.checkpoint_durability_mode
                != row["checkpoint_durability_mode"]
                or receipt.checkpoint_durability_registration_receipt_digest
                != registration_digest
                or receipt.checkpoint_manifest_checksum
                != row["checkpoint_manifest_checksum"]
                or receipt.checkpoint_object_ref_digest
                != row["checkpoint_object_ref_digest"]
            ):
                raise ContextFault("context_integrity_failure")
        elif not row["checkpoint_durability_legacy"]:
            raise ContextFault("context_integrity_failure")
        return envelope

    def _hold_replay(
        self,
        db,
        row,
        command: RetentionCommandV1,
        request_digest: str,
        stored_result: str,
    ) -> dict:
        try:
            envelope = json.loads(stored_result)
            payload = verify(
                envelope,
                {self.engine.signer.key_id: self.engine.signer.key.public_key()},
            )
            receipt = ContextRetentionReceiptV1.model_validate(payload)
        except (ContextFault, TypeError, ValueError) as exc:
            raise ContextFault("context_integrity_failure") from exc
        operation = db.execute(
            "SELECT request_digest,result FROM retention_operations "
            "WHERE cycle=? AND operation=? AND key=?",
            (row["id"], command.operation, command.idempotency_key),
        ).fetchone()
        event_type = (
            "context.retention_held"
            if command.operation == "hold"
            else "context.retention_released"
        )
        historical_event = False
        for event_row in db.execute(
            "SELECT body FROM events WHERE cycle=?", (row["id"],)
        ):
            try:
                event = json.loads(event_row["body"])
            except (TypeError, ValueError):
                continue
            if (
                event.get("type") == event_type
                and event.get("context_cycle_id") == row["id"]
                and event.get("state_revision") == receipt.revision
                and event.get("reason") == receipt.reason_code
                and event.get("timestamp") == receipt.effective_at
            ):
                historical_event = True
                break
        if (
            payload != receipt.model_dump(mode="json")
            or receipt.operation != command.operation
            or receipt.context_cycle_id != row["id"]
            or receipt.reason_code != command.reason_code
            or not command.issued_at <= receipt.effective_at < command.expires_at
            or receipt.revision != command.expected_revision + 1
            or operation is None
            or operation["request_digest"] != request_digest
            or operation["result"] != canonical(envelope).decode()
            or not historical_event
        ):
            raise ContextFault("context_integrity_failure")
        return envelope

    def apply(self, capability: dict, signed_command: dict) -> dict:
        raw = verify(signed_command, self.engine.authority_keys)
        command = RetentionCommandV1.model_validate(raw)
        request_digest = digest(command)
        now = self.engine.now()

        with self.engine.store.transaction() as db:
            cap, row = self.engine._authorize(db, capability, "retention")
            if cap.role != "durability" or cap.namespace != command.namespace:
                raise ContextFault("context_retention_denied")
            replay = db.execute(
                "SELECT request_digest,result FROM retention_operations "
                "WHERE cycle=? AND operation=? AND key=?",
                (row["id"], command.operation, command.idempotency_key),
            ).fetchone()
            if replay is not None:
                if replay["request_digest"] != request_digest:
                    raise ContextFault("context_idempotency_conflict")
                if replay["result"] is not None:
                    if command.operation == "expire":
                        return self._deletion_replay(
                            db, row, command, request_digest, replay["result"]
                        )
                    return self._hold_replay(
                        db, row, command, request_digest, replay["result"]
                    )
            else:
                if not command.issued_at <= now < command.expires_at:
                    raise ContextFault("context_retention_authority_expired")
                if row["revision"] != command.expected_revision:
                    raise ContextFault("context_stale_state")
                db.execute(
                    "INSERT INTO retention_operations VALUES(?,?,?,?,NULL)",
                    (
                        row["id"],
                        command.operation,
                        command.idempotency_key,
                        request_digest,
                    ),
                )

            if command.operation == "hold":
                if replay is not None:
                    raise ContextFault("context_stale_state")
                if row["state"] == "EXPIRED" or row["cleanup_state"] != "NONE":
                    raise ContextFault("context_retention_denied")
                changed = db.execute(
                    "UPDATE cycles SET hold=1,hold_reason=?,revision=revision+1 "
                    "WHERE id=? AND revision=?",
                    (command.reason_code, row["id"], command.expected_revision),
                ).rowcount
                if changed != 1:
                    raise ContextFault("context_stale_state")
                self.engine._event(
                    db, row["id"], "context.retention_held", reason=command.reason_code
                )
                return self._finish_operation(
                    db,
                    row["id"],
                    command,
                    request_digest,
                    self.engine.signer.sign(
                        {
                            "schema_version": "ContextRetentionReceiptV1",
                            "operation": "hold",
                            "context_cycle_id": row["id"],
                            "reason_code": command.reason_code,
                            "effective_at": now,
                            "revision": command.expected_revision + 1,
                        }
                    ),
                )

            if command.operation == "release":
                if replay is not None:
                    raise ContextFault("context_stale_state")
                if row["state"] == "EXPIRED" or not row["hold"] or row["cleanup_state"] != "NONE":
                    raise ContextFault("context_retention_denied")
                changed = db.execute(
                    "UPDATE cycles SET hold=0,hold_reason=NULL,revision=revision+1 "
                    "WHERE id=? AND revision=?",
                    (row["id"], command.expected_revision),
                ).rowcount
                if changed != 1:
                    raise ContextFault("context_stale_state")
                self.engine._event(
                    db, row["id"], "context.retention_released", reason=command.reason_code
                )
                return self._finish_operation(
                    db,
                    row["id"],
                    command,
                    request_digest,
                    self.engine.signer.sign(
                        {
                            "schema_version": "ContextRetentionReceiptV1",
                            "operation": "release",
                            "context_cycle_id": row["id"],
                            "reason_code": command.reason_code,
                            "effective_at": now,
                            "revision": command.expected_revision + 1,
                        }
                    ),
                )

            if (
                row["state"] not in {"SEALED", "ABORTED"}
                or row["hold"]
                or row["expires"] > now
                or row["cleanup_state"] not in {"NONE", "PENDING"}
            ):
                raise ContextFault("context_retention_denied")
            if row["cleanup_state"] == "PENDING":
                cleanup_key = row["cleanup_operation_key"]
                cleanup_digest = row["cleanup_request_digest"]
                if cleanup_key is None and cleanup_digest is None:
                    # Rolling recovery for a PENDING row written before the
                    # cleanup-owner columns existed. Only one exact unfinished
                    # expiry can be adopted; ambiguity stays fail-closed.
                    pending = db.execute(
                        "SELECT key,request_digest FROM retention_operations "
                        "WHERE cycle=? AND operation='expire' AND result IS NULL",
                        (row["id"],),
                    ).fetchall()
                    if len(pending) != 1:
                        raise ContextFault("context_cleanup_in_progress")
                    cleanup_key = pending[0]["key"]
                    cleanup_digest = pending[0]["request_digest"]
                    db.execute(
                        "UPDATE cycles SET cleanup_operation_key=?,"
                        "cleanup_request_digest=? WHERE id=? AND cleanup_state='PENDING'",
                        (cleanup_key, cleanup_digest, row["id"]),
                    )
                elif cleanup_key is None or cleanup_digest is None:
                    raise ContextFault("context_cleanup_in_progress")
                if (
                    cleanup_key != command.idempotency_key
                    or cleanup_digest != request_digest
                ):
                    raise ContextFault("context_cleanup_in_progress")
            namespace = cap.namespace.model_dump()
            key_ref = row["key_ref"]
            key_version = row["key_version"]
            root = row["root"]
            seal_digest = row["seal_digest"]
            retention_class = row["retention_class"]
            checkpoint_durability_mode = row["checkpoint_durability_mode"]
            if checkpoint_durability_mode == "unregistered":
                if (
                    row["checkpoint"] != 0
                    or row["checkpoint_manifest_checksum"] is not None
                    or row["checkpoint_durability_command_digest"] is not None
                    or row["checkpoint_durability_receipt"] is not None
                ):
                    raise ContextFault("context_integrity_failure")
            elif checkpoint_durability_mode in {
                "local_ephemeral",
                "remote_registered",
            }:
                self.engine._verify_checkpoint_durability_registration(row)
            else:
                raise ContextFault("context_integrity_failure")
            remote_checkpoint_registered = (
                row["checkpoint_manifest_checksum"] is not None
                and checkpoint_durability_mode == "remote_registered"
            )
            object_ref_digest = command.checkpoint_object_ref_digest
            if row["cleanup_state"] == "NONE":
                if remote_checkpoint_registered:
                    if (
                        command.remote_deletion_receipt is None
                        or object_ref_digest is None
                    ):
                        raise ContextFault("context_remote_deletion_proof_required")
                    remote_envelope = command.remote_deletion_receipt.model_dump(mode="json")
                    remote_proof = self._verify_remote_deletion(
                        remote_envelope,
                        row,
                        namespace,
                        object_ref_digest,
                        now=now,
                        require_fresh=True,
                    )
                else:
                    if command.remote_deletion_receipt is not None:
                        raise ContextFault("context_remote_deletion_proof_mismatch")
                    remote_envelope = None
                    remote_proof = None
                changed = db.execute(
                    "UPDATE cycles SET cleanup_state='PENDING',"
                    "checkpoint_object_ref_digest=?,cleanup_remote_receipt=?,"
                    "cleanup_operation_key=?,cleanup_request_digest=?,"
                    "revision=revision+1 WHERE id=? AND revision=?",
                    (
                        object_ref_digest,
                        (
                            canonical(remote_envelope).decode()
                            if remote_envelope is not None
                            else None
                        ),
                        command.idempotency_key,
                        request_digest,
                        row["id"],
                        command.expected_revision,
                    ),
                ).rowcount
                if changed != 1:
                    raise ContextFault("context_stale_state")
                self.engine._event(
                    db, row["id"], "context.cleanup_started", reason=command.reason_code
                )
            else:
                if remote_checkpoint_registered:
                    try:
                        remote_envelope = json.loads(row["cleanup_remote_receipt"])
                    except (TypeError, ValueError) as exc:
                        raise ContextFault("context_remote_deletion_proof_invalid") from exc
                    if (
                        command.remote_deletion_receipt is None
                        or object_ref_digest is None
                        or digest(remote_envelope)
                        != digest(command.remote_deletion_receipt.model_dump(mode="json"))
                        or row["checkpoint_object_ref_digest"]
                        != command.checkpoint_object_ref_digest
                    ):
                        raise ContextFault("context_idempotency_conflict")
                    remote_proof = self._verify_remote_deletion(
                        remote_envelope,
                        row,
                        namespace,
                        object_ref_digest,
                        now=now,
                        require_fresh=False,
                    )
                else:
                    if command.remote_deletion_receipt is not None:
                        raise ContextFault("context_idempotency_conflict")
                    remote_envelope = None
                    remote_proof = None

        local_key_destruction = None
        key_destruction_envelope = None
        key_destruction_proof = None
        provider_receipt_digest = None
        cryptographic_erasure = False
        if key_ref is not None:
            if self.engine.cycle_keys is None:
                raise ContextFault("context_key_provider_required")
            if key_version is None:
                raise ContextFault("context_key_reference_invalid")
            if self.engine._provider_key_version(key_ref) != key_version:
                raise ContextFault("context_key_version_unavailable")
            provider = getattr(self.engine.cycle_keys, "provider", None)
            if provider == FileCycleKeyProvider.provider:
                local_key_destruction = self.engine.cycle_keys.destroy(
                    key_ref, namespace, now
                )
                if not local_key_destruction.get("verified"):
                    raise ContextFault("context_key_destruction_unverified")
            else:
                managed_ready, _, provider_receipt_digests = (
                    self.engine._managed_key_qualification(
                        expected_key_version=key_version
                    )
                )
                if not managed_ready:
                    raise ContextFault("context_managed_key_provider_unqualified")
                provider_receipt_digest = provider_receipt_digests[key_version]
                key_destruction_envelope = self.engine.cycle_keys.destroy(
                    key_ref, namespace, now
                )
                key_destruction_proof = self._verify_key_destruction(
                    key_destruction_envelope,
                    key_ref,
                    key_version,
                    namespace,
                    now,
                    command_issued_at=command.issued_at,
                    require_fresh=row["cleanup_state"] == "NONE",
                )
                cryptographic_erasure = bool(
                    key_destruction_proof
                    and (
                        row["checkpoint_manifest_checksum"] is None
                        or checkpoint_durability_mode == "local_ephemeral"
                        or remote_proof is not None
                    )
                )
        if self.after_key_destroy is not None:
            self.after_key_destroy()

        with self.engine.store.transaction() as db:
            row = self.engine._cycle(db, command.namespace.context_cycle_id)
            replay = db.execute(
                "SELECT request_digest,result FROM retention_operations "
                "WHERE cycle=? AND operation=? AND key=?",
                (row["id"], command.operation, command.idempotency_key),
            ).fetchone()
            if replay is None or replay["request_digest"] != request_digest:
                raise ContextFault("context_idempotency_conflict")
            if replay["result"] is not None:
                return self._deletion_replay(
                    db, row, command, request_digest, replay["result"]
                )
            if row["state"] not in {"SEALED", "ABORTED"} or row[
                "cleanup_state"
            ] != "PENDING":
                raise ContextFault("context_retention_denied")
            if (
                row["cleanup_operation_key"] != command.idempotency_key
                or row["cleanup_request_digest"] != request_digest
                or row["checkpoint_durability_mode"] != checkpoint_durability_mode
            ):
                raise ContextFault("context_cleanup_in_progress")
            db.execute("DELETE FROM postings WHERE cycle=?", (row["id"],))
            db.execute("DELETE FROM segments WHERE cycle=?", (row["id"],))
            db.execute("DELETE FROM operations WHERE cycle=?", (row["id"],))
            deletion_payload = {
                    "schema_version": "ContextDeletionReceiptV1",
                    "context_cycle_id": row["id"],
                    "segment_chain_root": root,
                    "context_seal_digest": seal_digest,
                    "retention_class": retention_class,
                    "retention_expired_at": row["expires"],
                    "deleted_at": now,
                    "reason_code": command.reason_code,
                    "retention_operation_key": command.idempotency_key,
                    "retention_request_digest": request_digest,
                    "logical_deletion": True,
                    "cryptographic_erasure": cryptographic_erasure,
                    "local_key_destruction": local_key_destruction,
                    "key_destruction_receipt": key_destruction_envelope,
                    "key_destruction_receipt_digest": (
                        digest(key_destruction_envelope)
                        if key_destruction_envelope is not None
                        else None
                    ),
                    "managed_key_provider_receipt_digest": (
                        provider_receipt_digest
                        if key_destruction_envelope is not None
                        else None
                    ),
                    "remote_deletion_receipt_digest": (
                        digest(remote_envelope) if remote_envelope is not None else None
                    ),
                    "checkpoint_manifest_checksum": (
                        remote_proof.manifest_checksum if remote_proof is not None else None
                    ),
                    "checkpoint_object_ref_digest": (
                        remote_proof.object_ref_digest if remote_proof is not None else None
                    ),
                }
            if not row["checkpoint_durability_legacy"]:
                registration_receipt_digest = (
                    digest(json.loads(row["checkpoint_durability_receipt"]))
                    if row["checkpoint_durability_receipt"] is not None
                    else None
                )
                deletion_payload.update(
                    schema_version="ContextDeletionReceiptV2",
                    checkpoint_manifest_checksum=row["checkpoint_manifest_checksum"],
                    checkpoint_durability_mode=checkpoint_durability_mode,
                    checkpoint_durability_registration_receipt_digest=(
                        registration_receipt_digest
                    ),
                )
                deletion_payload = ContextDeletionReceiptV2.model_validate(
                    deletion_payload
                ).model_dump(mode="json")
            receipt = self.engine.signer.sign(deletion_payload)
            db.execute(
                "UPDATE cycles SET state='EXPIRED',authority=NULL,snapshot=NULL,"
                "cleanup_state='COMPLETE',deletion_receipt=?,revision=revision+1 WHERE id=?",
                (canonical(receipt).decode(), row["id"]),
            )
            self.engine._event(
                db,
                row["id"],
                "context.expired",
                root=root,
                deletion_receipt_digest=digest(receipt),
                cryptographic_erasure=cryptographic_erasure,
            )
            return self._finish_operation(
                db, row["id"], command, request_digest, receipt
            )

    def _verify_remote_deletion(
        self,
        envelope: dict,
        row: Any,
        namespace: dict,
        object_ref_digest: str,
        *,
        now: int,
        require_fresh: bool,
    ) -> RemoteDeletionReceiptV1:
        if not self.engine.remote_deletion_keys:
            raise ContextFault("context_remote_deletion_keys_required")
        try:
            proof = RemoteDeletionReceiptV1.model_validate(
                verify(envelope, self.engine.remote_deletion_keys)
            )
        except (ValueError, TypeError) as exc:
            raise ContextFault("context_remote_deletion_proof_invalid") from exc
        if (
            proof.namespace.model_dump() != namespace
            or row["checkpoint_durability_mode"] != "remote_registered"
            or row["checkpoint_manifest_checksum"] is None
            or proof.manifest_checksum != row["checkpoint_manifest_checksum"]
            or proof.object_ref_digest != object_ref_digest
        ):
            raise ContextFault("context_remote_deletion_proof_mismatch")
        if require_fresh and not proof.issued_at <= now < proof.expires_at:
            raise ContextFault("context_remote_deletion_proof_expired")
        return proof

    def _verify_key_destruction(
        self,
        envelope: dict,
        key_ref: str,
        key_version: str,
        namespace: dict,
        now: int,
        *,
        command_issued_at: int,
        require_fresh: bool,
    ) -> KeyDestructionReceiptV1:
        if not self.engine.key_destruction_keys:
            raise ContextFault("context_key_destruction_keys_required")
        try:
            proof = KeyDestructionReceiptV1.model_validate(
                verify(envelope, self.engine.key_destruction_keys)
            )
        except (ContextFault, ValueError, TypeError) as exc:
            raise ContextFault("context_key_destruction_unverified") from exc
        if (
            proof.provider != getattr(self.engine.cycle_keys, "provider", None)
            or proof.key_ref != key_ref
            or proof.namespace_digest != digest(namespace)
            or proof.key_version != key_version
            or not proof.destroyed_at <= now
            or proof.destroyed_at < command_issued_at
            or (require_fresh and not proof.issued_at <= now < proof.expires_at)
        ):
            raise ContextFault("context_key_destruction_unverified")
        return proof

    @staticmethod
    def _finish_operation(db, cycle, command, request_digest, result):
        changed = db.execute(
            "UPDATE retention_operations SET result=? WHERE cycle=? AND operation=? "
            "AND key=? AND request_digest=? AND result IS NULL",
            (
                canonical(result).decode(),
                cycle,
                command.operation,
                command.idempotency_key,
                request_digest,
            ),
        ).rowcount
        if changed != 1:
            raise ContextFault("context_stale_state")
        return result


__all__ = ["FileCycleKeyProvider", "RetentionManager"]
