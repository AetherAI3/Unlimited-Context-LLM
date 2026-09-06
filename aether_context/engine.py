"""Model-independent context plane. All external calls require signed capabilities."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from .contracts import (
    AppendRequestV1,
    CapabilityV1,
    ClosureProofV1,
    ContextBindingV1,
    ContextProfileV1,
    RetrieveRequestV1,
    canonical,
    digest,
)
from .crypto import ContextFault, EnvelopeCipher, ReceiptSigner, verify, require_disjoint_keys
from .policy import NOTICE, filter_text, token_bound, visible, words, writable
from .storage_v2 import SegmentStoreV2

ZERO = "0" * 64
TERMINAL = {"SEALED", "EXPIRED", "ABORTED", "QUARANTINED"}


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
    ):
        if profile.index_version != sqlite3.sqlite_version:
            raise ContextFault("context_index_version_mismatch")
        if not authority_keys or not proof_keys:
            raise ContextFault("context_verification_keys_required")
        require_disjoint_keys(authority_keys, proof_keys, {signer.key_id: signer.key.public_key()})
        self.profile, self.cipher, self.signer = profile, cipher, signer
        self.authority_keys, self.proof_keys, self.clock = authority_keys, proof_keys, clock
        self.store = SegmentStoreV2(path, profile.resident_cache_bytes)
        self._audit_startup()

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
                    binding = ContextBindingV1.model_validate_json(cycle["binding"])
                    if digest(binding) != cycle["binding_digest"]:
                        raise ContextFault("context_integrity_failure")
                    ns = binding.namespace.model_dump()
                    self._snapshot(cycle, ns)
                    authority = self.cipher.decrypt(
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
                        value = self._record(record, ns)
                        postings = {
                            row[0]
                            for row in db.execute(
                                "SELECT term FROM postings WHERE cycle=? AND seq=?",
                                (cycle["id"], count),
                            )
                        }
                        if postings != {
                            self.cipher.token(ns, word) for word in words(value["text"])
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

    def health(self) -> dict:
        free = shutil.disk_usage(self.store.path.parent).free
        with self.store.connection() as db:
            db.execute("SELECT count(*) FROM cycles").fetchone()
        return {
            "schema_version": "ContextHealthV1",
            "ready": free >= self.profile.disk_free_floor,
            "profile_digest": digest(self.profile),
            "index_implementation": self.profile.index_implementation,
            "index_version": sqlite3.sqlite_version,
            "disk_free_bytes": free,
            "resident_cache_limit_bytes": self.profile.resident_cache_bytes,
            "estimated_reachable_tokens": None,
            "reach_claim_enabled": False,
        }

    def _space(self) -> None:
        if not self.health()["ready"]:
            raise ContextFault("context_disk_watermark")

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
        request = verify(signed_reservation, self.authority_keys)
        if set(request) != {
            "operation",
            "owner_id",
            "project_id",
            "idempotency_key",
            "request_digest",
            "profile_digest",
            "expires_at",
        }:
            raise ContextFault("context_reservation_schema")
        if request["operation"] != "reserve" or request["profile_digest"] != digest(self.profile):
            raise ContextFault("context_profile_mismatch")
        if not self.now() < request["expires_at"] <= self.now() + 3600:
            raise ContextFault("context_reservation_expired")
        for field in ("owner_id", "project_id", "idempotency_key", "request_digest"):
            if not isinstance(request[field], str) or not 1 <= len(request[field]) <= 200:
                raise ContextFault("context_reservation_schema")
        self._space()
        identity = {k: request[k] for k in ("owner_id", "project_id", "idempotency_key")}
        cycle = "ctx_" + digest(identity)[:32]
        with self.store.transaction() as db:
            for expired in db.execute(
                "SELECT id FROM cycles WHERE state='RESERVED' AND expires<=?", (self.now(),)
            ).fetchall():
                db.execute(
                    "UPDATE cycles SET state='EXPIRED',revision=revision+1 WHERE id=?",
                    (expired[0],),
                )
                self._event(db, expired[0], "context.reservation_expired")
            existing = db.execute("SELECT * FROM cycles WHERE id=?", (cycle,)).fetchone()
            if existing is not None:
                if existing["request_digest"] != request["request_digest"]:
                    raise ContextFault("context_idempotency_conflict")
                if existing["state"] in TERMINAL:
                    raise ContextFault("context_reservation_terminal")
            else:
                count = db.execute(
                    "SELECT count(*) FROM cycles WHERE owner=? AND state NOT IN ('SEALED','EXPIRED','ABORTED','QUARANTINED')",
                    (request["owner_id"],),
                ).fetchone()[0]
                if count >= self.profile.max_concurrent_cycles:
                    raise ContextFault("context_cycle_quota")
                db.execute(
                    "INSERT INTO cycles(id,owner,project,request_key,request_digest,profile,state,root,expires) VALUES(?,?,?,?,?,?,'RESERVED',?,?)",
                    (
                        cycle,
                        request["owner_id"],
                        request["project_id"],
                        request["idempotency_key"],
                        request["request_digest"],
                        digest(self.profile),
                        ZERO,
                        request["expires_at"],
                    ),
                )
                self._event(db, cycle, "context.reserved")
            return self.signer.sign(
                {
                    "schema_version": "ContextReservationReceiptV1",
                    "context_cycle_id": cycle,
                    "context_bucket_id": "bucket_" + cycle[4:],
                    "request_digest": request["request_digest"],
                    "profile_digest": digest(self.profile),
                }
            )

    def bind(self, signed_binding: dict, authority: dict, project_snapshot: dict) -> dict:
        binding = ContextBindingV1.model_validate(verify(signed_binding, self.authority_keys))
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
            if row["binding"]:
                if row["binding_digest"] != digest(binding):
                    raise ContextFault("context_binding_conflict")
            else:
                if row["state"] != "RESERVED" or row["expires"] <= self.now():
                    raise ContextFault("context_state_conflict")
                db.execute(
                    "UPDATE cycles SET binding=?,binding_digest=?,authority=?,snapshot=?,state='BOUND',expires=?,revision=revision+1 WHERE id=?",
                    (
                        canonical(binding).decode(),
                        digest(binding),
                        self.cipher.encrypt(authority, ns),
                        self.cipher.encrypt(project_snapshot, {"namespace": ns, "plane": "P1"}),
                        binding.expires_at,
                        cycle,
                    ),
                )
                self._event(db, cycle, "context.bound", binding_digest=digest(binding))
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
                    "binding_digest": digest(binding),
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
        if cap.namespace.model_dump() != binding["namespace"]:
            raise ContextFault("context_namespace_denied")
        if (
            operation not in cap.operations
            or cap.expires_at <= self.now()
            or (operation not in {"expire", "status"} and cap.expires_at > binding["expires_at"])
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
        if row["state"] in {"EXPIRED", "ABORTED", "QUARANTINED"} and operation != "status":
            raise ContextFault("context_terminal")
        return cap, row

    def _operation_aad(self, cycle: str, operation: str, key: str) -> dict:
        return {"cycle": cycle, "operation": operation, "key": key}

    def _replay(self, db, cycle: str, operation: str, key: str, request_digest: str):
        row = db.execute(
            "SELECT * FROM operations WHERE cycle=? AND operation=? AND key=?",
            (cycle, operation, key),
        ).fetchone()
        if row is None:
            return None
        if row["request_digest"] != request_digest:
            raise ContextFault("context_idempotency_conflict")
        return self.cipher.decrypt(
            row["result"],
            self._operation_aad(cycle, operation, key),
            self.profile.max_checkpoint_bytes * (2 if operation == "checkpoint" else 1),
        )

    def _remember(
        self, db, cycle: str, operation: str, key: str, request_digest: str, result: dict
    ):
        limit = self.profile.max_operations_per_cycle + (
            4 if operation in {"checkpoint", "seal"} else 0
        )
        if (
            db.execute("SELECT count(*) FROM operations WHERE cycle=?", (cycle,)).fetchone()[0]
            >= limit
        ):
            raise ContextFault("context_operation_quota")
        blob = self.cipher.encrypt(result, self._operation_aad(cycle, operation, key))
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
        value = self.cipher.decrypt(
            cycle["snapshot"], {"namespace": ns, "plane": "P1"}, self.profile.max_snapshot_bytes
        )
        if digest(value) != json.loads(cycle["binding"])["graph_checksum"]:
            raise ContextFault("context_integrity_failure")
        return value

    def append(self, capability: dict, request: AppendRequestV1) -> dict:
        text = filter_text(request.text)
        size = len(text.encode())
        self._space()
        with self._transaction(capability) as db:
            cap, cycle = self._authorize(db, capability, "append")
            writable(cap, request.plane)
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
            replay = self._replay(db, cycle["id"], "append", request.idempotency_key, req_digest)
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
            ns = cap.namespace.model_dump()
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
            encrypted = self.cipher.encrypt(
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
                    (cycle["id"], self.cipher.token(ns, word), seq),
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
                db, cycle["id"], "append", request.idempotency_key, req_digest, result
            )

    def _record(self, row, namespace: dict) -> dict:
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
        body = self.cipher.decrypt(
            row["payload"],
            {"namespace": namespace, "record_id": row["id"], "metadata_digest": row["root"]},
            self.profile.max_record_bytes * 2,
        )
        text = filter_text(body["text"])
        if hashlib.sha256(text.encode()).hexdigest() != row["digest"]:
            raise ContextFault("context_integrity_failure")
        return {"metadata": metadata, "text": text}

    def retrieve(self, capability: dict, request: RetrieveRequestV1) -> dict:
        filter_text(request.query)
        with self._transaction(capability) as db:
            cap, cycle = self._authorize(db, capability, "retrieve")
            req_digest = digest(
                {
                    "request": request.model_dump(),
                    "principal": cap.principal_id,
                    "lane": cap.lane_id,
                    "task": cap.task_id,
                    "role": cap.role,
                }
            )
            replay = self._replay(db, cycle["id"], "retrieve", request.turn_id, req_digest)
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
            ns = cap.namespace.model_dump()
            terms = [self.cipher.token(ns, word) for word in words(request.query)[:32]]
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
                entry = self._record(row, ns)
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
            return self._remember(db, cycle["id"], "retrieve", request.turn_id, req_digest, result)

    def checkpoint(self, capability: dict, idempotency_key: str) -> dict:
        self._space()
        with self._transaction(capability) as db:
            cap, cycle = self._authorize(db, capability, "checkpoint")
            if cap.role not in {"coordinator", "verifier", "durability"}:
                raise ContextFault("context_role_denied")
            replay = self._replay(
                db, cycle["id"], "checkpoint", idempotency_key, digest({"key": idempotency_key})
            )
            if replay:
                return replay
            if cycle["state"] not in {"ACTIVE", "SEALING"}:
                raise ContextFault("context_not_active")
            ns = cap.namespace.model_dump()
            records: list[dict] = []
            previous = ZERO
            for row in db.execute(
                "SELECT * FROM segments WHERE cycle=? ORDER BY seq", (cycle["id"],)
            ):
                if row["previous"] != previous or row["seq"] != len(records) + 1:
                    raise ContextFault("context_chain_corrupt")
                entry = self._record(row, ns)
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
                "authority": self.cipher.decrypt(
                    cycle["authority"], ns, self.profile.max_record_bytes
                ),
                "project_snapshot": self._snapshot(cycle, ns),
                "records": records,
                "cursor": cycle["cursor"],
                "root": previous,
                "request_key": cycle["request_key"],
                "request_digest": cycle["request_digest"],
                "fence": cycle["fence"],
                "checkpoint": cycle["checkpoint"] + 1,
            }
            snapshot["state"] = cycle["state"]
            # Replays survive host replacement too, including immutable capsules.
            import base64

            snapshot["operations"] = [
                {**dict(row), "result": base64.b64encode(row["result"]).decode()}
                for row in db.execute(
                    "SELECT * FROM operations WHERE cycle=? AND operation!='checkpoint'",
                    (cycle["id"],),
                )
            ]
            if len(canonical(snapshot)) > self.profile.max_checkpoint_bytes:
                raise ContextFault("context_checkpoint_quota")
            pack = self.cipher.encrypt(snapshot, {"namespace": ns, "domain": "checkpoint/v1"})
            checksum = hashlib.sha256(pack).hexdigest()
            number = cycle["checkpoint"] + 1
            receipt = self.signer.sign(
                {
                    "schema_version": "ContextCheckpointReceiptV1",
                    "namespace": ns,
                    "checkpoint_number": number,
                    "cursor": cycle["cursor"],
                    "segment_chain_root": previous,
                    "manifest_checksum": checksum,
                    "durability": "local",
                    "bytes": len(pack),
                    "records": len(records),
                    "policy_digest": cap.policy_digest,
                }
            )
            result = {"receipt": receipt, "pack": base64.b64encode(pack).decode()}
            db.execute(
                "UPDATE cycles SET checkpoint=?,revision=revision+1 WHERE id=?",
                (number, cycle["id"]),
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
            )

    def hydrate(self, signed_grant: dict, receipt: dict, pack: bytes, receipt_keys: dict) -> dict:
        """Host replacement needs a fresh fence and the exact cloud-selected root."""
        grant = verify(signed_grant, self.authority_keys)
        if (
            set(grant) != {"operation", "namespace", "manifest_checksum", "fence", "expires_at"}
            or grant["operation"] != "hydrate"
        ):
            raise ContextFault("context_hydrate_grant_invalid")
        if grant["expires_at"] <= self.now():
            raise ContextFault("context_capability_denied")
        claim = verify(receipt, receipt_keys)
        checksum = hashlib.sha256(pack).hexdigest()
        if (
            checksum != grant["manifest_checksum"]
            or checksum != claim["manifest_checksum"]
            or claim["namespace"] != grant["namespace"]
            or len(pack) > self.profile.max_checkpoint_bytes
        ):
            raise ContextFault("context_integrity_failure")
        snapshot = self.cipher.decrypt(
            pack,
            {"namespace": grant["namespace"], "domain": "checkpoint/v1"},
            self.profile.max_checkpoint_bytes,
        )
        binding = ContextBindingV1.model_validate(snapshot["binding"])
        if (
            binding.namespace.model_dump() != grant["namespace"]
            or binding.profile_digest != digest(self.profile)
            or grant["fence"] <= snapshot["fence"]
            or binding.expires_at <= self.now()
            or snapshot["root"] != claim["segment_chain_root"]
            or snapshot["cursor"] != claim["cursor"]
        ):
            raise ContextFault("context_hydrate_grant_invalid")
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
        import base64

        with self.store.transaction() as db:
            existing = db.execute("SELECT * FROM cycles WHERE id=?", (cycle,)).fetchone()
            if existing:
                if existing["root"] == previous and existing["fence"] == grant["fence"]:
                    return self.signer.sign(
                        {
                            "operation": "hydrate",
                            "context_cycle_id": cycle,
                            "root": previous,
                            "fence": grant["fence"],
                        }
                    )
                raise ContextFault("context_hydrate_conflict")
            db.execute(
                "INSERT INTO cycles(id,owner,project,request_key,request_digest,profile,state,fence,binding,binding_digest,authority,snapshot,cursor,root,bytes,expires,checkpoint) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cycle,
                    ns["owner_id"],
                    ns["project_id"],
                    snapshot["request_key"],
                    snapshot["request_digest"],
                    digest(self.profile),
                    snapshot["state"],
                    grant["fence"],
                    canonical(binding).decode(),
                    digest(binding),
                    self.cipher.encrypt(snapshot["authority"], ns),
                    self.cipher.encrypt(
                        snapshot["project_snapshot"], {"namespace": ns, "plane": "P1"}
                    ),
                    snapshot["cursor"],
                    previous,
                    sum(len(x["text"].encode()) for x in records),
                    binding.expires_at,
                    snapshot["checkpoint"],
                ),
            )
            for record in records:
                m = record["metadata"]
                blob = self.cipher.encrypt(
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
                        (cycle, self.cipher.token(ns, word), m["sequence"]),
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
            return self.signer.sign(
                {
                    "operation": "hydrate",
                    "context_cycle_id": cycle,
                    "root": previous,
                    "fence": grant["fence"],
                }
            )

    def freeze(self, capability: dict) -> dict:
        with self.store.transaction() as db:
            cap, cycle = self._authorize(db, capability, "seal")
            if cap.role != "verifier" or cycle["state"] not in {"ACTIVE", "SEALING"}:
                raise ContextFault("context_freeze_denied")
            if cycle["state"] == "ACTIVE":
                db.execute(
                    "UPDATE cycles SET state='SEALING',revision=revision+1 WHERE id=?",
                    (cycle["id"],),
                )
                self._event(db, cycle["id"], "context.sealing")
            return self.signer.sign(
                {
                    "schema_version": "ContextFreezeReceiptV1",
                    "context_cycle_id": cycle["id"],
                    "cursor": cycle["cursor"],
                    "root": cycle["root"],
                    "binding_digest": cycle["binding_digest"],
                }
            )

    def seal(self, capability: dict, signed_proof: dict, checkpoint_receipt: dict) -> dict:
        proof = ClosureProofV1.model_validate(verify(signed_proof, self.proof_keys))
        with self._transaction(capability) as db:
            cap, cycle = self._authorize(db, capability, "seal")
            if cap.role != "verifier":
                raise ContextFault("context_role_denied")
            req_digest = digest(proof)
            replay = self._replay(db, cycle["id"], "seal", "terminal", req_digest)
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
            checkpoint = verify(
                checkpoint_receipt, {self.signer.key_id: self.signer.key.public_key()}
            )
            if (
                checkpoint.get("schema_version") != "ContextCheckpointReceiptV1"
                or checkpoint["namespace"] != cap.namespace.model_dump()
                or checkpoint["cursor"] != cycle["cursor"]
                or checkpoint["segment_chain_root"] != cycle["root"]
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
                meta = self._record(row, cap.namespace.model_dump())["metadata"]
                if not meta["evidence_refs"] or meta["source_class"] in {
                    "model_note",
                    "legacy_untrusted",
                }:
                    raise ContextFault("context_independent_evidence_required")
                promoted.append(
                    {
                        "record_id": record_id,
                        "content_digest": meta["content_digest"],
                        "evidence_refs": meta["evidence_refs"],
                    }
                )
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
                }
            )
            db.execute(
                "UPDATE cycles SET state='SEALED',expires=?,revision=revision+1 WHERE id=?",
                (expiry, cycle["id"]),
            )
            self._event(
                db,
                cycle["id"],
                "context.sealed",
                seal_digest=digest(result),
                accepted_count=len(promoted),
            )
            return self._remember(db, cycle["id"], "seal", "terminal", req_digest, result)

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
            transitions = {
                "degrade": ({"ACTIVE"}, "DEGRADED"),
                "recover": ({"DEGRADED"}, "ACTIVE"),
                "abort": ({"RESERVED", "BOUND", "ACTIVE", "DEGRADED"}, "ABORTED"),
                "quarantine": ({"ACTIVE", "DEGRADED", "SEALING"}, "QUARANTINED"),
            }
            op = command["operation"]
            if op == "fence":
                if command["fence"] <= row["fence"] or row["state"] in TERMINAL:
                    raise ContextFault("context_stale_fence")
                db.execute(
                    "UPDATE cycles SET fence=?,revision=revision+1 WHERE id=?",
                    (command["fence"], row["id"]),
                )
            elif op in transitions and row["state"] in transitions[op][0]:
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
                    "checkpoint_number": row["checkpoint"],
                    "expires_at": row["expires"],
                    "estimated_reachable_tokens": None,
                    "native_model_window": None,
                    "events": events,
                }
            )

    def expire(self, capability: dict) -> dict:
        with self.store.transaction() as db:
            cap, row = self._authorize(db, capability, "expire")
            if (
                cap.role != "durability"
                or row["state"] != "SEALED"
                or row["hold"]
                or row["expires"] > self.now()
            ):
                raise ContextFault("context_retention_denied")
            # This is logical deletion. Cryptographic erasure is performed by
            # the managed per-cycle key provider and separately receipted.
            db.execute("DELETE FROM postings WHERE cycle=?", (row["id"],))
            db.execute("DELETE FROM segments WHERE cycle=?", (row["id"],))
            db.execute("DELETE FROM operations WHERE cycle=? AND operation!='seal'", (row["id"],))
            db.execute(
                "UPDATE cycles SET state='EXPIRED',authority=NULL,snapshot=NULL,revision=revision+1 WHERE id=?",
                (row["id"],),
            )
            self._event(db, row["id"], "context.expired", root=row["root"])
            return self.signer.sign(
                {
                    "schema_version": "ContextDeletionReceiptV1",
                    "context_cycle_id": row["id"],
                    "root": row["root"],
                    "logical_deletion": True,
                    "cryptographic_erasure": False,
                }
            )
