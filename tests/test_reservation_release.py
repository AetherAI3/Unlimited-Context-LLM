from concurrent.futures import ThreadPoolExecutor

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from aether_context.contracts import (
    CapabilityV1,
    ClosureProofV1,
    ContextBindingCommandV2,
    ContextBindingCommandV3,
    ContextBindingV1,
    NamespaceV1,
    ContextReservationAbortCommandV1,
    ContextReservationAbortExpiryCommandV1,
    ContextReservationAbortReceiptV1,
    ContextReservationCommandV2,
    ContextReservationCommandV3,
    ContextReservationExpiryCommandV1,
    ContextReservationExpiryReceiptV1,
    ContextReservationReceiptV1,
    ContextReservationReceiptV2,
    ContextReservationReceiptV3,
    ContextReservationReleaseCommandV1,
    ContextReservationReleaseReceiptV1,
    digest,
)
from aether_context.crypto import ContextFault, ReceiptSigner, verify
from aether_context.service import ContextService
from test_hosted_context import hosted  # noqa: F401


def _reservation(engine, authority, clock, *, key="retry-reservation", ttl=100):
    payload = ContextReservationCommandV2(
        owner_id="retry-owner",
        project_id="retry-project",
        idempotency_key=key,
        request_digest=digest(key),
        profile_digest=digest(engine.profile),
        expires_at=clock[0] + ttl,
        released_revision=None,
    ).model_dump(mode="json")
    signed = authority.sign(payload)
    receipt = ContextService(engine).dispatch(
        {"operation": "reserve_v2", "reservation": signed}
    )
    claim = ContextReservationReceiptV2.model_validate(
        verify(receipt, {engine.signer.key_id: engine.signer.key.public_key()})
    )
    return payload, signed, receipt, claim


def _release_payload(engine, claim, clock, *, key="retry-reservation", **updates):
    payload = ContextReservationReleaseCommandV1(
        context_cycle_id=claim.context_cycle_id,
        owner_id="retry-owner",
        project_id="retry-project",
        idempotency_key=key,
        request_digest=digest(key),
        profile_digest=digest(engine.profile),
        expected_revision=claim.revision,
        reason_code="pre_dispatch_authorization_failed",
        dispatch_started=False,
        authorization_failure_receipt_digest=digest("denied-one-use-authority"),
        issued_at=clock[0],
        expires_at=clock[0] + 100,
    ).model_dump(mode="json")
    payload.update(updates)
    return payload


def _abort_payload(
    engine,
    claim,
    reservation_receipt,
    clock,
    *,
    key="retry-reservation",
    **updates,
):
    payload = ContextReservationAbortCommandV1(
        context_cycle_id=claim.context_cycle_id,
        owner_id="retry-owner",
        project_id="retry-project",
        objective_id="retry-objective",
        idempotency_key=key,
        request_digest=digest(key),
        profile_digest=digest(engine.profile),
        expected_revision=claim.revision,
        reason_code="timeout",
        dispatch_started=True,
        objective_dispatch_receipt_digest=digest("objective-dispatch-started"),
        reservation_receipt=reservation_receipt,
        issued_at=clock[0],
        expires_at=clock[0] + 100,
    ).model_dump(mode="json")
    payload.update(updates)
    return payload


def _no_dispatch_receipt(
    engine,
    authority,
    claim,
    reservation_receipt,
    clock,
    *,
    key="retry-reservation",
    **updates,
):
    snapshot = {
        "owner_user_id": "retry-owner",
        "objective_id": "objective-" + key,
        "idempotency_key": key,
        "request_digest": "sha256:" + digest(key),
        "context_cycle_id": claim.context_cycle_id,
        "reservation_receipt_digest": digest(reservation_receipt),
        "reservation_revision": claim.revision,
        "state": "closed_without_dispatch",
        "revision": 7,
        "canonical_result_state": "planned",
        "canonical_result_digest": digest(
            {"canonical": key, "state": "planned"}
        ),
    }
    snapshot.update(updates.pop("ledger_snapshot_updates", {}))
    payload = {
        "schema_version": "ContextInvocationNoDispatchReceiptV1",
        "operation": "close_without_dispatch",
        "ledger_snapshot": snapshot,
        "ledger_snapshot_digest": digest(snapshot),
        "issued_at": clock[0],
        "expires_at": clock[0] + 100,
    }
    payload.update(updates)
    return authority.sign(payload)


def _expiry_payload(
    engine,
    authority,
    claim,
    reservation_receipt,
    clock,
    *,
    key="retry-reservation",
    **updates,
):
    no_dispatch = _no_dispatch_receipt(
        engine, authority, claim, reservation_receipt, clock, key=key
    )
    payload = ContextReservationExpiryCommandV1(
        context_cycle_id=claim.context_cycle_id,
        context_bucket_id=claim.context_bucket_id,
        objective_id="objective-" + key,
        owner_id="retry-owner",
        project_id="retry-project",
        idempotency_key=key,
        request_digest=digest(key),
        profile_digest=digest(engine.profile),
        expected_revision=claim.revision,
        reason_code="reservation_timeout",
        reservation_receipt=reservation_receipt,
        invocation_no_dispatch_receipt=no_dispatch,
        invocation_no_dispatch_receipt_digest=digest(no_dispatch),
        issued_at=clock[0],
        expires_at=clock[0] + 100,
    ).model_dump(mode="json")
    payload.update(updates)
    return payload


def _binding_command(engine, authority, reservation_receipt, claim, clock):
    namespace = NamespaceV1(
        owner_id="retry-owner",
        project_id="retry-project",
        objective_id="retry-objective",
        context_cycle_id=claim.context_cycle_id,
    )
    authority_payload = {"objective": "retry safely", "paths": ["src"]}
    snapshot = {
        "project_id": "retry-project",
        "graph_id": "retry-graph",
        "policy_digest": digest("retry-policy"),
        "nodes": [],
    }
    binding = ContextBindingV1(
        namespace=namespace,
        context_bucket_id=claim.context_bucket_id,
        repository_id="retry-repository",
        repo_main_sha="a" * 40,
        project_graph_id="retry-graph",
        graph_revision=1,
        graph_checksum=digest(snapshot),
        plan_digest=digest("retry-plan"),
        execution_profile_digest=digest("retry-execution-profile"),
        shared_ir_digest=digest("retry-ir"),
        authorization_receipt_ref="retry-authorization",
        authorization_digest=digest(authority_payload),
        policy_digest=digest("retry-policy"),
        redaction_digest=digest("retry-redaction"),
        profile_digest=digest(engine.profile),
        captain_binding_id="retry-captain",
        created_at=clock[0],
        expires_at=clock[0] + 1000,
    )
    command_model = (
        ContextBindingCommandV3
        if isinstance(claim, ContextReservationReceiptV3)
        else ContextBindingCommandV2
    )
    command = command_model(
        binding=binding,
        reservation_receipt=reservation_receipt,
        expected_reservation_revision=claim.revision,
    )
    return authority.sign(command.model_dump(mode="json")), authority_payload, snapshot, binding


def test_signed_release_lost_ack_concurrency_revival_and_post_bind_denial(hosted):  # noqa: F811
    engine, _, _, _, authority, _, _, clock = hosted
    _, original_reservation, reservation_receipt, claim = _reservation(
        engine, authority, clock
    )
    delayed_binding = _binding_command(
        engine, authority, reservation_receipt, claim, clock
    )
    command_payload = _release_payload(engine, claim, clock)
    command = authority.sign(command_payload)
    service = ContextService(engine)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: service.dispatch(
                    {"operation": "release_reservation", "command": command}
                ),
                range(4),
            )
        )
    assert all(item == results[0] for item in results)
    released = results[0]
    release_claim = ContextReservationReleaseReceiptV1.model_validate(
        verify(released, {engine.signer.key_id: engine.signer.key.public_key()})
    )
    assert release_claim.release_command_digest == digest(command_payload)
    assert release_claim.revision == claim.revision + 1
    # The original still-valid reserve authority cannot be replayed as a new attempt.
    with pytest.raises(ContextFault, match="reservation_terminal"):
        engine.reserve_v2(original_reservation)

    # An exact retry remains recoverable after the short-lived command expires.
    clock[0] += 101
    assert engine.release_reservation(command) == released
    conflicting_payload = dict(command_payload)
    conflicting_payload["authorization_failure_receipt_digest"] = digest(
        "different-auth-failure"
    )
    with pytest.raises(ContextFault, match="reservation_release_conflict"):
        engine.release_reservation(authority.sign(conflicting_payload))
    fresh_payload = {
        **_reservation_payload_from_claim(engine, claim),
        "expires_at": clock[0] + 100,
        "released_revision": release_claim.revision,
    }
    signed_revival = authority.sign(fresh_payload)
    revived = engine.reserve_v2(signed_revival)
    revived_claim = ContextReservationReceiptV2.model_validate(
        verify(revived, {engine.signer.key_id: engine.signer.key.public_key()})
    )
    assert revived_claim.context_cycle_id == claim.context_cycle_id
    assert revived_claim.revision == release_claim.revision + 1
    assert engine.reserve_v2(signed_revival) == revived
    stale_payload = dict(command_payload)
    stale_payload.update(issued_at=clock[0], expires_at=clock[0] + 100)
    with pytest.raises(ContextFault, match="stale_state"):
        engine.release_reservation(authority.sign(stale_payload))

    with pytest.raises(ContextFault, match="stale_state"):
        engine.bind_v2(*delayed_binding[:3])
    fresh_binding = _binding_command(engine, authority, revived, revived_claim, clock)
    with pytest.raises(ContextFault, match="binding_revision_required"):
        engine.bind(
            authority.sign(fresh_binding[3].model_dump(mode="json")),
            fresh_binding[1],
            fresh_binding[2],
        )
    bound_receipt = ContextService(engine).dispatch(
        {
            "operation": "bind_v2",
            "binding": fresh_binding[0],
            "authority": fresh_binding[1],
            "project_snapshot": fresh_binding[2],
        }
    )
    with engine.store.connection() as db:
        row = db.execute(
            "SELECT state,revision,binding FROM cycles WHERE id=?",
            (claim.context_cycle_id,),
        ).fetchone()
    assert row["state"] == "ACTIVE" and row["binding"] is not None
    post_dispatch = _release_payload(
        engine,
        claim,
        clock,
        expected_revision=row["revision"],
        issued_at=clock[0],
        expires_at=clock[0] + 100,
    )
    with pytest.raises(ContextFault, match="reservation_release_denied"):
        engine.release_reservation(authority.sign(post_dispatch))
    assert bound_receipt["payload"]["binding_digest"] == digest(
        fresh_binding[3]
    )


def test_signed_post_dispatch_abort_is_exact_replay_safe_and_terminal(hosted):  # noqa: F811
    engine, _, _, _, authority, _, _, clock = hosted
    _, original_reservation, reservation_receipt, claim = _reservation(
        engine, authority, clock
    )
    delayed_binding = _binding_command(
        engine, authority, reservation_receipt, claim, clock
    )
    command_payload = _abort_payload(
        engine, claim, reservation_receipt, clock
    )
    command = authority.sign(command_payload)
    service = ContextService(engine)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: service.dispatch(
                    {"operation": "abort_reservation", "command": command}
                ),
                range(4),
            )
        )
    assert all(item == results[0] for item in results)
    aborted = results[0]
    abort_claim = ContextReservationAbortReceiptV1.model_validate(
        verify(aborted, {engine.signer.key_id: engine.signer.key.public_key()})
    )
    assert abort_claim.abort_command_digest == digest(command_payload)
    assert abort_claim.objective_id == "retry-objective"
    assert abort_claim.objective_dispatch_receipt_digest == digest(
        "objective-dispatch-started"
    )
    assert abort_claim.revision == claim.revision + 1
    assert abort_claim.retention_expires_at == clock[0] + engine.profile.retention_seconds
    with engine.store.connection() as db:
        row = db.execute(
            "SELECT state,revision,expires,binding,reservation_abort_digest "
            "FROM cycles WHERE id=?",
            (claim.context_cycle_id,),
        ).fetchone()
        events = [json[0] for json in db.execute(
            "SELECT body FROM events WHERE cycle=?", (claim.context_cycle_id,)
        )]
    assert tuple(row) == (
        "ABORTED",
        abort_claim.revision,
        abort_claim.retention_expires_at,
        None,
        digest(command_payload),
    )
    assert sum("context.reservation_aborted" in event for event in events) == 1

    clock[0] += 101
    assert engine.abort_reservation(command) == aborted
    conflict = dict(command_payload)
    conflict["reason_code"] = "cancel"
    with pytest.raises(ContextFault, match="reservation_abort_conflict"):
        engine.abort_reservation(authority.sign(conflict))
    with pytest.raises(ContextFault, match="reservation_terminal"):
        engine.reserve_v2(original_reservation)
    revival = _reservation_payload_from_claim(engine, claim)
    revival.update(
        expires_at=clock[0] + 100,
        released_revision=abort_claim.revision,
    )
    with pytest.raises(ContextFault, match="reservation_terminal"):
        engine.reserve_v2(authority.sign(revival))
    with pytest.raises(ContextFault, match="state_conflict"):
        engine.bind_v2(*delayed_binding[:3])
    release_after_abort = _release_payload(
        engine,
        claim,
        clock,
        expected_revision=abort_claim.revision,
        issued_at=clock[0],
        expires_at=clock[0] + 100,
    )
    with pytest.raises(ContextFault, match="reservation_release_denied"):
        engine.release_reservation(authority.sign(release_after_abort))


def test_abort_reservation_and_bind_are_mutually_exclusive(hosted):  # noqa: F811
    engine, _, _, _, authority, _, _, clock = hosted
    _, _, reservation_receipt, claim = _reservation(engine, authority, clock)
    abort_command = authority.sign(
        _abort_payload(engine, claim, reservation_receipt, clock)
    )
    binding = _binding_command(engine, authority, reservation_receipt, claim, clock)

    def attempt(operation):
        try:
            if operation == "abort":
                return operation, engine.abort_reservation(abort_command)
            return operation, engine.bind_v2(*binding[:3])
        except ContextFault as exc:
            return operation, exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = dict(pool.map(attempt, ("abort", "bind")))
    successes = [value for value in outcomes.values() if isinstance(value, dict)]
    assert len(successes) == 1
    with engine.store.connection() as db:
        row = db.execute(
            "SELECT state,binding FROM cycles WHERE id=?", (claim.context_cycle_id,)
        ).fetchone()
    if row["state"] == "ABORTED":
        assert row["binding"] is None
        assert outcomes["bind"] == "context_state_conflict"
    else:
        assert row["state"] == "ACTIVE" and row["binding"] is not None
        assert outcomes["abort"] == "context_reservation_abort_denied"


@pytest.mark.parametrize(
    "mutation,error",
    [
        ({"dispatch_started": False}, "reservation_abort_schema"),
        ({"unexpected": "field"}, "reservation_abort_schema"),
        ({"reason_code": "unknown"}, "reservation_abort_schema"),
        ({"objective_id": "x" * 201}, "reservation_abort_schema"),
        ({"issued_at": 1000, "expires_at": 1901}, "reservation_abort_schema"),
        ({"issued_at": 1100, "expires_at": 1200}, "reservation_abort_expired"),
        ({"owner_id": "other-owner"}, "reservation_abort_denied"),
        ({"expected_revision": 2}, "stale_state"),
    ],
)
def test_abort_rejects_unclosed_unproven_mismatched_or_stale_commands(
    hosted, mutation, error  # noqa: F811
):
    engine, _, _, _, authority, _, _, clock = hosted
    _, _, reservation_receipt, claim = _reservation(engine, authority, clock)
    payload = _abort_payload(engine, claim, reservation_receipt, clock)
    payload.update(mutation)
    with pytest.raises(ContextFault, match=error):
        engine.abort_reservation(authority.sign(payload))


def test_abort_requires_gateway_signer_and_exact_v2_reservation_receipt(hosted):  # noqa: F811
    engine, _, _, _, authority, _, _, clock = hosted
    _, _, reservation_receipt, claim = _reservation(engine, authority, clock)
    payload = _abort_payload(engine, claim, reservation_receipt, clock)
    stranger = ReceiptSigner("abort-stranger", Ed25519PrivateKey.generate())
    with pytest.raises(ContextFault, match="signature_invalid"):
        engine.abort_reservation(stranger.sign(payload))

    forged_receipt = stranger.sign(reservation_receipt["payload"])
    with pytest.raises(ContextFault, match="reservation_abort_receipt_invalid"):
        engine.abort_reservation(
            authority.sign({**payload, "reservation_receipt": forged_receipt})
        )

    legacy_receipt = engine.reserve(
        authority.sign(
            {
                "operation": "reserve",
                "owner_id": "retry-owner",
                "project_id": "retry-project",
                "idempotency_key": "legacy-abort-ineligible",
                "request_digest": digest("legacy-abort-ineligible"),
                "profile_digest": digest(engine.profile),
                "expires_at": clock[0] + 100,
            }
        )
    )
    with pytest.raises(ContextFault, match="reservation_abort_receipt_invalid"):
        engine.abort_reservation(
            authority.sign({**payload, "reservation_receipt": legacy_receipt})
        )


def test_abort_refuses_released_or_bound_reservations(hosted):  # noqa: F811
    engine, _, _, _, authority, _, _, clock = hosted
    _, _, released_receipt, released_claim = _reservation(
        engine, authority, clock, key="abort-after-release"
    )
    release = authority.sign(
        _release_payload(
            engine,
            released_claim,
            clock,
            key="abort-after-release",
        )
    )
    released = engine.release_reservation(release)
    released_abort = _abort_payload(
        engine,
        released_claim,
        released_receipt,
        clock,
        key="abort-after-release",
        expected_revision=released["payload"]["revision"],
    )
    with pytest.raises(ContextFault, match="reservation_abort_denied"):
        engine.abort_reservation(authority.sign(released_abort))

    _, _, bound_receipt, bound_claim = _reservation(
        engine, authority, clock, key="abort-after-bind"
    )
    binding = _binding_command(engine, authority, bound_receipt, bound_claim, clock)
    engine.bind_v2(*binding[:3])
    with engine.store.connection() as db:
        revision = db.execute(
            "SELECT revision FROM cycles WHERE id=?", (bound_claim.context_cycle_id,)
        ).fetchone()[0]
    bound_abort = _abort_payload(
        engine,
        bound_claim,
        bound_receipt,
        clock,
        key="abort-after-bind",
        expected_revision=revision,
    )
    with pytest.raises(ContextFault, match="reservation_abort_denied"):
        engine.abort_reservation(authority.sign(bound_abort))


def test_no_dispatch_expiry_is_exact_revivable_and_excludes_release_abort(hosted):  # noqa: F811
    engine, _, _, _, authority, _, _, clock = hosted
    _, original, reservation_receipt, claim = _reservation(engine, authority, clock)
    expiry_payload = _expiry_payload(
        engine, authority, claim, reservation_receipt, clock
    )
    expiry_command = authority.sign(expiry_payload)
    service = ContextService(engine)

    expired = service.dispatch(
        {"operation": "expire_reservation", "command": expiry_command}
    )
    expiry_claim = ContextReservationExpiryReceiptV1.model_validate(
        verify(expired, {engine.signer.key_id: engine.signer.key.public_key()})
    )
    assert expiry_claim.reservation_receipt_digest == digest(reservation_receipt)
    assert expiry_claim.invocation_no_dispatch_receipt_digest == expiry_payload[
        "invocation_no_dispatch_receipt_digest"
    ]
    assert expiry_claim.expiry_command_digest == digest(expiry_payload)
    assert expiry_claim.revision == claim.revision + 1
    assert engine.expire_reservation(expiry_command) == expired

    release = _release_payload(
        engine,
        claim,
        clock,
        issued_at=clock[0],
        expires_at=clock[0] + 100,
    )
    with pytest.raises(ContextFault, match="reservation_release_denied"):
        engine.release_reservation(authority.sign(release))
    abort = _abort_payload(
        engine, claim, reservation_receipt, clock
    )
    with pytest.raises(ContextFault, match="reservation_abort_denied"):
        engine.abort_reservation(authority.sign(abort))
    with pytest.raises(ContextFault, match="reservation_terminal"):
        engine.reserve_v2(
            authority.sign(
                {
                    **original["payload"],
                    "expires_at": clock[0] + 100,
                }
            )
        )

    revival_payload = ContextReservationCommandV3(
        owner_id="retry-owner",
        project_id="retry-project",
        idempotency_key="retry-reservation",
        request_digest=digest("retry-reservation"),
        profile_digest=digest(engine.profile),
        expires_at=clock[0] + 100,
        expired_reservation_receipt=expired,
    ).model_dump(mode="json")
    revival = authority.sign(revival_payload)
    revived = service.dispatch({"operation": "reserve_v3", "reservation": revival})
    revived_claim = ContextReservationReceiptV3.model_validate(
        verify(revived, {engine.signer.key_id: engine.signer.key.public_key()})
    )
    assert revived_claim.revision == expiry_claim.revision + 1
    assert revived_claim.prior_expiry_receipt_digest == digest(expired)
    assert revived_claim.prior_expiry_revision == expiry_claim.revision
    assert revived_claim.invocation_no_dispatch_receipt_digest == (
        expiry_claim.invocation_no_dispatch_receipt_digest
    )
    assert revived_claim.revival_command_digest == digest(revival_payload)
    assert engine.reserve_v3(revival) == revived
    with pytest.raises(ContextFault, match="stale_state"):
        engine.expire_reservation(expiry_command)
    with pytest.raises(ContextFault, match="stale_state"):
        engine.reserve_v2(
            authority.sign(
                {
                    **original["payload"],
                    "expires_at": clock[0] + 100,
                }
            )
        )
    revived_binding = _binding_command(
        engine, authority, revived, revived_claim, clock
    )
    with pytest.raises(ContextFault, match="binding_invalid"):
        engine.bind_v2(*revived_binding[:3])
    bound = service.dispatch(
        {
            "operation": "bind_v3",
            "binding": revived_binding[0],
            "authority": revived_binding[1],
            "project_snapshot": revived_binding[2],
        }
    )
    assert bound["payload"]["binding_digest"] == digest(revived_binding[3])


def test_no_dispatch_expiry_accepts_natural_sweep_and_loses_bind_race(hosted):  # noqa: F811
    engine, _, _, _, authority, _, _, clock = hosted
    _, _, reservation_receipt, claim = _reservation(
        engine, authority, clock, key="naturally-expired", ttl=10
    )
    expiry_payload = _expiry_payload(
        engine,
        authority,
        claim,
        reservation_receipt,
        clock,
        key="naturally-expired",
    )
    clock[0] += 11
    _reservation(engine, authority, clock, key="natural-expiry-trigger")
    expiry_payload.update(issued_at=clock[0], expires_at=clock[0] + 100)
    expired = engine.expire_reservation(authority.sign(expiry_payload))
    assert expired["payload"]["revision"] == claim.revision + 2

    _, _, race_receipt, race_claim = _reservation(
        engine, authority, clock, key="expiry-bind-race"
    )
    expiry_command = authority.sign(
        _expiry_payload(
            engine,
            authority,
            race_claim,
            race_receipt,
            clock,
            key="expiry-bind-race",
        )
    )
    binding = _binding_command(engine, authority, race_receipt, race_claim, clock)

    def attempt(operation):
        try:
            if operation == "expire":
                return operation, engine.expire_reservation(expiry_command)
            return operation, engine.bind_v2(*binding[:3])
        except ContextFault as exc:
            return operation, exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = dict(pool.map(attempt, ("expire", "bind")))
    assert sum(isinstance(value, dict) for value in outcomes.values()) == 1
    with engine.store.connection() as db:
        row = db.execute(
            "SELECT state,binding FROM cycles WHERE id=?",
            (race_claim.context_cycle_id,),
        ).fetchone()
    if row["state"] == "EXPIRED":
        assert row["binding"] is None
        assert outcomes["bind"] == "context_state_conflict"
    else:
        assert row["state"] == "ACTIVE"
        assert outcomes["expire"] == "context_reservation_expiry_denied"


def test_v3_reservation_can_expire_and_revive_across_repeated_generations(hosted):  # noqa: F811
    engine, _, _, _, authority, _, _, clock = hosted
    key = "repeat-expiry"
    _, _, first_receipt, first_claim = _reservation(
        engine, authority, clock, key=key
    )
    first_expiry_command = authority.sign(
        _expiry_payload(
            engine, authority, first_claim, first_receipt, clock, key=key
        )
    )
    first_expiry = engine.expire_reservation(first_expiry_command)
    first_expiry_claim = ContextReservationExpiryReceiptV1.model_validate(
        verify(first_expiry, {engine.signer.key_id: engine.signer.key.public_key()})
    )

    first_revival_payload = ContextReservationCommandV3(
        owner_id="retry-owner",
        project_id="retry-project",
        idempotency_key=key,
        request_digest=digest(key),
        profile_digest=digest(engine.profile),
        expires_at=clock[0] + 100,
        expired_reservation_receipt=first_expiry,
    ).model_dump(mode="json")
    first_revival_command = authority.sign(first_revival_payload)
    first_revival = engine.reserve_v3(first_revival_command)
    first_revival_claim = ContextReservationReceiptV3.model_validate(
        verify(first_revival, {engine.signer.key_id: engine.signer.key.public_key()})
    )
    stale_binding = _binding_command(
        engine, authority, first_revival, first_revival_claim, clock
    )
    stale_abort = authority.sign(
        _abort_payload(
            engine,
            first_revival_claim,
            first_revival,
            clock,
            key=key,
        )
    )
    second_expiry_command = authority.sign(
        _expiry_payload(
            engine,
            authority,
            first_revival_claim,
            first_revival,
            clock,
            key=key,
        )
    )
    second_expiry = engine.expire_reservation(second_expiry_command)
    second_expiry_claim = ContextReservationExpiryReceiptV1.model_validate(
        verify(second_expiry, {engine.signer.key_id: engine.signer.key.public_key()})
    )
    assert second_expiry_claim.revision == first_revival_claim.revision + 1
    assert second_expiry_claim.reservation_receipt_digest == digest(first_revival)

    second_revival_payload = ContextReservationCommandV3(
        owner_id="retry-owner",
        project_id="retry-project",
        idempotency_key=key,
        request_digest=digest(key),
        profile_digest=digest(engine.profile),
        expires_at=clock[0] + 100,
        expired_reservation_receipt=second_expiry,
    ).model_dump(mode="json")
    second_revival = engine.reserve_v3(authority.sign(second_revival_payload))
    second_revival_claim = ContextReservationReceiptV3.model_validate(
        verify(second_revival, {engine.signer.key_id: engine.signer.key.public_key()})
    )
    assert second_revival_claim.revision == second_expiry_claim.revision + 1
    assert second_revival_claim.prior_expiry_receipt_digest == digest(second_expiry)
    assert second_revival_claim.revival_command_digest == digest(second_revival_payload)

    with pytest.raises(ContextFault, match="stale_state"):
        engine.bind_v3(*stale_binding[:3])
    with pytest.raises(ContextFault, match="stale_state"):
        engine.abort_reservation(stale_abort)
    with pytest.raises(ContextFault, match="stale_state"):
        engine.expire_reservation(second_expiry_command)
    with pytest.raises(ContextFault, match="stale_state"):
        engine.expire_reservation(first_expiry_command)
    assert first_expiry_claim.revision < second_revival_claim.revision


def test_v3_reservation_abort_cleanup_and_replay_survive_retention(hosted):  # noqa: F811
    engine, _, _, _, authority, _, _, clock = hosted
    key = "revived-abort"
    _, _, first_receipt, first_claim = _reservation(
        engine, authority, clock, key=key
    )
    expired = engine.expire_reservation(
        authority.sign(
            _expiry_payload(
                engine, authority, first_claim, first_receipt, clock, key=key
            )
        )
    )
    revival_payload = ContextReservationCommandV3(
        owner_id="retry-owner",
        project_id="retry-project",
        idempotency_key=key,
        request_digest=digest(key),
        profile_digest=digest(engine.profile),
        expires_at=clock[0] + 100,
        expired_reservation_receipt=expired,
    ).model_dump(mode="json")
    revival_receipt = engine.reserve_v3(authority.sign(revival_payload))
    revival_claim = ContextReservationReceiptV3.model_validate(
        verify(
            revival_receipt,
            {engine.signer.key_id: engine.signer.key.public_key()},
        )
    )
    abort_payload = _abort_payload(
        engine, revival_claim, revival_receipt, clock, key=key
    )
    abort_command = authority.sign(abort_payload)
    aborted = engine.abort_reservation(abort_command)
    abort_claim = ContextReservationAbortReceiptV1.model_validate(
        verify(aborted, {engine.signer.key_id: engine.signer.key.public_key()})
    )
    clock[0] = abort_claim.retention_expires_at
    cleanup_payload = ContextReservationAbortExpiryCommandV1(
        namespace=NamespaceV1(
            owner_id="retry-owner",
            project_id="retry-project",
            objective_id="retry-objective",
            context_cycle_id=revival_claim.context_cycle_id,
        ),
        idempotency_key="cleanup-" + key,
        expected_revision=abort_claim.revision,
        reason_code="policy_expiry",
        abort_receipt=aborted,
        issued_at=clock[0],
        expires_at=clock[0] + 100,
    ).model_dump(mode="json")
    cleanup_command = authority.sign(cleanup_payload)
    deleted = engine.expire_aborted_reservation(cleanup_command)
    assert deleted["payload"]["checkpoint_durability_mode"] == "unregistered"
    clock[0] += 101
    assert engine.expire_aborted_reservation(cleanup_command) == deleted
    assert engine.abort_reservation(abort_command) == aborted


def test_no_dispatch_expiry_rejects_ambiguous_forged_legacy_and_conflicting_proof(hosted):  # noqa: F811
    engine, _, _, _, authority, _, _, clock = hosted
    _, _, reservation_receipt, claim = _reservation(engine, authority, clock)
    payload = _expiry_payload(engine, authority, claim, reservation_receipt, clock)
    stranger = ReceiptSigner("expiry-stranger", Ed25519PrivateKey.generate())

    with pytest.raises(ContextFault, match="signature_invalid"):
        engine.expire_reservation(stranger.sign(payload))
    ambiguous = _no_dispatch_receipt(
        engine,
        authority,
        claim,
        reservation_receipt,
        clock,
        ledger_snapshot_updates={"state": "dispatch_started"},
    )
    ambiguous_payload = {
        **payload,
        "invocation_no_dispatch_receipt": ambiguous,
        "invocation_no_dispatch_receipt_digest": digest(ambiguous),
    }
    with pytest.raises(ContextFault, match="invocation_no_dispatch_receipt_invalid"):
        engine.expire_reservation(authority.sign(ambiguous_payload))
    forged = stranger.sign(payload["invocation_no_dispatch_receipt"]["payload"])
    forged_payload = {
        **payload,
        "invocation_no_dispatch_receipt": forged,
        "invocation_no_dispatch_receipt_digest": digest(forged),
    }
    with pytest.raises(ContextFault, match="invocation_no_dispatch_receipt_invalid"):
        engine.expire_reservation(authority.sign(forged_payload))

    legacy = engine.reserve(
        authority.sign(
            {
                "operation": "reserve",
                "owner_id": "retry-owner",
                "project_id": "retry-project",
                "idempotency_key": "expiry-legacy",
                "request_digest": digest("expiry-legacy"),
                "profile_digest": digest(engine.profile),
                "expires_at": clock[0] + 100,
            }
        )
    )
    with pytest.raises(ContextFault, match="reservation_expiry_receipt_invalid"):
        engine.expire_reservation(
            authority.sign({**payload, "reservation_receipt": legacy})
        )

    command = authority.sign(payload)
    engine.expire_reservation(command)
    changed_no_dispatch = _no_dispatch_receipt(
        engine,
        authority,
        claim,
        reservation_receipt,
        clock,
        ledger_snapshot_updates={"revision": 8},
    )
    changed = {
        **payload,
        "invocation_no_dispatch_receipt": changed_no_dispatch,
        "invocation_no_dispatch_receipt_digest": digest(changed_no_dispatch),
    }
    with pytest.raises(ContextFault, match="reservation_expiry_conflict"):
        engine.expire_reservation(authority.sign(changed))


def _reservation_payload_from_claim(engine, claim):
    return ContextReservationCommandV2(
        owner_id="retry-owner",
        project_id="retry-project",
        idempotency_key="retry-reservation",
        request_digest=claim.request_digest,
        profile_digest=digest(engine.profile),
        expires_at=1,
        released_revision=None,
    ).model_dump(mode="json")


def test_natural_expiry_accepts_only_fresh_signed_predispatch_proof(hosted):  # noqa: F811
    engine, _, _, _, authority, _, _, clock = hosted
    _, _, _, claim = _reservation(
        engine, authority, clock, key="retry-reservation", ttl=10
    )
    clock[0] += 11
    _reservation(engine, authority, clock, key="expiry-sweep-trigger")
    with engine.store.connection() as db:
        expired = db.execute(
            "SELECT state,revision FROM cycles WHERE id=?", (claim.context_cycle_id,)
        ).fetchone()
    assert tuple(expired) == ("EXPIRED", claim.revision + 1)

    payload = _release_payload(
        engine,
        claim,
        clock,
        issued_at=clock[0],
        expires_at=clock[0] + 100,
    )
    released = engine.release_reservation(authority.sign(payload))
    assert released["payload"]["revision"] == expired["revision"] + 1
    fresh = _reservation_payload_from_claim(engine, claim)
    fresh["expires_at"] = clock[0] + 100
    fresh["released_revision"] = released["payload"]["revision"]
    revived = engine.reserve_v2(authority.sign(fresh))
    assert revived["payload"]["context_cycle_id"] == claim.context_cycle_id


@pytest.mark.parametrize(
    "mutation,error",
    [
        ({"dispatch_started": True}, "reservation_release_schema"),
        ({"unexpected": "field"}, "reservation_release_schema"),
        ({"owner_id": "other-owner"}, "reservation_release_denied"),
        ({"issued_at": 1100, "expires_at": 1200}, "reservation_release_expired"),
    ],
)
def test_release_rejects_unclosed_unproven_or_mismatched_commands(
    hosted, mutation, error  # noqa: F811
):
    engine, _, _, _, authority, _, _, clock = hosted
    _, _, _, claim = _reservation(engine, authority, clock)
    payload = _release_payload(engine, claim, clock)
    payload.update(mutation)
    with pytest.raises(ContextFault, match=error):
        engine.release_reservation(authority.sign(payload))

    stranger = ReceiptSigner("stranger", Ed25519PrivateKey.generate())
    with pytest.raises(ContextFault, match="signature_invalid"):
        engine.release_reservation(stranger.sign(_release_payload(engine, claim, clock)))


def test_v1_reservation_receipt_wire_remains_exact(hosted):  # noqa: F811
    engine, _, _, _, authority, _, _, clock = hosted
    receipt = engine.reserve(
        authority.sign(
            {
                "operation": "reserve",
                "owner_id": "legacy-owner",
                "project_id": "legacy-project",
                "idempotency_key": "legacy-reservation",
                "request_digest": digest("legacy-reservation"),
                "profile_digest": digest(engine.profile),
                "expires_at": clock[0] + 100,
            }
        )
    )
    payload = verify(
        receipt, {engine.signer.key_id: engine.signer.key.public_key()}
    )
    assert ContextReservationReceiptV1.model_validate(payload).model_dump(
        mode="json"
    ) == payload
    assert set(payload) == {
        "schema_version",
        "context_cycle_id",
        "context_bucket_id",
        "request_digest",
        "profile_digest",
    }


def test_v1_client_can_bind_checkpoint_seal_and_restart_during_drain(hosted):  # noqa: F811
    engine, _, _, _, authority, proof, make, clock = hosted
    key = "legacy-drain-flow"
    receipt = engine.reserve(
        authority.sign(
            {
                "operation": "reserve",
                "owner_id": "retry-owner",
                "project_id": "retry-project",
                "idempotency_key": key,
                "request_digest": digest(key),
                "profile_digest": digest(engine.profile),
                "expires_at": clock[0] + 100,
            }
        )
    )
    claim = ContextReservationReceiptV1.model_validate(
        verify(receipt, {engine.signer.key_id: engine.signer.key.public_key()})
    )
    namespace = NamespaceV1(
        owner_id="retry-owner",
        project_id="retry-project",
        objective_id="retry-objective",
        context_cycle_id=claim.context_cycle_id,
    )
    authority_payload = {"objective": "legacy drain", "paths": ["src"]}
    project_snapshot = {
        "project_id": "retry-project",
        "graph_id": "legacy-graph",
        "policy_digest": digest("retry-policy"),
        "nodes": [],
    }
    binding = ContextBindingV1(
        namespace=namespace,
        context_bucket_id=claim.context_bucket_id,
        repository_id="retry-repository",
        repo_main_sha="a" * 40,
        project_graph_id="legacy-graph",
        graph_revision=1,
        graph_checksum=digest(project_snapshot),
        plan_digest=digest("legacy-plan"),
        execution_profile_digest=digest("legacy-execution-profile"),
        shared_ir_digest=digest("legacy-ir"),
        authorization_receipt_ref="legacy-authorization",
        authorization_digest=digest(authority_payload),
        policy_digest=digest("retry-policy"),
        redaction_digest=digest("retry-redaction"),
        profile_digest=digest(engine.profile),
        captain_binding_id="legacy-captain",
        created_at=clock[0],
        expires_at=clock[0] + 1000,
    )
    engine.bind(
        authority.sign(binding.model_dump(mode="json")),
        authority_payload,
        project_snapshot,
    )

    def capability(role):
        return authority.sign(
            CapabilityV1(
                namespace=namespace,
                principal_id="legacy-worker",
                lane_id="legacy-lane",
                task_id="legacy-task",
                role=role,
                operations=["checkpoint", "seal", "status"],
                source_class="tool_observation",
                binding_digest=digest(binding),
                policy_digest=binding.policy_digest,
                profile_digest=digest(engine.profile),
                fence=1,
                expires_at=clock[0] + 1000,
                max_bytes=32768,
                max_tokens=12000,
            ).model_dump(mode="json")
        )

    checkpoint = engine.checkpoint(capability("durability"), "legacy-checkpoint")
    with engine.store.connection() as db:
        durability = db.execute(
            "SELECT checkpoint_durability_mode,checkpoint_durability_legacy,"
            "reservation_protocol_version FROM cycles WHERE id=?",
            (claim.context_cycle_id,),
        ).fetchone()
    assert tuple(durability) == ("remote_registered", 1, 1)
    proof_payload = ClosureProofV1(
        namespace=namespace,
        binding_digest=digest(binding),
        final_root="0" * 64,
        final_cursor=0,
        execution_dag_digest=digest("legacy-dag"),
        plan_ir_digest=binding.shared_ir_digest,
        shared_ir_digest=digest("legacy-final-ir"),
        repo_main_sha=binding.repo_main_sha,
        git_head_sha="b" * 40,
        pr_head_sha="b" * 40,
        pr_url="https://github.com/org/repo/pull/1",
        ci_receipt=digest("legacy-ci"),
        memory_candidate_digest=digest("legacy-memory"),
        nano_receipt=digest("legacy-nano"),
        proof_receipt=digest("legacy-proof"),
        promotion_set_root=digest([]),
        accepted_record_ids=[],
        rejected_record_ids=[],
        expires_at=clock[0] + 100,
    )
    sealed = engine.seal(
        capability("verifier"),
        proof.sign(proof_payload.model_dump(mode="json")),
        checkpoint["receipt"],
    )
    reopened = make(engine.store.path)
    assert reopened.status(capability("verifier"))["payload"]["state"] == "SEALED"
    assert reopened.seal(
        capability("verifier"),
        proof.sign(proof_payload.model_dump(mode="json")),
        checkpoint["receipt"],
    ) == sealed

    release = ContextReservationReleaseCommandV1(
        context_cycle_id=claim.context_cycle_id,
        owner_id="retry-owner",
        project_id="retry-project",
        idempotency_key=key,
        request_digest=digest(key),
        profile_digest=digest(engine.profile),
        expected_revision=1,
        reason_code="pre_dispatch_authorization_failed",
        dispatch_started=False,
        authorization_failure_receipt_digest=digest("legacy-denial"),
        issued_at=clock[0],
        expires_at=clock[0] + 100,
    )
    with pytest.raises(ContextFault, match="reservation_release_denied"):
        engine.release_reservation(authority.sign(release.model_dump(mode="json")))
