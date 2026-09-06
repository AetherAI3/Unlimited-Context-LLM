"""Crash-safe handoff of startup hydrate evidence to the host coordinator."""

from concurrent.futures import ThreadPoolExecutor
import asyncio
import base64
import hashlib
import json
import os
import stat

import pytest

from aether_context.contracts import (
    AppendRequestV1,
    ContextHydrateReceiptV1,
    LIVE_IPC_MAX_CHECKPOINT_BYTES,
    canonical,
)
from aether_context.crypto import ContextFault, verify
from aether_context.service import (
    SMALL_REQUEST_LIMIT,
    MAX_SIGNED_HYDRATE_ENVELOPE,
    MAX_LOCAL_IO_TIMEOUT_SECONDS,
    MIN_LOCAL_IO_BYTES_PER_SECOND,
    ContextService,
    _restore_engine,
    _restore_requested,
    _write_restore_result_once,
)
from test_hosted_context import hosted  # noqa: F401


def _restore_material(hosted, tmp_path):  # noqa: F811
    engine, cap, _, binding, authority, _, make, _ = hosted
    engine.append(
        cap(),
        AppendRequestV1(
            idempotency_key="restore-result-record",
            text="needle durable recovery result",
            action_ref="restore-result-test",
        ),
    )
    checkpoint = engine.checkpoint(cap(role="durability"), "restore-result-checkpoint")
    claim = checkpoint["receipt"]["payload"]
    grant = authority.sign(
        {
            "operation": "hydrate",
            "namespace": binding.namespace.model_dump(),
            "manifest_checksum": claim["manifest_checksum"],
            "fence": 2,
            "expires_at": 1900,
        }
    )
    pack_path = tmp_path / "checkpoint.enc"
    grant_path = tmp_path / "grant.json"
    receipt_path = tmp_path / "checkpoint-receipt.json"
    pack_path.write_bytes(base64.b64decode(checkpoint["pack"], validate=True))
    grant_path.write_bytes(canonical(grant))
    receipt_path.write_bytes(canonical(checkpoint["receipt"]))
    return {
        "original": engine,
        "replacement": make(tmp_path / "replacement.sqlite3"),
        "profile": engine.profile,
        "pack_path": pack_path,
        "grant_path": grant_path,
        "receipt_path": receipt_path,
        "result_path": tmp_path / "hydrate-result.json",
        "receipt_keys": {engine.signer.key_id: engine.signer.key.public_key()},
    }


def _restore(item):
    return _restore_engine(
        item["replacement"],
        profile=item["profile"],
        pack_path=str(item["pack_path"]),
        grant_path=str(item["grant_path"]),
        receipt_path=str(item["receipt_path"]),
        result_path=str(item["result_path"]),
        receipt_keys=item["receipt_keys"],
    )


def test_restore_requires_all_input_paths_and_result_path():
    assert _restore_requested(None, None, None, None) is False
    assert _restore_requested("pack", "grant", "receipt", "result") is True
    for missing in range(4):
        paths = ["pack", "grant", "receipt", "result"]
        paths[missing] = None
        with pytest.raises(SystemExit, match="context_restore_config_incomplete"):
            _restore_requested(*paths)


def test_signed_restore_result_is_atomic_and_lost_ack_replay_is_exact(hosted, tmp_path):  # noqa: F811
    item = _restore_material(hosted, tmp_path)

    first = _restore(item)
    first_info = item["result_path"].stat()
    first_bytes = item["result_path"].read_bytes()
    assert first_bytes == canonical(first) + b"\n"
    if os.name != "nt":
        assert stat.S_IMODE(first_info.st_mode) == 0o600
    payload = verify(first, item["receipt_keys"])
    assert ContextHydrateReceiptV1.model_validate(payload).model_dump(mode="json") == payload
    assert payload["manifest_checksum"] == json.loads(
        item["receipt_path"].read_text()
    )["payload"]["manifest_checksum"]
    assert payload["checkpoint_number"] == 1
    assert payload["fence"] == 2

    # Simulate Cloud losing the first acknowledgement and issuing the exact
    # recovery start again. The engine and write-once file both replay.
    second = _restore(item)
    assert second == first
    assert item["result_path"].read_bytes() == first_bytes
    assert item["result_path"].stat().st_ino == first_info.st_ino
    with item["replacement"].store.connection() as db:
        events = [json.loads(row[0]) for row in db.execute("SELECT body FROM events")]
    assert sum(event["type"] == "context.recovered" for event in events) == 1


def test_hydrate_ipc_is_closed_configured_and_lost_ack_safe(hosted, tmp_path):  # noqa: F811
    item = _restore_material(hosted, tmp_path)
    request = {
        "operation": "hydrate",
        "grant": json.loads(item["grant_path"].read_text()),
        "checkpoint_receipt": json.loads(item["receipt_path"].read_text()),
        "pack": base64.b64encode(item["pack_path"].read_bytes()).decode("ascii"),
    }
    service = ContextService(
        item["replacement"], checkpoint_receipt_keys=item["receipt_keys"]
    )
    before = canonical(request)
    first = service.dispatch(request)
    assert service.dispatch(request) == first
    assert canonical(request) == before
    owned_request = json.loads(before)
    assert service._dispatch(owned_request, consume_hydrate_pack=True) == first
    assert "pack" not in owned_request
    assert ContextHydrateReceiptV1.model_validate(
        verify(first, item["receipt_keys"])
    )
    with pytest.raises(ContextFault, match="context_request_schema"):
        service.dispatch({**request, "receipt_keys": item["receipt_keys"]})
    with pytest.raises(ContextFault, match="context_request_schema"):
        service.dispatch({**request, "pack": "not canonical base64"})
    with pytest.raises(ContextFault, match="context_hydrate_unavailable"):
        ContextService(item["replacement"]).dispatch(request)


def test_hydrate_ipc_accepts_pack_near_live_service_cap(hosted, tmp_path, monkeypatch):  # noqa: F811
    item = _restore_material(hosted, tmp_path)
    observed = {}

    def hydrate(grant, receipt, pack, receipt_keys, *, snapshot_max_bytes):
        observed.update(
            grant=grant,
            receipt=receipt,
            pack=pack,
            receipt_keys=receipt_keys,
            snapshot_max_bytes=snapshot_max_bytes,
        )
        return {"accepted": True}

    monkeypatch.setattr(item["replacement"], "hydrate", hydrate)
    service = ContextService(
        item["replacement"], checkpoint_receipt_keys=item["receipt_keys"]
    )
    pack = b"x" * (service.max_live_hydrate_pack_bytes - 1)
    request = {
        "operation": "hydrate",
        "grant": {"signed": "grant"},
        "checkpoint_receipt": {"signed": "checkpoint"},
        "pack": base64.b64encode(pack).decode("ascii"),
    }
    encoded_size = len(canonical(request))
    assert encoded_size <= service.max_hydrate_request
    assert service.dispatch(request) == {"accepted": True}
    assert observed == {
        "grant": request["grant"],
        "receipt": request["checkpoint_receipt"],
        "pack": pack,
        "receipt_keys": item["receipt_keys"],
        "snapshot_max_bytes": service.max_live_checkpoint_snapshot_bytes,
    }


def test_hydrate_ipc_rejects_pack_above_configured_checkpoint_bound(
    hosted, tmp_path, monkeypatch  # noqa: F811
):
    item = _restore_material(hosted, tmp_path)
    item["replacement"].profile = item["profile"].model_copy(
        update={"max_checkpoint_bytes": 1024}
    )
    service = ContextService(
        item["replacement"], checkpoint_receipt_keys=item["receipt_keys"]
    )
    monkeypatch.setattr(
        item["replacement"],
        "hydrate",
        lambda *_args, **_kwargs: pytest.fail("oversized pack reached the engine"),
    )
    request = {
        "operation": "hydrate",
        "grant": {"signed": "grant"},
        "checkpoint_receipt": {"signed": "checkpoint"},
        "pack": base64.b64encode(b"x" * 1025).decode("ascii"),
    }
    with pytest.raises(ContextFault, match="context_request_limit"):
        service.dispatch(request)


def test_live_hydrate_bounds_decompressed_snapshot_before_db_effect(
    hosted, tmp_path  # noqa: F811
):
    engine, cap, _, binding, authority, _, make, _ = hosted
    checkpoint = engine.checkpoint(
        cap(role="durability"), "hydrate-snapshot-ceiling-source"
    )
    checkpoint_pack = base64.b64decode(checkpoint["pack"], validate=True)
    namespace = binding.namespace.model_dump(mode="json")
    snapshot = engine.cipher.decrypt(
        checkpoint_pack,
        {"namespace": namespace, "domain": "checkpoint/v1"},
        engine.profile.max_checkpoint_bytes,
    )
    snapshot["oversized_transport_padding"] = "x" * 16_100_000
    oversized_pack = engine.cipher.encrypt(
        snapshot, {"namespace": namespace, "domain": "checkpoint/v1"}
    )
    replacement = make(tmp_path / "hydrate-snapshot-ceiling.sqlite3")
    service = ContextService(
        replacement,
        checkpoint_receipt_keys={
            engine.signer.key_id: engine.signer.key.public_key()
        },
    )
    assert len(oversized_pack) < service.max_live_hydrate_pack_bytes
    oversized_checksum = hashlib.sha256(oversized_pack).hexdigest()
    receipt_payload = {
        **checkpoint["receipt"]["payload"],
        "manifest_checksum": oversized_checksum,
        "bytes": len(oversized_pack),
    }
    oversized_receipt = engine.signer.sign(receipt_payload)
    oversized_grant = authority.sign(
        {
            "operation": "hydrate",
            "namespace": namespace,
            "manifest_checksum": oversized_checksum,
            "fence": 2,
            "expires_at": 1900,
        }
    )
    with pytest.raises(ContextFault, match="context_pack_limit"):
        service.dispatch(
            {
                "operation": "hydrate",
                "grant": oversized_grant,
                "checkpoint_receipt": oversized_receipt,
                "pack": base64.b64encode(oversized_pack).decode("ascii"),
            }
        )
    with replacement.store.connection() as db:
        assert db.execute("SELECT count(*) FROM cycles").fetchone()[0] == 0

    startup_replacement = make(
        tmp_path / "hydrate-startup-snapshot-ceiling.sqlite3"
    )
    pack_path = tmp_path / "oversized-compressed-checkpoint.enc"
    grant_path = tmp_path / "oversized-compressed-grant.json"
    receipt_path = tmp_path / "oversized-compressed-receipt.json"
    pack_path.write_bytes(oversized_pack)
    grant_path.write_bytes(canonical(oversized_grant))
    receipt_path.write_bytes(canonical(oversized_receipt))
    with pytest.raises(ContextFault, match="context_pack_limit"):
        _restore_engine(
            startup_replacement,
            profile=engine.profile,
            pack_path=str(pack_path),
            grant_path=str(grant_path),
            receipt_path=str(receipt_path),
            result_path=str(tmp_path / "oversized-compressed-result.json"),
            receipt_keys={
                engine.signer.key_id: engine.signer.key.public_key()
            },
        )
    with startup_replacement.store.connection() as db:
        assert db.execute("SELECT count(*) FROM cycles").fetchone()[0] == 0

    source_checksum = checkpoint["receipt"]["payload"]["manifest_checksum"]
    accepted = service.dispatch(
        {
            "operation": "hydrate",
            "grant": authority.sign(
                {
                    "operation": "hydrate",
                    "namespace": namespace,
                    "manifest_checksum": source_checksum,
                    "fence": 2,
                    "expires_at": 1900,
                }
            ),
            "checkpoint_receipt": checkpoint["receipt"],
            "pack": checkpoint["pack"],
        }
    )
    assert accepted["payload"]["schema_version"] == "ContextHydrateReceiptV1"


def test_startup_restore_bounds_actual_pack_and_closed_header_files(
    hosted, tmp_path  # noqa: F811
):
    item = _restore_material(hosted, tmp_path)
    original_pack = item["pack_path"].read_bytes()
    original_grant = item["grant_path"].read_bytes()

    item["pack_path"].write_bytes(b"x" * 1025)
    narrow_profile = item["profile"].model_copy(
        update={"max_checkpoint_bytes": 1024}
    )
    with pytest.raises(ContextFault, match="context_pack_limit"):
        _restore_engine(
            item["replacement"],
            profile=narrow_profile,
            pack_path=str(item["pack_path"]),
            grant_path=str(item["grant_path"]),
            receipt_path=str(item["receipt_path"]),
            result_path=str(item["result_path"]),
            receipt_keys=item["receipt_keys"],
        )

    item["pack_path"].write_bytes(original_pack)
    item["grant_path"].write_bytes(b"x" * (MAX_SIGNED_HYDRATE_ENVELOPE + 1))
    with pytest.raises(ContextFault, match="context_restore_input_invalid"):
        _restore(item)

    item["grant_path"].write_bytes(b'{"payload":{},"payload":{}}')
    with pytest.raises(ContextFault, match="context_restore_input_invalid"):
        _restore(item)

    item["grant_path"].write_bytes(original_grant)
    with item["replacement"].store.connection() as db:
        assert db.execute("SELECT count(*) FROM cycles").fetchone()[0] == 0


class _Reader:
    def __init__(self, value):
        self.value = value

    async def read(self, _size):
        value, self.value = self.value, b""
        return value


class _Transport:
    def pause_reading(self):
        pass

    def resume_reading(self):
        pass


class _Writer:
    def __init__(self):
        self.transport = _Transport()
        self.output = bytearray()

    def write(self, value):
        self.output.extend(value)

    async def drain(self):
        pass

    def close(self):
        pass

    async def wait_closed(self):
        pass


class _BlockingWriter(_Writer):
    def __init__(self, draining, release):
        super().__init__()
        self.draining = draining
        self.release = release

    async def drain(self):
        self.draining.set()
        await self.release.wait()


def _handle(service, frame):
    writer = _Writer()
    asyncio.run(service.handle(_Reader(frame), writer))
    return json.loads(writer.output)


def test_live_ipc_frame_exact_bound_overflow_eof_crlf_and_duplicate_newline(hosted):  # noqa: F811
    service = ContextService(hosted[0])
    prefix = canonical({"operation": "health", "padding": ""})
    exact = canonical(
        {
            "operation": "health",
            "padding": "x" * (SMALL_REQUEST_LIMIT - len(prefix)),
        }
    )
    assert len(exact) == SMALL_REQUEST_LIMIT
    assert _handle(service, exact + b"\n")["error"] == "context_request_schema"
    assert _handle(service, exact[:-2] + b"xxx\n")["error"] == "context_request_limit"
    assert _handle(service, b'{"operation":"health"}')["error"] == (
        "context_request_framing"
    )
    assert _handle(service, b'{"operation":"health"}\r\n')["error"] == (
        "context_request_framing"
    )
    assert _handle(service, b'{"operation":"health"}\n\n')["error"] == (
        "context_request_framing"
    )


def test_live_ipc_uses_only_top_level_operation_for_admission(
    hosted, monkeypatch  # noqa: F811
):
    engine = hosted[0]
    service = ContextService(engine, checkpoint_receipt_keys={"unused": object()})
    monkeypatch.setattr(
        engine, "hydrate", lambda *_, **__: {"classified": "hydrate"}
    )
    request = {
        "checkpoint_receipt": {"payload": {"operation": "health"}},
        "grant": {"payload": {"operation": "bind"}},
        "operation": "hydrate",
        "pack": base64.b64encode(b"pack").decode(),
    }
    assert _handle(service, canonical(request) + b"\n") == {
        "ok": True,
        "result": {"classified": "hydrate"},
    }


def test_live_ipc_large_response_does_not_block_profile_small_call_pool(
    hosted, monkeypatch  # noqa: F811
):
    engine = hosted[0]
    service = ContextService(engine, checkpoint_receipt_keys={"unused": object()})
    monkeypatch.setattr(engine, "hydrate", lambda *_, **__: {"accepted": True})
    monkeypatch.setattr(engine, "health", lambda: {"ready": True})
    hydrate = canonical(
        {
            "checkpoint_receipt": {"signed": "checkpoint"},
            "grant": {"signed": "grant"},
            "operation": "hydrate",
            "pack": base64.b64encode(b"pack").decode(),
        }
    ) + b"\n"

    async def scenario():
        draining = asyncio.Event()
        release = asyncio.Event()
        first_writer = _BlockingWriter(draining, release)
        first = asyncio.create_task(
            service.handle(_Reader(hydrate), first_writer)
        )
        await asyncio.wait_for(draining.wait(), timeout=2)

        small_writers = [_Writer() for _ in range(service.max_small_operations)]
        await asyncio.gather(
            *(
                service.handle(
                    _Reader(b'{"operation":"health"}\n'), writer
                )
                for writer in small_writers
            )
        )
        assert all(json.loads(writer.output)["ok"] for writer in small_writers)

        second_writer = _Writer()
        second = asyncio.create_task(
            service.handle(_Reader(hydrate), second_writer)
        )
        for _ in range(10):
            if service._heavy_admitted == 2:
                break
            await asyncio.sleep(0)
        assert service._heavy_admitted == 2
        third_writer = _Writer()
        await service.handle(_Reader(hydrate), third_writer)
        assert json.loads(third_writer.output) == {
            "ok": False,
            "error": "context_service_busy",
        }
        release.set()
        await asyncio.gather(first, second)
        assert json.loads(first_writer.output)["ok"] is True
        assert json.loads(second_writer.output)["ok"] is True
        assert service._heavy_admitted == 0

    asyncio.run(scenario())


def test_ipc_timeout_covers_advertised_profile_size(hosted, tmp_path):  # noqa: F811
    item = _restore_material(hosted, tmp_path)
    service = ContextService(
        item["replacement"], checkpoint_receipt_keys=item["receipt_keys"]
    )
    assert service.io_timeout_seconds <= MAX_LOCAL_IO_TIMEOUT_SECONDS
    assert (
        service.io_timeout_seconds * MIN_LOCAL_IO_BYTES_PER_SECOND
        >= service.max_request
    )
    assert service.estimated_peak_rss_bytes <= 512 * 1024 * 1024


def test_live_ipc_rejects_profiles_outside_checkpoint_or_memory_envelope(
    hosted, tmp_path  # noqa: F811
):
    item = _restore_material(hosted, tmp_path)
    item["replacement"].profile = item["profile"].model_copy(
        update={"max_checkpoint_bytes": 64_000_000}
    )
    legacy_wide_service = ContextService(
        item["replacement"], checkpoint_receipt_keys=item["receipt_keys"]
    )
    assert (
        legacy_wide_service.max_live_hydrate_pack_bytes
        == LIVE_IPC_MAX_CHECKPOINT_BYTES
    )
    assert legacy_wide_service.engine.health_v3()["live_ipc_checkpoint_bytes"] == (
        LIVE_IPC_MAX_CHECKPOINT_BYTES
    )

    item["replacement"].profile = item["profile"].model_copy(
        update={
            "max_checkpoint_bytes": 16_000_000,
            "max_records": 200_000,
            "max_operations_per_cycle": 200_000,
        }
    )
    with pytest.raises(ContextFault, match="profile_transport_limit"):
        ContextService(item["replacement"], checkpoint_receipt_keys=item["receipt_keys"])


def test_concurrent_same_receipt_publication_converges(tmp_path):
    target = tmp_path / "hydrate-result.json"
    receipt = {"payload": {"fence": 2}, "key_id": "contextd", "signature": "a" * 88}
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: _write_restore_result_once(target, receipt), range(32)))
    assert target.read_bytes() == canonical(receipt) + b"\n"
    assert not list(tmp_path.glob(".hydrate-result.json.*.tmp"))


def test_existing_different_result_is_never_replaced(tmp_path):
    target = tmp_path / "hydrate-result.json"
    first = {"payload": {"fence": 2}, "key_id": "contextd", "signature": "a" * 88}
    second = {"payload": {"fence": 3}, "key_id": "contextd", "signature": "b" * 88}
    _write_restore_result_once(target, first)
    before = target.read_bytes()
    with pytest.raises(ContextFault, match="context_restore_result_conflict"):
        _write_restore_result_once(target, second)
    assert target.read_bytes() == before


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_restore_result_rejects_target_and_parent_symlinks(tmp_path):
    victim = tmp_path / "victim.json"
    victim.write_text("unchanged")
    target = tmp_path / "hydrate-result.json"
    linked_parent = tmp_path / "linked-parent"
    actual_parent = tmp_path / "actual-parent"
    actual_parent.mkdir()
    try:
        target.symlink_to(victim)
        linked_parent.symlink_to(actual_parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is not permitted")
    receipt = {"payload": {}, "key_id": "contextd", "signature": "a" * 88}
    with pytest.raises(ContextFault, match="context_restore_result_path_invalid"):
        _write_restore_result_once(target, receipt)
    with pytest.raises(ContextFault, match="context_restore_result_path_invalid"):
        _write_restore_result_once(linked_parent / "result.json", receipt)
    assert victim.read_text() == "unchanged"
    assert not (actual_parent / "result.json").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode contract")
def test_restore_result_rejects_preexisting_group_readable_file(tmp_path):
    target = tmp_path / "hydrate-result.json"
    receipt = {"payload": {}, "key_id": "contextd", "signature": "a" * 88}
    target.write_bytes(canonical(receipt) + b"\n")
    target.chmod(0o640)
    with pytest.raises(ContextFault, match="context_restore_result_path_invalid"):
        _write_restore_result_once(target, receipt)


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory-descriptor contract")
def test_restore_result_parent_swap_never_redirects_publish(tmp_path, monkeypatch):
    from aether_context import service

    parent = tmp_path / "result-parent"
    held = tmp_path / "held-parent"
    redirected = tmp_path / "redirected-parent"
    parent.mkdir(mode=0o700)
    redirected.mkdir(mode=0o700)
    target = parent / "hydrate-result.json"
    receipt = {"payload": {}, "key_id": "contextd", "signature": "a" * 88}
    real_open = service.os.open
    swapped = False

    def swap_before_temp_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if (
            not swapped
            and isinstance(path, str)
            and path.startswith(".hydrate-result.json.")
        ):
            swapped = True
            parent.rename(held)
            parent.symlink_to(redirected, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(service.os, "open", swap_before_temp_open)
    with pytest.raises(ContextFault, match="context_restore_result_path_invalid"):
        _write_restore_result_once(target, receipt)
    assert swapped is True
    assert not (redirected / target.name).exists()
    # A publish already linked through the stable descriptor remains confined to
    # the original directory and startup fails because its path identity changed.
    assert (held / target.name).read_bytes() == canonical(receipt) + b"\n"


@pytest.mark.skipif(os.name == "nt" or not hasattr(os, "mkfifo"), reason="POSIX FIFO")
def test_restore_result_rejects_fifo_before_open(tmp_path):
    target = tmp_path / "hydrate-result.json"
    os.mkfifo(target, 0o600)
    receipt = {"payload": {}, "key_id": "contextd", "signature": "a" * 88}
    with pytest.raises(ContextFault, match="context_restore_result_path_invalid"):
        _write_restore_result_once(target, receipt)


def test_failed_publish_never_exposes_partial_result(tmp_path, monkeypatch):
    from aether_context import service

    target = tmp_path / "hydrate-result.json"
    receipt = {"payload": {}, "key_id": "contextd", "signature": "a" * 88}

    def fail_link(*_args, **_kwargs):
        raise OSError("simulated link failure")

    monkeypatch.setattr(service.os, "link", fail_link)
    with pytest.raises(ContextFault, match="context_restore_result_unavailable"):
        _write_restore_result_once(target, receipt)
    assert not target.exists()
    assert not list(tmp_path.glob(".hydrate-result.json.*.tmp"))
