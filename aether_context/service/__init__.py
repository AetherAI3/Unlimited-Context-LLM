"""Private Unix-domain IPC. No public HTTP listener or model execution."""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
import errno
import secrets
import socket
import stat
from pathlib import Path

from pydantic import ValidationError

from ..contracts import (
    AppendRequestV1,
    ContextHydrateReceiptV1,
    ContextHydrateReceiptV2,
    LIVE_IPC_MAX_CHECKPOINT_BYTES,
    RetrieveRequestV1,
    canonical,
)
from ..crypto import ContextFault, verify
from ..engine import ContextEngine

MAX_REQUEST = 24_000_000
MAX_RESTORE_RESULT = 64_000
MAX_SIGNED_HYDRATE_ENVELOPE = 64_000
MAX_LIVE_HYDRATE_PACK_BYTES = LIVE_IPC_MAX_CHECKPOINT_BYTES
MAX_LIVE_CHECKPOINT_SNAPSHOT_BYTES = 16_000_000
LIVE_PROCESS_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
LIVE_MEMORY_SAFETY_BYTES = 32 * 1024 * 1024
MIN_LOCAL_IO_BYTES_PER_SECOND = 1_048_576
MIN_LOCAL_IO_TIMEOUT_SECONDS = 10
MAX_LOCAL_IO_TIMEOUT_SECONDS = 180
_EMPTY_CANONICAL_OBJECT_BYTES = len(canonical({}))
_HYDRATE_REQUEST_FIXED_BYTES = len(
    canonical(
        {
            "operation": "hydrate",
            "grant": {},
            "checkpoint_receipt": {},
            "pack": "",
        }
    )
) - (2 * _EMPTY_CANONICAL_OBJECT_BYTES)
# Valid hydrate headers fit in two signed-envelope bounds; valid bind headers
# may additionally contain a max-sized authority before project_snapshot.
HEADER_LIMIT = 600_000
SMALL_REQUEST_LIMIT = 400_000
_HEAVY_OPERATIONS = {"hydrate", "checkpoint"}
_BIND_OPERATIONS = {"bind", "bind_v2", "bind_v3"}
_SECURE_DIR_FD_AVAILABLE = (
    os.name != "nt"
    and all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW"))
    and all(
        call in os.supports_dir_fd
        for call in (os.open, os.stat, os.link, os.unlink)
    )
)


def _hydrate_request_limit(max_checkpoint_bytes: int) -> int:
    encoded_pack_bytes = 4 * ((max_checkpoint_bytes + 2) // 3)
    return (
        encoded_pack_bytes
        + (2 * MAX_SIGNED_HYDRATE_ENVELOPE)
        + _HYDRATE_REQUEST_FIXED_BYTES
    )


def _local_io_timeout_seconds(max_request_bytes: int) -> int:
    transfer_seconds = (
        max_request_bytes + MIN_LOCAL_IO_BYTES_PER_SECOND - 1
    ) // MIN_LOCAL_IO_BYTES_PER_SECOND
    timeout = max(MIN_LOCAL_IO_TIMEOUT_SECONDS, transfer_seconds + 5)
    if timeout > MAX_LOCAL_IO_TIMEOUT_SECONDS:
        raise ContextFault("context_profile_transport_limit")
    return timeout


def _resident_bytes() -> int:
    """Return current RSS on the production Linux host, failing closed elsewhere."""

    if os.name == "nt":
        # Unit tests on Windows do not host the Unix daemon. Keep construction
        # deterministic while serve() continues to reject this platform.
        return 0
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError) as exc:
        raise ContextFault("context_transport_memory_unavailable") from exc
    raise ContextFault("context_transport_memory_unavailable")


def _live_hydrate_peak_estimate(
    max_pack_bytes: int,
    *,
    max_snapshot_bytes: int,
    resident_cache_bytes: int,
    baseline_rss_bytes: int,
    max_records: int,
    max_operations: int,
    max_connections: int,
) -> int:
    """Conservative whole-object transport bound pending streaming framing."""

    return (
        baseline_rss_bytes
        + resident_cache_bytes
        + LIVE_MEMORY_SAFETY_BYTES
        + (8 * max_pack_bytes)
        + (4 * max_snapshot_bytes)
        + (2_048 * max_records)
        + (1_024 * max_operations)
        + (HEADER_LIMIT * max_connections)
    )


async def _read_request_frame(reader, *, max_request: int, timeout: int) -> bytes:
    line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    if not line or not line.endswith(b"\n"):
        raise ContextFault("context_request_framing")
    if len(line) > max_request + 1:
        raise ContextFault("context_request_limit")
    frame = line[:-1]
    if b"\n" in frame or b"\r" in frame:
        raise ContextFault("context_request_framing")
    return frame


def _json_no_duplicates(raw: bytes | bytearray) -> object:
    def closed_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    return json.loads(raw, object_pairs_hook=closed_object)


def _json_string_end(raw: bytes | bytearray, start: int) -> int | None:
    escaped = False
    for index in range(start + 1, len(raw)):
        value = raw[index]
        if escaped:
            escaped = False
        elif value == 0x5C:
            escaped = True
        elif value == 0x22:
            return index
    return None


def _top_level_operation(raw: bytes | bytearray) -> str | None:
    """Read only the depth-one operation key from an incomplete JSON object."""

    depth = 0
    index = 0
    while index < len(raw):
        value = raw[index]
        if value in (0x7B, 0x5B):  # { [
            depth += 1
            index += 1
            continue
        if value in (0x7D, 0x5D):  # } ]
            depth -= 1
            index += 1
            continue
        if value != 0x22:
            index += 1
            continue
        end = _json_string_end(raw, index)
        if end is None:
            return None
        if depth == 1:
            cursor = end + 1
            while cursor < len(raw) and raw[cursor] in b" \t":
                cursor += 1
            if cursor < len(raw) and raw[cursor] == 0x3A:  # :
                try:
                    key = json.loads(bytes(raw[index : end + 1]))
                except (UnicodeDecodeError, ValueError):
                    return ""
                if key == "operation":
                    cursor += 1
                    while cursor < len(raw) and raw[cursor] in b" \t":
                        cursor += 1
                    if cursor >= len(raw):
                        return None
                    if raw[cursor] != 0x22:
                        return ""
                    value_end = _json_string_end(raw, cursor)
                    if value_end is None:
                        return None
                    try:
                        operation = json.loads(bytes(raw[cursor : value_end + 1]))
                    except (UnicodeDecodeError, ValueError):
                        return ""
                    return operation if isinstance(operation, str) else ""
        index = end + 1
    return None


def _restore_requested(*paths: str | None) -> bool:
    """A recovery start is all-or-nothing, including its durable result path."""

    if any(paths) and not all(paths):
        raise SystemExit("context_restore_config_incomplete")
    return bool(paths[0])


def _safe_restore_result_path(value: str | Path) -> Path:
    path = Path(value).absolute()
    parent = path.parent
    try:
        for current in (parent, *parent.parents):
            info = current.lstat()
            if (
                stat.S_ISLNK(info.st_mode)
                or bool(getattr(info, "st_file_attributes", 0) & 0x400)
                or not stat.S_ISDIR(info.st_mode)
            ):
                raise ContextFault("context_restore_result_path_invalid")
    except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        raise ContextFault("context_restore_result_path_invalid") from exc
    return path


def _open_restore_parent(path: Path) -> int | None:
    """Open the parent through stable, no-follow directory handles on Unix."""

    if os.name == "nt":
        return None
    if not _SECURE_DIR_FD_AVAILABLE:
        raise ContextFault("context_restore_result_path_invalid")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parts = path.parent.parts
    if not path.is_absolute() or not parts:
        raise ContextFault("context_restore_result_path_invalid")
    current = os.open(path.anchor, flags)
    try:
        for component in parts[1:]:
            next_fd = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = next_fd
        info = os.fstat(current)
        getuid = getattr(os, "getuid", None)
        if (
            not stat.S_ISDIR(info.st_mode)
            or getuid is None
            or info.st_uid != getuid()
            or stat.S_IMODE(info.st_mode) & 0o022
        ):
            raise ContextFault("context_restore_result_path_invalid")
        return current
    except BaseException:
        os.close(current)
        raise


def _parent_path_matches(path: Path, parent_fd: int) -> bool:
    try:
        by_path = path.lstat()
        by_fd = os.fstat(parent_fd)
    except OSError:
        return False
    return (
        stat.S_ISDIR(by_path.st_mode)
        and not stat.S_ISLNK(by_path.st_mode)
        and by_path.st_dev == by_fd.st_dev
        and by_path.st_ino == by_fd.st_ino
    )


def _read_restore_result(path: Path, *, parent_fd: int | None = None) -> bytes:
    try:
        path_info = (
            path.lstat()
            if parent_fd is None
            else os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        )
    except FileNotFoundError:
        raise
    if (
        not stat.S_ISREG(path_info.st_mode)
        or stat.S_ISLNK(path_info.st_mode)
        or bool(getattr(path_info, "st_file_attributes", 0) & 0x400)
    ):
        raise ContextFault("context_restore_result_path_invalid")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(path if parent_fd is None else path.name, flags, dir_fd=parent_fd)
    except (FileNotFoundError, FileExistsError):
        raise
    except OSError as exc:
        raise ContextFault("context_restore_result_path_invalid") from exc
    try:
        info = os.fstat(fd)
        getuid = getattr(os, "getuid", None)
        if (
            not stat.S_ISREG(info.st_mode)
            or bool(getattr(info, "st_file_attributes", 0) & 0x400)
            or info.st_dev != path_info.st_dev
            or info.st_ino != path_info.st_ino
            or info.st_size > MAX_RESTORE_RESULT
            or (
                os.name != "nt"
                and (
                    stat.S_IMODE(info.st_mode) != 0o600
                    or getuid is None
                    or info.st_uid != getuid()
                )
            )
        ):
            raise ContextFault("context_restore_result_path_invalid")
        chunks: list[bytes] = []
        remaining = MAX_RESTORE_RESULT + 1
        while remaining:
            chunk = os.read(fd, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > MAX_RESTORE_RESULT:
            raise ContextFault("context_restore_result_path_invalid")
        return raw
    finally:
        os.close(fd)


def _read_bounded_regular_file(
    value: str | Path,
    limit: int,
    *,
    failure: str,
) -> bytes:
    """Read one stable regular-file identity without allocating past its limit."""

    path = Path(value)
    try:
        path_info = path.lstat()
    except OSError as exc:
        raise ContextFault(failure) from exc
    if (
        not stat.S_ISREG(path_info.st_mode)
        or stat.S_ISLNK(path_info.st_mode)
        or bool(getattr(path_info, "st_file_attributes", 0) & 0x400)
    ):
        raise ContextFault(failure)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ContextFault(failure) from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_dev != path_info.st_dev
            or info.st_ino != path_info.st_ino
        ):
            raise ContextFault(failure)
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(fd, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > limit:
            raise ContextFault(failure)
        return raw
    except OSError as exc:
        raise ContextFault(failure) from exc
    finally:
        os.close(fd)


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_restore_result_once(value: str | Path, receipt: dict) -> None:
    """Publish one complete signed hydrate receipt without replacing prior evidence."""

    target = _safe_restore_result_path(value)
    raw = canonical(receipt) + b"\n"
    if len(raw) > MAX_RESTORE_RESULT:
        raise ContextFault("context_restore_result_invalid")
    parent_fd = _open_restore_parent(target)
    try:
        try:
            existing = _read_restore_result(target, parent_fd=parent_fd)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if hmac.compare_digest(existing, raw):
                return
            raise ContextFault("context_restore_result_conflict")

        temporary_name = "." + target.name + "." + secrets.token_hex(16) + ".tmp"
        temporary = target.parent / temporary_name
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        linked = False
        try:
            if parent_fd is not None and not _parent_path_matches(target.parent, parent_fd):
                raise ContextFault("context_restore_result_path_invalid")
            if parent_fd is None:
                fd = os.open(temporary, flags, 0o600)
            else:
                fd = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
            try:
                if os.name != "nt":
                    os.fchmod(fd, 0o600)
                info = os.fstat(fd)
                getuid = getattr(os, "getuid", None)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or (
                        os.name != "nt"
                        and (
                            stat.S_IMODE(info.st_mode) != 0o600
                            or getuid is None
                            or info.st_uid != getuid()
                        )
                    )
                ):
                    raise ContextFault("context_restore_result_path_invalid")
                offset = 0
                while offset < len(raw):
                    written = os.write(fd, raw[offset:])
                    if written <= 0:
                        raise OSError("short restore-result write")
                    offset += written
                os.fsync(fd)
            finally:
                os.close(fd)
            try:
                if parent_fd is None:
                    os.link(temporary, target, follow_symlinks=False)
                else:
                    os.link(
                        temporary_name,
                        target.name,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                linked = True
            except FileExistsError:
                existing = _read_restore_result(target, parent_fd=parent_fd)
                if not hmac.compare_digest(existing, raw):
                    raise ContextFault("context_restore_result_conflict") from None
            if parent_fd is None:
                _sync_directory(target.parent)
            else:
                os.fsync(parent_fd)
                if not _parent_path_matches(target.parent, parent_fd):
                    raise ContextFault("context_restore_result_path_invalid")
        except ContextFault:
            raise
        except OSError as exc:
            raise ContextFault("context_restore_result_unavailable") from exc
        finally:
            try:
                if parent_fd is None:
                    temporary.unlink(missing_ok=True)
                else:
                    try:
                        os.unlink(temporary_name, dir_fd=parent_fd)
                    except FileNotFoundError:
                        pass
            except OSError:
                if not linked:
                    raise ContextFault("context_restore_result_unavailable") from None
        if parent_fd is None:
            _sync_directory(target.parent)
        else:
            os.fsync(parent_fd)
    except ContextFault:
        raise
    except OSError as exc:
        raise ContextFault("context_restore_result_unavailable") from exc
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def _restore_engine(
    engine: ContextEngine,
    *,
    profile,
    pack_path: str,
    grant_path: str,
    receipt_path: str,
    result_path: str,
    receipt_keys: dict,
) -> dict:
    pack_limit = min(
        profile.max_checkpoint_bytes, MAX_LIVE_HYDRATE_PACK_BYTES
    )
    pack_bytes = _read_bounded_regular_file(
        pack_path, pack_limit, failure="context_pack_limit"
    )
    grant_bytes = _read_bounded_regular_file(
        grant_path,
        MAX_SIGNED_HYDRATE_ENVELOPE,
        failure="context_restore_input_invalid",
    )
    receipt_bytes = _read_bounded_regular_file(
        receipt_path,
        MAX_SIGNED_HYDRATE_ENVELOPE,
        failure="context_restore_input_invalid",
    )
    try:
        grant = _json_no_duplicates(grant_bytes)
        checkpoint_receipt = _json_no_duplicates(receipt_bytes)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ContextFault("context_restore_input_invalid") from exc
    if not isinstance(grant, dict) or not isinstance(checkpoint_receipt, dict):
        raise ContextFault("context_restore_input_invalid")
    result = engine.hydrate(
        grant,
        checkpoint_receipt,
        pack_bytes,
        receipt_keys,
        snapshot_max_bytes=min(
            profile.max_checkpoint_bytes, MAX_LIVE_CHECKPOINT_SNAPSHOT_BYTES
        ),
    )
    try:
        payload = verify(
            result,
            {engine.signer.key_id: engine.signer.key.public_key()},
        )
        receipt_model = (
            ContextHydrateReceiptV2
            if payload.get("schema_version") == "ContextHydrateReceiptV2"
            else ContextHydrateReceiptV1
        )
        validated = receipt_model.model_validate(payload).model_dump(mode="json")
    except (ContextFault, TypeError, ValueError) as exc:
        raise ContextFault("context_restore_result_invalid") from exc
    if payload != validated:
        raise ContextFault("context_restore_result_invalid")
    _write_restore_result_once(result_path, result)
    return result


class ContextService:
    def __init__(
        self,
        engine: ContextEngine,
        *,
        checkpoint_receipt_keys: dict | None = None,
    ):
        self.engine = engine
        self.checkpoint_receipt_keys = dict(checkpoint_receipt_keys or {})
        self.max_small_operations = engine.profile.max_concurrent_cycles
        # One active heavy frame, one prefix-only heavy waiter, and a small
        # operation for every concurrency lane. Two slots cover health/control.
        self.max_accepted_connections = self.max_small_operations + 4
        self.max_live_hydrate_pack_bytes = min(
            engine.profile.max_checkpoint_bytes,
            MAX_LIVE_HYDRATE_PACK_BYTES,
        )
        self.max_live_checkpoint_snapshot_bytes = min(
            engine.profile.max_checkpoint_bytes,
            MAX_LIVE_CHECKPOINT_SNAPSHOT_BYTES,
        )
        self.max_hydrate_request = _hydrate_request_limit(
            self.max_live_hydrate_pack_bytes
        )
        self.baseline_rss_bytes = _resident_bytes()
        self.estimated_peak_rss_bytes = _live_hydrate_peak_estimate(
            self.max_live_hydrate_pack_bytes,
            max_snapshot_bytes=self.max_live_checkpoint_snapshot_bytes,
            resident_cache_bytes=engine.profile.resident_cache_bytes,
            baseline_rss_bytes=self.baseline_rss_bytes,
            max_records=engine.profile.max_records,
            max_operations=engine.profile.max_operations_per_cycle,
            max_connections=self.max_accepted_connections,
        )
        if self.estimated_peak_rss_bytes > LIVE_PROCESS_MEMORY_LIMIT_BYTES:
            raise ContextFault("context_profile_transport_limit")
        self.max_request = max(MAX_REQUEST, self.max_hydrate_request)
        self.io_timeout_seconds = _local_io_timeout_seconds(self.max_request)
        self.heavy_gate = asyncio.Semaphore(1)
        self.bind_gate = asyncio.Semaphore(1)
        self.small_gate = asyncio.Semaphore(self.max_small_operations)
        self._heavy_admitted = 0
        self._connection_tasks: set[asyncio.Task] = set()

    def dispatch(self, request: dict) -> dict:
        return self._dispatch(request, consume_hydrate_pack=False)

    def _dispatch(self, request: dict, *, consume_hydrate_pack: bool) -> dict:
        operation = request.get("operation")
        fields = {
            "health": {"operation"},
            "health_v2": {"operation"},
            "health_v3": {"operation"},
            "reserve": {"operation", "reservation"},
            "reserve_v2": {"operation", "reservation"},
            "reserve_v3": {"operation", "reservation"},
            "release_reservation": {"operation", "command"},
            "expire_reservation": {"operation", "command"},
            "abort_reservation": {"operation", "command"},
            "expire_aborted_reservation": {"operation", "command"},
            "bind": {"operation", "binding", "authority", "project_snapshot"},
            "bind_v2": {"operation", "binding", "authority", "project_snapshot"},
            "bind_v3": {"operation", "binding", "authority", "project_snapshot"},
            "checkpoint_durability": {"operation", "command"},
            "hydrate": {"operation", "grant", "checkpoint_receipt", "pack"},
            "append": {"operation", "capability", "request"},
            "retrieve": {"operation", "capability", "request"},
            "checkpoint": {"operation", "capability", "idempotency_key"},
            "seal": {"operation", "capability", "proof", "checkpoint_receipt"},
            "control": {"operation", "command"},
            "status": {"operation", "capability", "after"},
            "expire": {"operation", "capability"},
            "retention": {"operation", "capability", "command"},
            "freeze": {"operation", "capability"},
        }
        if operation not in fields or set(request) != fields[operation]:
            raise ContextFault("context_request_schema")
        request_limit = (
            self.max_hydrate_request
            if operation == "hydrate"
            else MAX_REQUEST
            if operation in {"bind", "bind_v2", "bind_v3"}
            else 400_000
        )
        if operation != "hydrate" and len(canonical(request)) > request_limit:
            raise ContextFault("context_request_limit")
        if operation == "health":
            return self.engine.health()
        if operation == "health_v2":
            return self.engine.health_v2()
        if operation == "health_v3":
            return self.engine.health_v3()
        if operation == "reserve":
            return self.engine.reserve(request["reservation"])
        if operation == "reserve_v2":
            return self.engine.reserve_v2(request["reservation"])
        if operation == "reserve_v3":
            return self.engine.reserve_v3(request["reservation"])
        if operation == "release_reservation":
            return self.engine.release_reservation(request["command"])
        if operation == "expire_reservation":
            return self.engine.expire_reservation(request["command"])
        if operation == "abort_reservation":
            return self.engine.abort_reservation(request["command"])
        if operation == "expire_aborted_reservation":
            return self.engine.expire_aborted_reservation(request["command"])
        if operation == "bind":
            return self.engine.bind(
                request["binding"], request["authority"], request["project_snapshot"]
            )
        if operation == "bind_v2":
            return self.engine.bind_v2(
                request["binding"], request["authority"], request["project_snapshot"]
            )
        if operation == "bind_v3":
            return self.engine.bind_v3(
                request["binding"], request["authority"], request["project_snapshot"]
            )
        if operation == "hydrate":
            if not self.checkpoint_receipt_keys:
                raise ContextFault("context_hydrate_unavailable")
            encoded_pack = request["pack"]
            if (
                not isinstance(request["grant"], dict)
                or not isinstance(request["checkpoint_receipt"], dict)
                or not isinstance(encoded_pack, str)
            ):
                raise ContextFault("context_request_schema")
            grant_bytes = canonical(request["grant"])
            checkpoint_receipt_bytes = canonical(request["checkpoint_receipt"])
            if (
                len(grant_bytes) > MAX_SIGNED_HYDRATE_ENVELOPE
                or len(checkpoint_receipt_bytes) > MAX_SIGNED_HYDRATE_ENVELOPE
                or _HYDRATE_REQUEST_FIXED_BYTES
                + len(grant_bytes)
                + len(checkpoint_receipt_bytes)
                + len(encoded_pack)
                > request_limit
            ):
                raise ContextFault("context_request_limit")
            if len(encoded_pack) > 4 * ((self.max_live_hydrate_pack_bytes + 2) // 3):
                raise ContextFault("context_request_limit")
            try:
                pack = base64.b64decode(encoded_pack, validate=True)
            except (TypeError, ValueError) as exc:
                raise ContextFault("context_request_schema") from exc
            if (
                len(pack) > self.max_live_hydrate_pack_bytes
                or base64.b64encode(pack).decode("ascii") != encoded_pack
            ):
                raise ContextFault("context_request_limit")
            if consume_hydrate_pack:
                # handle() owns its parsed request. Release the large base64
                # string before decrypting/materializing the checkpoint while
                # keeping the public in-process dispatch() API non-mutating.
                request.pop("pack")
            del encoded_pack
            return self.engine.hydrate(
                request["grant"],
                request["checkpoint_receipt"],
                pack,
                self.checkpoint_receipt_keys,
                snapshot_max_bytes=self.max_live_checkpoint_snapshot_bytes,
            )
        if operation == "append":
            return self.engine.append(
                request["capability"], AppendRequestV1.model_validate(request["request"])
            )
        if operation == "retrieve":
            return self.engine.retrieve(
                request["capability"], RetrieveRequestV1.model_validate(request["request"])
            )
        if operation == "checkpoint":
            return self.engine.checkpoint(
                request["capability"],
                request["idempotency_key"],
                transport_max_bytes=self.max_live_hydrate_pack_bytes,
                transport_snapshot_max_bytes=(
                    self.max_live_checkpoint_snapshot_bytes
                ),
            )
        if operation == "checkpoint_durability":
            return self.engine.register_checkpoint_durability(request["command"])
        if operation == "seal":
            return self.engine.seal(
                request["capability"], request["proof"], request["checkpoint_receipt"]
            )
        if operation == "control":
            return self.engine.control(request["command"])
        if operation == "expire":
            return self.engine.expire(request["capability"])
        if operation == "retention":
            return self.engine.retention(request["capability"], request["command"])
        if operation == "freeze":
            return self.engine.freeze(request["capability"])
        return self.engine.status(request["capability"], request["after"])

    async def _acquire_request_gate(self, operation: str, writer) -> str:
        transport = writer.transport
        transport.pause_reading()
        if operation in _HEAVY_OPERATIONS:
            # One active and at most one prefix-only waiter. A third large caller
            # is rejected before it can fill a StreamReader.
            if self._heavy_admitted >= 2:
                raise ContextFault("context_service_busy")
            self._heavy_admitted += 1
            try:
                await self.heavy_gate.acquire()
            except BaseException:
                self._heavy_admitted -= 1
                raise
            transport.resume_reading()
            return "heavy"
        if operation in _BIND_OPERATIONS:
            if self.bind_gate.locked():
                raise ContextFault("context_service_busy")
            await self.bind_gate.acquire()
            transport.resume_reading()
            return "bind"
        await self.small_gate.acquire()
        transport.resume_reading()
        return "small"

    def _release_request_gate(self, gate: str | None) -> None:
        if gate == "heavy":
            self.heavy_gate.release()
            self._heavy_admitted -= 1
        elif gate == "bind":
            self.bind_gate.release()
        elif gate == "small":
            self.small_gate.release()

    async def _read_classified_request(self, reader, writer) -> tuple[bytearray, str, str]:
        buffer = bytearray()
        operation: str | None = None
        gate: str | None = None
        limit = HEADER_LIMIT
        deadline = asyncio.get_running_loop().time() + self.io_timeout_seconds
        try:
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                chunk = await asyncio.wait_for(reader.read(65_536), timeout=remaining)
                if not chunk:
                    raise ContextFault("context_request_framing")
                buffer.extend(chunk)
                if b"\r" in chunk:
                    raise ContextFault("context_request_framing")
                newline = buffer.find(b"\n")
                if newline >= 0 and newline != len(buffer) - 1:
                    raise ContextFault("context_request_framing")
                if operation is None:
                    observed_operation = _top_level_operation(buffer)
                    if observed_operation is not None:
                        operation = observed_operation
                        gate = await self._acquire_request_gate(operation, writer)
                        limit = (
                            self.max_hydrate_request
                            if operation == "hydrate"
                            else MAX_REQUEST
                            if operation in _BIND_OPERATIONS
                            else SMALL_REQUEST_LIMIT
                        )
                if len(buffer) > limit + 1:
                    raise ContextFault("context_request_limit")
                if newline >= 0:
                    if operation is None or gate is None:
                        raise ContextFault("context_request_schema")
                    return buffer[:-1], operation, gate
                if operation is None and len(buffer) > HEADER_LIMIT:
                    raise ContextFault("context_request_limit")
        except BaseException:
            self._release_request_gate(gate)
            raise

    async def handle(self, reader, writer):
        gate: str | None = None
        try:
            frame, classified_operation, gate = await self._read_classified_request(
                reader, writer
            )
            request = _json_no_duplicates(frame)
            del frame
            if (
                not isinstance(request, dict)
                or request.get("operation") != classified_operation
            ):
                raise ContextFault("context_request_schema")
            result = await asyncio.to_thread(
                self._dispatch, request, consume_hydrate_pack=True
            )
            response = {"ok": True, "result": result}
        except ContextFault as exc:
            response = {"ok": False, "error": exc.code}
        except (ValueError, KeyError, TypeError, ValidationError, TimeoutError):
            response = {"ok": False, "error": "context_request_invalid"}
        except Exception:
            # Never echo exceptions with caller text or encryption material.
            response = {"ok": False, "error": "context_service_failure"}
        try:
            writer.write(canonical(response) + b"\n")
            await asyncio.wait_for(
                writer.drain(), timeout=self.io_timeout_seconds
            )
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            finally:
                self._release_request_gate(gate)

    def _accept_connection(self, reader, writer) -> None:
        """Pause at accept time so excess peers cannot fill StreamReader buffers."""

        writer.transport.pause_reading()
        if len(self._connection_tasks) >= self.max_accepted_connections:
            writer.write(
                canonical({"ok": False, "error": "context_service_busy"}) + b"\n"
            )
            writer.close()
            return
        task = asyncio.create_task(self._handle_accepted_connection(reader, writer))
        self._connection_tasks.add(task)
        task.add_done_callback(self._connection_tasks.discard)

    async def _handle_accepted_connection(self, reader, writer) -> None:
        writer.transport.resume_reading()
        await self.handle(reader, writer)

    async def serve(self, socket_path: str):
        if os.name == "nt":
            raise ContextFault("context_daemon_unix_required")
        path = Path(socket_path)
        if path.exists() or path.is_symlink():
            info = path.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != getattr(os, "getuid")():
                raise ContextFault("context_socket_already_exists")
            with socket.socket(getattr(socket, "AF_UNIX"), socket.SOCK_STREAM) as probe:
                try:
                    probe.connect(str(path))
                except OSError as exc:
                    if exc.errno != errno.ECONNREFUSED:
                        raise ContextFault("context_socket_already_exists") from None
                else:
                    raise ContextFault("context_socket_already_exists")
            if path.lstat().st_ino != info.st_ino:
                raise ContextFault("context_socket_already_exists")
            path.unlink()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        old = os.umask(0o077)
        try:
            server = await getattr(asyncio, "start_unix_server")(
                self._accept_connection, path=str(path), limit=HEADER_LIMIT
            )
        finally:
            os.umask(old)
        os.chmod(path, 0o660)
        inode = path.lstat().st_ino
        try:
            async with server:
                await server.serve_forever()
        finally:
            if (
                path.exists()
                and path.lstat().st_ino == inode
                and stat.S_ISSOCK(path.lstat().st_mode)
            ):
                path.unlink()


def main() -> None:
    """Credential paths should be supplied by systemd LoadCredential."""
    import argparse
    import sqlite3
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from ..contracts import ContextProfileV1
    from ..crypto import EnvelopeCipher, ReceiptSigner

    parser = argparse.ArgumentParser(description="Private Aether Context daemon")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--restore-pack")
    parser.add_argument("--restore-grant")
    parser.add_argument("--restore-receipt")
    parser.add_argument("--restore-result")
    parser.add_argument("--cycle-key-directory")
    parser.add_argument("--benchmark-receipt")
    parser.add_argument("--runtime-build-manifest")
    parser.add_argument("--managed-key-provider-receipt")
    args = parser.parse_args()
    restore = _restore_requested(
        args.restore_pack,
        args.restore_grant,
        args.restore_receipt,
        args.restore_result,
    )
    root = Path(args.credentials)

    def public_keys(filename):
        return {
            key: Ed25519PublicKey.from_public_bytes(base64.b64decode(value, validate=True))
            for key, value in json.loads((root / filename).read_text()).items()
        }

    profile = ContextProfileV1.model_validate_json(Path(args.profile).read_text())
    if profile.index_version != sqlite3.sqlite_version:
        raise SystemExit("context_index_version_mismatch")
    cycle_keys = None
    if args.cycle_key_directory:
        from ..retention import FileCycleKeyProvider

        cycle_keys = FileCycleKeyProvider(
            args.cycle_key_directory,
            EnvelopeCipher(
                (root / "context-cycle-wrapper-key").read_bytes(),
                domain=b"context-cycle-wrapper/v1",
            ),
        )
    benchmark_receipt = (
        json.loads(Path(args.benchmark_receipt).read_text()) if args.benchmark_receipt else None
    )
    runtime_build_manifest = (
        json.loads(Path(args.runtime_build_manifest).read_text())
        if args.runtime_build_manifest
        else None
    )
    managed_key_provider_config = (
        json.loads(Path(args.managed_key_provider_receipt).read_text())
        if args.managed_key_provider_receipt
        else None
    )
    if managed_key_provider_config is not None and not isinstance(
        managed_key_provider_config, (dict, list)
    ):
        raise SystemExit("context_key_provider_config_invalid")
    managed_key_provider_receipt = (
        managed_key_provider_config
        if isinstance(managed_key_provider_config, dict)
        else None
    )
    managed_key_provider_receipts = (
        managed_key_provider_config
        if isinstance(managed_key_provider_config, list)
        else None
    )
    engine = ContextEngine(
        args.database,
        profile,
        EnvelopeCipher((root / "context-dek").read_bytes()),
        ReceiptSigner(
            "contextd-v1",
            Ed25519PrivateKey.from_private_bytes((root / "context-signing-key").read_bytes()),
        ),
        public_keys("context-authority-keys.json"),
        public_keys("context-proof-keys.json"),
        cycle_keys=cycle_keys,
        benchmark_receipt=benchmark_receipt,
        benchmark_keys=(
            public_keys("context-benchmark-keys.json") if benchmark_receipt is not None else None
        ),
        remote_deletion_keys=(
            public_keys("context-remote-deletion-keys.json") if cycle_keys is not None else None
        ),
        executor_stability_keys=(
            public_keys("context-executor-stability-keys.json")
            if benchmark_receipt is not None
            else None
        ),
        managed_key_provider_receipt=managed_key_provider_receipt,
        managed_key_provider_receipts=managed_key_provider_receipts,
        key_provider_keys=(
            public_keys("context-key-provider-keys.json")
            if managed_key_provider_config is not None
            else None
        ),
        key_destruction_keys=(
            public_keys("context-key-destruction-keys.json")
            if managed_key_provider_config is not None
            else None
        ),
        runtime_build_manifest=runtime_build_manifest,
        runtime_build_keys=(
            public_keys("context-runtime-build-keys.json")
            if runtime_build_manifest is not None
            else None
        ),
        data_plane_isolation_keys=(
            public_keys("context-data-plane-isolation-keys.json")
            if benchmark_receipt is not None
            else None
        ),
    )
    checkpoint_receipt_keys = public_keys("context-checkpoint-keys.json")
    if restore:
        _restore_engine(
            engine,
            profile=profile,
            pack_path=args.restore_pack,
            grant_path=args.restore_grant,
            receipt_path=args.restore_receipt,
            result_path=args.restore_result,
            receipt_keys=checkpoint_receipt_keys,
        )
    asyncio.run(
        ContextService(
            engine, checkpoint_receipt_keys=checkpoint_receipt_keys
        ).serve(args.socket)
    )
