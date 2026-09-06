"""Private Unix-domain IPC. No public HTTP listener or model execution."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import errno
import socket
import stat
from pathlib import Path

from pydantic import ValidationError

from ..contracts import AppendRequestV1, RetrieveRequestV1, canonical
from ..crypto import ContextFault
from ..engine import ContextEngine

MAX_REQUEST = 24_000_000


class ContextService:
    def __init__(self, engine: ContextEngine):
        self.engine = engine
        self.gate = asyncio.Semaphore(8)

    def dispatch(self, request: dict) -> dict:
        operation = request.get("operation")
        fields = {
            "health": {"operation"},
            "reserve": {"operation", "reservation"},
            "bind": {"operation", "binding", "authority", "project_snapshot"},
            "append": {"operation", "capability", "request"},
            "retrieve": {"operation", "capability", "request"},
            "checkpoint": {"operation", "capability", "idempotency_key"},
            "seal": {"operation", "capability", "proof", "checkpoint_receipt"},
            "control": {"operation", "command"},
            "status": {"operation", "capability", "after"},
            "expire": {"operation", "capability"},
            "freeze": {"operation", "capability"},
        }
        if operation not in fields or set(request) != fields[operation]:
            raise ContextFault("context_request_schema")
        if len(canonical(request)) > (MAX_REQUEST if operation == "bind" else 400_000):
            raise ContextFault("context_request_limit")
        if operation == "health":
            return self.engine.health()
        if operation == "reserve":
            return self.engine.reserve(request["reservation"])
        if operation == "bind":
            return self.engine.bind(
                request["binding"], request["authority"], request["project_snapshot"]
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
            return self.engine.checkpoint(request["capability"], request["idempotency_key"])
        if operation == "seal":
            return self.engine.seal(
                request["capability"], request["proof"], request["checkpoint_receipt"]
            )
        if operation == "control":
            return self.engine.control(request["command"])
        if operation == "expire":
            return self.engine.expire(request["capability"])
        if operation == "freeze":
            return self.engine.freeze(request["capability"])
        return self.engine.status(request["capability"], request["after"])

    async def handle(self, reader, writer):
        try:
            async with self.gate:
                line = await asyncio.wait_for(reader.readline(), timeout=5)
                if len(line) > MAX_REQUEST:
                    raise ContextFault("context_request_limit")
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ContextFault("context_request_schema")
                result = await asyncio.to_thread(self.dispatch, request)
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
            await asyncio.wait_for(writer.drain(), timeout=10)
        finally:
            writer.close()
            await writer.wait_closed()

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
                self.handle, path=str(path), limit=MAX_REQUEST + 1
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
    args = parser.parse_args()
    root = Path(args.credentials)

    def public_keys(filename):
        return {
            key: Ed25519PublicKey.from_public_bytes(base64.b64decode(value, validate=True))
            for key, value in json.loads((root / filename).read_text()).items()
        }

    profile = ContextProfileV1.model_validate_json(Path(args.profile).read_text())
    if profile.index_version != sqlite3.sqlite_version:
        raise SystemExit("context_index_version_mismatch")
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
    )
    restore = (args.restore_pack, args.restore_grant, args.restore_receipt)
    if any(restore):
        if not all(restore):
            raise SystemExit("context_restore_config_incomplete")
        pack_path = Path(args.restore_pack)
        if pack_path.stat().st_size > profile.max_checkpoint_bytes:
            raise SystemExit("context_pack_limit")
        engine.hydrate(
            json.loads(Path(args.restore_grant).read_text()),
            json.loads(Path(args.restore_receipt).read_text()),
            pack_path.read_bytes(),
            public_keys("context-checkpoint-keys.json"),
        )
    asyncio.run(ContextService(engine).serve(args.socket))
