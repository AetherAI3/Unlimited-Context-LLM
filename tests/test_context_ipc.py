"""Actual Unix socket lifecycle; Windows exercises the engine directly."""

import asyncio
import json
import os
import socket
import stat
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from aether_context.service import ContextService
from aether_context.crypto import ContextFault
from test_hosted_context import hosted  # noqa: F401


@pytest.mark.skipif(os.name == "nt", reason="production daemon uses Unix-domain sockets")
def test_live_socket_ownership_crash_rebind_and_safe_errors(hosted, tmp_path):  # noqa: F811
    service = ContextService(hosted[0])
    socket_root = TemporaryDirectory(prefix="ctx-", dir="/tmp")
    path = Path(socket_root.name) / "context.sock"

    async def exercise():
        # A stale socket left by SIGKILL may be replaced; a live one may not.
        stale = socket.socket(getattr(socket, "AF_UNIX"), socket.SOCK_STREAM)
        stale.bind(str(path))
        stale.close()
        serving = asyncio.create_task(service.serve(str(path)))
        try:
            for _ in range(100):
                await asyncio.sleep(0.01)
                if serving.done():
                    await serving
                try:
                    reader, writer = await getattr(asyncio, "open_unix_connection")(str(path))
                    break
                except ConnectionRefusedError:
                    continue
            else:
                raise AssertionError("socket startup timed out")
            assert stat.S_IMODE(path.stat().st_mode) == 0o660
            writer.write(b'{"operation":"health"}\n')
            await writer.drain()
            response = json.loads(await reader.readline())
            writer.close()
            await writer.wait_closed()
            assert response["ok"] and not response["result"]["reach_claim_enabled"]
            with pytest.raises(ContextFault, match="socket_already_exists"):
                await service.serve(str(path))
            reader, writer = await getattr(asyncio, "open_unix_connection")(str(path))
            writer.write(b'{"operation":"unknown","text":"private-input-canary"}\n')
            await writer.drain()
            response = await reader.readline()
            assert b"private-input-canary" not in response and not json.loads(response)["ok"]
            writer.close()
            await writer.wait_closed()
        finally:
            serving.cancel()
            with pytest.raises(asyncio.CancelledError):
                await serving
        assert not path.exists()

    try:
        asyncio.run(exercise())
    finally:
        socket_root.cleanup()
