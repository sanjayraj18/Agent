from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent.persistence.database import SqliteDatabase
from agent.persistence.event_log import EventLog
from agent.persistence.migrations import migrate
from agent.persistence.sessions import SessionStore
from agent.server.jsonrpc import JsonRpcServer
from agent.server.session_service import SessionService
from agent.server.unix_socket import UnixSocketServer


def _rpc(tmp_path) -> JsonRpcServer:
    database = SqliteDatabase(tmp_path / "sessions.sqlite3")
    migrate(database)

    def runner_factory(session, approval_handler):
        async def unused_runner(history, emit):
            if False:
                yield emit

        return unused_runner

    service = SessionService(
        sessions=SessionStore(database),
        event_log=EventLog(database),
        runner_factory=runner_factory,
    )
    return JsonRpcServer(
        session_service=service,
        default_workspace=tmp_path,
        default_model="claude-sonnet-5",
    )


async def test_unix_socket_serves_jsonrpc_with_owner_only_permissions(tmp_path):
    # macOS allows only a short AF_UNIX pathname. Pytest's normal temporary
    # directory includes the test name and exceeds that kernel limit.
    with TemporaryDirectory(prefix="agent-", dir="/private/tmp") as directory:
        socket_path = Path(directory) / "agent.sock"
        server = UnixSocketServer(rpc=_rpc(tmp_path), path=socket_path)
        try:
            await server.start()
        except PermissionError:
            pytest.skip(
                "the current host sandbox forbids Unix-domain sockets"
            )

        reader, writer = await asyncio.open_unix_connection(socket_path)
        writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "session.create",
                    }
                )
                + "\n"
            ).encode()
        )
        await writer.drain()
        response = json.loads(
            await asyncio.wait_for(reader.readline(), timeout=1)
        )

        writer.close()
        await writer.wait_closed()
        permissions = stat.S_IMODE(socket_path.stat().st_mode)
        await server.close()

        assert response["result"]["session"]["model"] == "claude-sonnet-5"
        assert permissions == 0o600
        assert not socket_path.exists()


async def test_unix_socket_refuses_to_replace_a_regular_file(tmp_path):
    socket_path = tmp_path / "agent.sock"
    socket_path.write_text("do not replace", encoding="utf-8")
    server = UnixSocketServer(rpc=_rpc(tmp_path), path=socket_path)

    with pytest.raises(RuntimeError, match="non-socket"):
        await server.start()
