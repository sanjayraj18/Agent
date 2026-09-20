from __future__ import annotations

import json

from agent.events import AssistantEnd, AssistantStart, Usage
from agent.persistence.database import SqliteDatabase
from agent.persistence.event_log import EventLog
from agent.persistence.migrations import migrate
from agent.persistence.sessions import SessionStore
from agent.server.jsonrpc import JsonRpcServer
from agent.server.session_service import SessionService


def _server(tmp_path) -> JsonRpcServer:
    database = SqliteDatabase(tmp_path / "sessions.sqlite3")
    migrate(database)

    def runner_factory(session, approval_handler):
        async def runner(history, emit):
            yield emit(AssistantStart)
            yield emit(
                AssistantEnd,
                stop_reason="end_turn",
                usage=Usage(),
            )

        return runner

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


async def _messages(server: JsonRpcServer, payload: dict) -> list[dict]:
    return [
        message
        async for message in server.handle_line(json.dumps(payload))
    ]


async def test_session_rpc_creates_runs_replays_forks_and_archives(tmp_path):
    server = _server(tmp_path)
    created = await _messages(
        server,
        {"jsonrpc": "2.0", "id": 1, "method": "session.create"},
    )
    session_id = created[0]["result"]["session"]["session_id"]

    run = await _messages(
        server,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session.run",
            "params": {"session_id": session_id, "prompt": "hello"},
        },
    )
    replay = await _messages(
        server,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session.replay",
            "params": {"session_id": session_id},
        },
    )
    fork = await _messages(
        server,
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "session.fork",
            "params": {"session_id": session_id},
        },
    )
    archived = await _messages(
        server,
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "session.archive",
            "params": {"session_id": session_id},
        },
    )

    assert [message["params"]["event"]["type"] for message in run[:-1]] == [
        "session.started",
        "user.message",
        "assistant.start",
        "assistant.end",
    ]
    assert run[-1]["result"] == {
        "status": "completed",
        "session_id": session_id,
    }
    assert len(replay[0]["result"]["events"]) == 4
    assert fork[0]["result"]["session"]["parent_session_id"] == session_id
    assert archived[0]["result"]["session"]["status"] == "archived"


async def test_session_attach_requires_a_streaming_connection(tmp_path):
    server = _server(tmp_path)

    messages = await _messages(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session.attach",
            "params": {"session_id": "missing"},
        },
    )

    assert messages == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32020,
                "message": "session.attach requires a streaming connection",
            },
        }
    ]
