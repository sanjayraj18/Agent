import io
import json

import pytest

from agent.events import AssistantEnd, AssistantStart, ErrorEvent, TextDelta, Usage
from agent.providers.base import EventFactory
from agent.server.jsonrpc import JsonRpcServer


async def successful_run(prompt: str):
    emit = EventFactory("server-session")

    yield emit(AssistantStart)
    yield emit(TextDelta, index=0, text=f"Answer: {prompt}")
    yield emit(
        AssistantEnd,
        stop_reason="end_turn",
        usage=Usage(),
    )


async def failed_run(prompt: str):
    emit = EventFactory("server-session")

    yield emit(
        ErrorEvent,
        kind="tool_failure",
        message="The tool could not complete",
        retryable=False,
    )


async def crashing_run(prompt: str):
    if False:
        yield

    raise RuntimeError("unexpected failure")


async def _handle(
    server: JsonRpcServer,
    line: str,
) -> list[dict]:
    return [
        message
        async for message in server.handle_line(line)
    ]


async def test_valid_agent_run_streams_events_then_returns_completed():
    server = JsonRpcServer(successful_run)

    messages = await _handle(
        server,
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "agent.run",
                "params": {"prompt": "Hello"},
            }
        ),
    )

    assert [
        message["params"]["event"]["type"]
        for message in messages[:-1]
    ] == [
        "assistant.start",
        "assistant.text_delta",
        "assistant.end",
    ]

    assert messages[-1] == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"status": "completed"},
    }


async def test_agent_error_event_produces_failed_final_status():
    server = JsonRpcServer(failed_run)

    messages = await _handle(
        server,
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "request-1",
                "method": "agent.run",
                "params": {"prompt": "Hello"},
            }
        ),
    )

    assert messages[0]["method"] == "agent.event"
    assert messages[0]["params"]["event"]["type"] == "error"

    assert messages[-1] == {
        "jsonrpc": "2.0",
        "id": "request-1",
        "result": {"status": "failed"},
    }


async def test_invalid_json_returns_parse_error():
    server = JsonRpcServer(successful_run)

    messages = await _handle(server, "{not valid json")

    assert messages == [
        {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32700,
                "message": "parse error",
            },
        }
    ]


@pytest.mark.parametrize(
    ("rpc_request", "code", "message"),
    [
        (
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "unknown.method",
            },
            -32601,
            "method not found: unknown.method",
        ),
        (
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "agent.run",
                "params": {},
            },
            -32602,
            "params.prompt must be a non-empty string",
        ),
        (
            {
                "jsonrpc": "1.0",
                "id": 1,
                "method": "agent.run",
                "params": {"prompt": "Hello"},
            },
            -32600,
            "invalid request",
        ),
    ],
)
async def test_invalid_requests_return_standard_jsonrpc_errors(
    rpc_request: dict,
    code: int,
    message: str,
):
    server = JsonRpcServer(successful_run)

    messages = await _handle(server, json.dumps(rpc_request))

    assert messages == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": code,
                "message": message,
            },
        }
    ]


async def test_unexpected_agent_exception_returns_internal_error():
    server = JsonRpcServer(crashing_run)

    messages = await _handle(
        server,
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "agent.run",
                "params": {"prompt": "Hello"},
            }
        ),
    )

    assert messages == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32603,
                "message": "internal error",
            },
        }
    ]


async def test_serve_reads_stdin_like_input_and_writes_json_lines():
    server = JsonRpcServer(successful_run)
    reader = io.StringIO(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "agent.run",
                "params": {"prompt": "Hello"},
            }
        )
        + "\n"
    )
    writer = io.StringIO()

    await server.serve(reader, writer)

    output_lines = [
        json.loads(line)
        for line in writer.getvalue().splitlines()
    ]

    assert output_lines[-1] == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"status": "completed"},
    }
