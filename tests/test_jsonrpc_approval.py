import asyncio
import io
import json
import queue

from agent.core.permission import PermissionResult, ToolAction
from agent.events import AssistantEnd, AssistantStart, Usage
from agent.providers.base import EventFactory
from agent.server.jsonrpc import JsonRpcServer


class QueueReader:
    """Blocking text reader backed by a test-controlled queue."""

    def __init__(self) -> None:
        self.lines: queue.Queue[str] = queue.Queue()

    def readline(self) -> str:
        return self.lines.get()


async def fallback_run(prompt: str):
    if False:
        yield


async def approval_aware_run(prompt: str, approval_handler):
    emit = EventFactory("approval-session")
    yield emit(AssistantStart)

    allowed = await approval_handler.approve(
        ToolAction("write_file", {"path": "notes.txt"}),
        PermissionResult(
            verdict="ask",
            reason="workspace writes require approval",
            risk="workspace_write",
        ),
    )

    assert allowed is True
    yield emit(
        AssistantEnd,
        stop_reason="end_turn",
        usage=Usage(),
    )


async def test_serve_accepts_an_approval_reply_while_a_run_is_streaming():
    reader = QueueReader()
    writer = io.StringIO()
    server = JsonRpcServer(
        fallback_run,
        run_agent_with_approval=approval_aware_run,
    )
    serving = asyncio.create_task(server.serve(reader, writer))

    reader.lines.put(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "run-1",
                "method": "agent.run",
                "params": {"prompt": "write notes"},
            }
        )
        + "\n"
    )

    for _ in range(100):
        await asyncio.sleep(0.01)
        messages = [
            json.loads(line)
            for line in writer.getvalue().splitlines()
        ]
        requests = [
            message
            for message in messages
            if message.get("method") == "agent.approval_requested"
        ]
        if requests:
            approval_id = requests[0]["params"]["approval_id"]
            break
    else:
        raise AssertionError("the server did not request approval")

    reader.lines.put(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "approval-1",
                "method": "agent.approval",
                "params": {"approval_id": approval_id, "allow": True},
            }
        )
        + "\n"
    )
    reader.lines.put("")

    await asyncio.wait_for(serving, timeout=2)
    messages = [
        json.loads(line)
        for line in writer.getvalue().splitlines()
    ]

    assert {
        "jsonrpc": "2.0",
        "id": "approval-1",
        "result": {"accepted": True},
    } in messages
    assert messages[-1] == {
        "jsonrpc": "2.0",
        "id": "run-1",
        "result": {"status": "completed"},
    }


async def test_noninteractive_handler_rejects_approval_replies_without_a_pending_run():
    server = JsonRpcServer(fallback_run)
    messages = [
        message
        async for message in server.handle_line(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": "approval-1",
                    "method": "agent.approval",
                    "params": {"approval_id": "unknown", "allow": True},
                }
            )
        )
    ]

    assert messages == [
        {
            "jsonrpc": "2.0",
            "id": "approval-1",
            "error": {
                "code": -32010,
                "message": "approval request is no longer pending",
            },
        }
    ]
