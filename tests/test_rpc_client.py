from collections import deque

from agent.events import AssistantEnd, PermissionApprovalRequested, Usage
from agent.providers.base import EventFactory
from agent.tui.rpc_client import AgentRpcClient


async def test_client_handles_an_approval_response_after_a_deferred_agent_event(
    monkeypatch,
):
    client = AgentRpcClient(command=("unused",))
    emit = EventFactory("rpc-client-session")
    assistant_end = emit(
        AssistantEnd,
        stop_reason="end_turn",
        usage=Usage(),
    )
    messages = deque(
        [
            {
                "jsonrpc": "2.0",
                "method": "agent.approval_requested",
                "params": {
                    "type": "permission.approval_requested",
                    "approval_id": "approval-1",
                    "request_id": "tui-1",
                    "tool_name": "write_file",
                    "arguments": {"path": "notes.txt"},
                    "reason": "workspace writes require approval",
                    "risk": "workspace_write",
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "agent.event",
                "params": {
                    "request_id": "tui-1",
                    "event": assistant_end.model_dump(mode="json"),
                },
            },
            {
                "jsonrpc": "2.0",
                "id": "tui-2",
                "result": {"accepted": True},
            },
            {
                "jsonrpc": "2.0",
                "id": "tui-1",
                "result": {"status": "completed"},
            },
        ]
    )
    sent: list[dict] = []

    async def start() -> None:
        return None

    async def send(message: dict) -> None:
        sent.append(message)

    async def read_wire_message() -> dict:
        return messages.popleft()

    monkeypatch.setattr(client, "start", start)
    monkeypatch.setattr(client, "_send", send)
    monkeypatch.setattr(client, "_read_wire_message", read_wire_message)

    received = []
    async for item in client.run("write notes"):
        received.append(item)
        if isinstance(item, PermissionApprovalRequested):
            await client.respond_to_approval(item.approval_id, allow=True)

    assert [type(item) for item in received] == [
        PermissionApprovalRequested,
        AssistantEnd,
    ]
    assert client.last_run_status == "completed"
    assert [message["method"] for message in sent] == [
        "agent.run",
        "agent.approval",
    ]
