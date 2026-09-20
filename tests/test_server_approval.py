import asyncio

from agent.core.permission import PermissionResult, ToolAction
from agent.events import PermissionApprovalRequested
from agent.server.approval import ApprovalBroker


def _action() -> ToolAction:
    return ToolAction(
        tool_name="write_file",
        arguments={"path": "notes.txt"},
    )


def _decision() -> PermissionResult:
    return PermissionResult(
        verdict="ask",
        reason="workspace writes require approval",
        risk="workspace_write",
    )


async def test_broker_resolves_only_the_matching_pending_approval():
    notifications: list[PermissionApprovalRequested] = []

    async def notify(event: PermissionApprovalRequested) -> None:
        notifications.append(event)

    broker = ApprovalBroker(request_id="run-1", notify=notify)
    waiting = asyncio.create_task(broker.approve(_action(), _decision()))

    await asyncio.sleep(0)
    request = notifications[0]

    assert request.request_id == "run-1"
    assert request.tool_name == "write_file"
    assert request.arguments == {"path": "notes.txt"}
    assert broker.resolve("unknown", allow=True) is False
    assert broker.resolve(request.approval_id, allow=True) is True
    assert await waiting is True
    assert broker.pending_count == 0


async def test_broker_denies_pending_actions_when_the_client_disconnects():
    broker = ApprovalBroker(
        request_id="run-1",
        notify=lambda event: None,
    )
    waiting = asyncio.create_task(broker.approve(_action(), _decision()))

    await asyncio.sleep(0)
    broker.close()

    assert await waiting is False
    assert broker.pending_count == 0


async def test_broker_denies_when_the_notification_cannot_be_delivered():
    def broken_notify(event: PermissionApprovalRequested) -> None:
        raise OSError("connection closed")

    broker = ApprovalBroker(request_id="run-1", notify=broken_notify)

    assert await broker.approve(_action(), _decision()) is False
