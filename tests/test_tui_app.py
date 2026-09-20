import asyncio

from agent.events import PermissionApprovalRequested
from agent.tui.app import AgentTuiApp
from agent.tui.state import ToolActivity, ToolStatus
from agent.tui.widgets.diff_viewer import DiffViewer


class RecordingClient:
    def __init__(self) -> None:
        self.responses: list[tuple[str, bool]] = []

    async def respond_to_approval(
        self,
        approval_id: str,
        *,
        allow: bool,
    ) -> None:
        self.responses.append((approval_id, allow))


async def test_app_denies_an_approval_when_the_dialog_is_dismissed():
    client = RecordingClient()
    app = AgentTuiApp(client=client)
    request = PermissionApprovalRequested(
        approval_id="approval-1",
        request_id="run-1",
        tool_name="bash",
        arguments={"command": "git status"},
        reason="shell execution requires approval",
        risk="shell_execute",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        worker = app.run_worker(app._handle_approval(request))
        await pilot.pause()
        await pilot.press("escape")
        await worker.wait()

    assert client.responses == [("approval-1", False)]


async def test_app_opens_the_latest_file_diff_without_starting_a_new_run():
    app = AgentTuiApp(client=RecordingClient())

    async with app.run_test(size=(120, 40)) as pilot:
        app.state.tools["write-1"] = ToolActivity(
            call_id="write-1",
            name="write_file",
            sequence=1,
            status=ToolStatus.COMPLETED,
            file_change={
                "path": "notes.txt",
                "before": "before\n",
                "after": "after\n",
                "operation": "updated",
            },
        )

        app.action_show_latest_diff()
        await pilot.pause()

        assert isinstance(app.screen, DiffViewer)
        await pilot.press("escape")
