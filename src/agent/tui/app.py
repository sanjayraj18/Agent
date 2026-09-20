from __future__ import annotations

import json

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header

from agent.events import PermissionApprovalRequested
from agent.tui.rpc_client import AgentRpcClient, RpcClientError
from agent.tui.state import RunStatus, TuiState, UiNotice
from agent.tui.widgets.approval_dialog import ApprovalDialog
from agent.tui.widgets.diff_viewer import DiffDocument, DiffViewer
from agent.tui.widgets.prompt_input import PromptInput
from agent.tui.widgets.status_bar import StatusBar
from agent.tui.widgets.tool_activity import ToolActivityView
from agent.tui.widgets.transcript import TranscriptView


class AgentTuiApp(App[None]):
    """Interactive terminal client for the headless local agent server."""

    TITLE = "Agent"
    SUB_TITLE = "Local coding agent"

    BINDINGS = [
        Binding(
            "ctrl+l",
            "clear_session",
            "Clear session",
            show=True,
        ),
        Binding(
            "ctrl+t",
            "toggle_thinking",
            "Show thinking",
            show=True,
        ),
        Binding(
            "ctrl+q",
            "quit",
            "Quit",
            show=True,
        ),
        Binding(
            "ctrl+d",
            "show_latest_diff",
            "View latest diff",
            show=True,
        ),
        Binding(
            "ctrl+n",
            "new_session",
            "New session",
            show=True,
        ),
        Binding(
            "ctrl+r",
            "resume_recent_session",
            "Resume latest",
            show=True,
        ),
        Binding(
            "ctrl+f",
            "fork_session",
            "Fork session",
            show=True,
        ),
    ]

    CSS = """
    #app-body {
        height: 1fr;
        padding: 1 1 0 1;
    }

    #conversation-column {
        width: 1fr;
        min-width: 40;
    }

    #tool-column {
        width: 42;
        min-width: 30;
        margin-left: 1;
    }

    #prompt-input {
        height: 8;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        *,
        client: AgentRpcClient | None = None,
    ) -> None:
        super().__init__()
        self._client = client or AgentRpcClient()
        self._state = TuiState()
        self._show_thinking = False
        self._session_id: str | None = None

    @property
    def state(self) -> TuiState:
        """Expose the current display state for focused UI tests."""

        return self._state

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield StatusBar()

        with Horizontal(id="app-body"):
            with Vertical(id="conversation-column"):
                yield TranscriptView()
            with Vertical(id="tool-column"):
                yield ToolActivityView()

        yield PromptInput()
        yield Footer()

    def on_mount(self) -> None:
        self._render_state()
        self.query_one(PromptInput).focus()

    def on_prompt_input_submitted(
        self,
        event: PromptInput.Submitted,
    ) -> None:
        if self._state.status == RunStatus.RUNNING:
            self.notify(
                "The agent is already working on a prompt.",
                severity="warning",
            )
            return

        self.query_one(PromptInput).clear()
        self.run_worker(
            self._run_prompt(event.value),
            exclusive=True,
            group="agent-run",
            description="Run agent prompt",
        )

    def action_clear_session(self) -> None:
        """Clear only the local display; it never deletes workspace data."""

        if self._state.status == RunStatus.RUNNING:
            self.notify(
                "Wait for the active run before clearing the display.",
                severity="warning",
            )
            return

        self._state.reset()
        self._session_id = None
        self._render_state()

    def action_new_session(self) -> None:
        """Start a new durable conversation; prior history stays archived."""

        if self._state.status == RunStatus.RUNNING:
            self.notify(
                "Wait for the active run before starting a new session.",
                severity="warning",
            )
            return

        self._session_id = None
        self._state.reset()
        self._render_state()
        self.notify("A new durable session will start with your next prompt.")

    def action_resume_recent_session(self) -> None:
        """Restore the most recently updated session into this TUI."""

        if self._state.status == RunStatus.RUNNING:
            self.notify(
                "Wait for the active run before switching sessions.",
                severity="warning",
            )
            return

        self.run_worker(
            self._resume_recent_session(),
            exclusive=True,
            group="session-control",
            description="Resume latest session",
        )

    def action_fork_session(self) -> None:
        """Branch the current history so experimentation never rewrites it."""

        if self._state.status == RunStatus.RUNNING:
            self.notify(
                "Wait for the active run before forking the session.",
                severity="warning",
            )
            return
        if self._session_id is None:
            self.notify("There is no durable session to fork yet.")
            return

        self.run_worker(
            self._fork_current_session(),
            exclusive=True,
            group="session-control",
            description="Fork session",
        )

    def action_toggle_thinking(self) -> None:
        self._show_thinking = not self._show_thinking
        self._render_state()

    def action_show_latest_diff(self) -> None:
        """Open the newest completed write/edit without pausing the agent."""

        change = self._state.latest_file_change

        if change is None:
            self.notify("No completed text-file changes yet.")
            return

        path = change.get("path")
        before = change.get("before")
        after = change.get("after")

        if (
            not isinstance(path, str)
            or not isinstance(before, str)
            or not isinstance(after, str)
        ):
            self.notify(
                "The latest file-change metadata is invalid.",
                severity="error",
            )
            return

        self.push_screen(
            DiffViewer(
                DiffDocument(
                    path=path,
                    before=before,
                    after=after,
                )
            )
        )

    async def _run_prompt(self, prompt: str) -> None:
        """Stream one headless-agent run into TuiState and redraw widgets."""

        self._state.status = RunStatus.RUNNING
        self._render_state()

        try:
            if self._session_id is None:
                session = await self._client.create_session()
                self._session_id = session.session_id

            async for item in self._client.run(
                prompt,
                session_id=self._session_id,
            ):
                if isinstance(item, PermissionApprovalRequested):
                    await self._handle_approval(item)
                    continue

                self._state.apply(item)
                self._render_state()

            if self._state.status == RunStatus.RUNNING:
                self._state.status = (
                    RunStatus.FAILED
                    if self._client.last_run_status == "failed"
                    else RunStatus.COMPLETED
                )
                self._render_state()

        except RpcClientError as exc:
            self._state.status = RunStatus.FAILED
            notice = UiNotice(kind="rpc_error", text=str(exc))
            self._state.last_error = notice
            self._state.notices.append(notice)
            self._render_state()
            self.notify(
                "Agent connection failed. Check the terminal diagnostics.",
                severity="error",
            )

    async def _resume_recent_session(self) -> None:
        try:
            sessions = await self._client.list_sessions()
            if not sessions:
                self.notify("There are no saved sessions yet.")
                return

            session = sessions[0]
            events = await self._client.replay_session(session.session_id)

            self._state.reset()
            self._state.session_id = session.session_id
            self._state.workspace = session.workspace
            self._state.model = session.model
            for event in events:
                self._state.apply(event)

            self._session_id = session.session_id
            self._render_state()
            self.notify(
                f"Resumed {session.title or session.session_id[:8]}."
            )
        except RpcClientError as exc:
            self.notify(f"Could not resume session: {exc}", severity="error")

    async def _fork_current_session(self) -> None:
        assert self._session_id is not None

        try:
            session = await self._client.fork_session(self._session_id)
            events = await self._client.replay_session(session.session_id)

            self._state.reset()
            self._state.session_id = session.session_id
            self._state.workspace = session.workspace
            self._state.model = session.model
            for event in events:
                self._state.apply(event)

            self._session_id = session.session_id
            self._render_state()
            self.notify(
                f"Forked into {session.title or session.session_id[:8]}."
            )
        except RpcClientError as exc:
            self.notify(f"Could not fork session: {exc}", severity="error")

    async def _handle_approval(
        self,
        request: PermissionApprovalRequested,
    ) -> None:
        """Display an explicit decision dialog, then reply over JSON-RPC."""

        approved = await self.push_screen_wait(
            ApprovalDialog(
                action=request.tool_name,
                details=_approval_details(request),
            )
        )
        allow = approved is True

        await self._client.respond_to_approval(
            request.approval_id,
            allow=allow,
        )
        self.notify(
            "Action allowed once." if allow else "Action denied.",
            severity="information" if allow else "warning",
        )

    def _render_state(self) -> None:
        """Redraw all widgets from one source of truth."""

        self.query_one(StatusBar).render_state(self._state)
        self.query_one(TranscriptView).render_state(
            self._state,
            show_thinking=self._show_thinking,
        )
        self.query_one(ToolActivityView).render_state(self._state)


def run_tui() -> None:
    """Launch the interactive client from the CLI entry point."""

    AgentTuiApp().run()


def _approval_details(request: PermissionApprovalRequested) -> str:
    """Render proposed tool input as literal dialog text, never Markdown."""

    arguments = json.dumps(
        request.arguments,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )
    return (
        f"Reason: {request.reason}\n"
        f"Risk: {request.risk}\n\n"
        f"Arguments:\n{arguments}"
    )
