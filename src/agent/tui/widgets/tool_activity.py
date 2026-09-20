from __future__ import annotations

import json

from rich.table import Table
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

from agent.tui.state import ToolActivity, ToolStatus, TuiState


class ToolActivityView(VerticalScroll):
    """Shows the lifecycle and latest result of agent tool calls."""

    DEFAULT_CSS = """
    ToolActivityView {
        height: 12;
        border: round $secondary;
    }

    #tool-activity-content {
        padding: 0 1;
        width: 100%;
    }
    """

    def __init__(
        self,
        *,
        widget_id: str = "tool-activity",
    ) -> None:
        super().__init__(id=widget_id)
        self._content = Static(id="tool-activity-content")

    def compose(self):
        yield self._content

    def render_state(self, state: TuiState) -> None:
        """Refresh the visible tool table from TuiState."""

        self._content.update(build_tool_table(state))


def build_tool_table(state: TuiState) -> Table:
    """Create a Rich table without embedding terminal markup in tool data."""

    table = Table(
        title="Tool activity",
        expand=True,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Tool", style="bold cyan", width=18)
    table.add_column("Status", width=12)
    table.add_column("Details", ratio=1)

    if not state.ordered_tools:
        table.add_row(
            "—",
            "idle",
            "No tools have run in this task.",
        )
        return table

    for tool in state.ordered_tools:
        table.add_row(
            tool.name,
            _status_text(tool.status),
            _tool_details(tool),
        )

    return table


def _status_text(status: ToolStatus) -> Text:
    styles = {
        ToolStatus.STREAMING: "yellow",
        ToolStatus.RUNNING: "blue",
        ToolStatus.COMPLETED: "green",
        ToolStatus.FAILED: "red",
    }

    return Text(
        status.value,
        style=styles[status],
    )


def _tool_details(tool: ToolActivity) -> str:
    parts: list[str] = []

    if tool.arguments is not None:
        arguments = json.dumps(
            tool.arguments,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        parts.append(f"input: {_shorten(arguments)}")

    elif tool.partial_json:
        parts.append(
            f"building input: {_shorten(tool.partial_json)}"
        )

    if tool.result is not None:
        prefix = "error" if tool.is_error else "result"
        parts.append(
            f"{prefix}: {_shorten(tool.result, limit=240)}"
        )

    if tool.file_change is not None:
        path = tool.file_change.get("path", "workspace file")
        parts.append(f"changed: {path} (Ctrl+D to view diff)")

    return "\n".join(parts) or "Waiting for tool input..."


def _shorten(
    value: str,
    *,
    limit: int = 160,
) -> str:
    if len(value) <= limit:
        return value

    return f"{value[:limit - 1]}…"
