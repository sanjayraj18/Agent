from __future__ import annotations

from decimal import Decimal

from rich.text import Text
from textual.widgets import Static

from agent.tui.state import RunStatus, TuiState


class StatusBar(Static):
    """Compact, continuously refreshed summary of the current run."""

    DEFAULT_CSS = """
    StatusBar {
        height: 3;
        padding: 0 1;
        border: round $accent;
        content-align: left middle;
    }
    """

    def __init__(
        self,
        *,
        widget_id: str = "status-bar",
    ) -> None:
        super().__init__(id=widget_id)

    def render_state(self, state: TuiState) -> None:
        """Refresh this bar from the framework-independent UI state."""

        self.update(build_status_text(state))


def build_status_text(state: TuiState) -> Text:
    """Build a width-safe Rich renderable for the session summary."""

    status_style = {
        RunStatus.IDLE: "dim",
        RunStatus.RUNNING: "yellow",
        RunStatus.COMPLETED: "green",
        RunStatus.FAILED: "red",
        RunStatus.INTERRUPTED: "yellow",
    }[state.status]

    usage = state.usage
    model = state.model or "no model selected"
    active_tool_count = len(state.active_tools)

    text = Text()
    text.append("● ", style=status_style)
    text.append(state.status.value.upper(), style=f"bold {status_style}")
    text.append(f"  ·  {model}", style="cyan")
    text.append(
        "  ·  "
        f"in {usage.input_tokens:,}"
        f" / out {usage.output_tokens:,}"
        f" / cache {usage.cache_read_tokens:,}",
    )
    text.append(
        f"  ·  {_format_cost(state.total_cost_usd)}",
        style="green" if state.total_cost_usd is not None else "dim",
    )

    if active_tool_count:
        suffix = "tool" if active_tool_count == 1 else "tools"
        text.append(
            f"  ·  {active_tool_count} {suffix} active",
            style="yellow",
        )

    return text


def _format_cost(cost: Decimal | None) -> str:
    if cost is None:
        return "pricing unavailable"

    return f"${cost:.5f}"
