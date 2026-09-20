from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from agent.core.costs import calculate_known_model_cost
from agent.events import (
    AssistantEnd,
    AssistantStart,
    ContextCompacted,
    ErrorEvent,
    Event,
    SessionStarted,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResult,
    Usage,
    UserMessage,
)


class RunStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ToolStatus(StrEnum):
    STREAMING = "streaming"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class TranscriptMessage:
    """One visible user or assistant message."""

    role: Literal["user", "assistant"]
    text: str = ""
    thinking: str = ""


@dataclass(slots=True)
class ToolActivity:
    """The current UI view of one requested tool call."""

    call_id: str
    name: str
    sequence: int
    status: ToolStatus = ToolStatus.STREAMING
    partial_json: str = ""
    arguments: dict[str, object] | None = None
    result: str | None = None
    is_error: bool = False
    file_change: dict[str, str] | None = None


@dataclass(slots=True)
class UiNotice:
    """A short system message shown outside the chat transcript."""

    kind: str
    text: str


@dataclass(slots=True)
class TokenTotals:
    """Cumulative token totals for the visible session."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def add(self, usage: Usage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_input_tokens
        self.cache_creation_tokens += (
            usage.cache_creation_input_tokens
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_creation_tokens
        )


@dataclass(slots=True)
class TuiState:
    """
    Framework-independent state for the terminal UI.

    Widgets read this state. The RPC client only calls apply(event).
    """

    session_id: str | None = None
    workspace: str | None = None
    model: str | None = None
    status: RunStatus = RunStatus.IDLE
    transcript: list[TranscriptMessage] = field(
        default_factory=list
    )
    tools: dict[str, ToolActivity] = field(
        default_factory=dict
    )
    notices: list[UiNotice] = field(default_factory=list)
    usage: TokenTotals = field(default_factory=TokenTotals)
    total_cost_usd: Decimal | None = Decimal("0")
    completed_turns: int = 0
    last_error: UiNotice | None = None
    _assistant_index: int | None = None

    def reset(self) -> None:
        """Return the UI to a clean state before a new session begins."""

        self.session_id = None
        self.workspace = None
        self.model = None
        self.status = RunStatus.IDLE
        self.transcript.clear()
        self.tools.clear()
        self.notices.clear()
        self.usage = TokenTotals()
        self.total_cost_usd = Decimal("0")
        self.completed_turns = 0
        self.last_error = None
        self._assistant_index = None

    def apply(self, event: Event) -> None:
        """Apply one validated agent event to the UI state."""

        if isinstance(event, SessionStarted):
            self.reset()
            self.session_id = event.session_id
            self.workspace = event.cwd
            self.model = event.model
            self.status = RunStatus.RUNNING
            return

        if isinstance(event, UserMessage):
            self.transcript.append(
                TranscriptMessage(
                    role="user",
                    text=event.text,
                )
            )
            self._assistant_index = None
            return

        if isinstance(event, AssistantStart):
            self._start_assistant_message()
            return

        if isinstance(event, TextDelta):
            self._assistant_message().text += event.text
            return

        if isinstance(event, ThinkingDelta):
            self._assistant_message().thinking += event.text
            return

        if isinstance(event, ToolCallStart):
            self.tools[event.call_id] = ToolActivity(
                call_id=event.call_id,
                name=event.name,
                sequence=event.seq,
            )
            return

        if isinstance(event, ToolCallDelta):
            tool = self._tool_for_event(event.call_id, event.seq)
            tool.partial_json += event.partial_json
            return

        if isinstance(event, ToolCallEnd):
            tool = self._tool_for_event(event.call_id, event.seq)
            tool.arguments = dict(event.arguments)
            tool.status = ToolStatus.RUNNING
            return

        if isinstance(event, ToolResult):
            tool = self._tool_for_event(event.call_id, event.seq)
            tool.result = event.content
            tool.is_error = event.is_error
            tool.file_change = event.file_change
            tool.status = (
                ToolStatus.FAILED
                if event.is_error
                else ToolStatus.COMPLETED
            )
            return

        if isinstance(event, ContextCompacted):
            self.notices.append(
                UiNotice(
                    kind="context_compacted",
                    text=(
                        "Context compacted: "
                        f"{event.discarded_message_count} old messages "
                        "were summarized."
                    ),
                )
            )
            return

        if isinstance(event, AssistantEnd):
            self._record_usage(event.usage)
            self._update_run_status(event)
            return

        if isinstance(event, ErrorEvent):
            notice = UiNotice(
                kind=event.kind,
                text=event.message,
            )
            self.last_error = notice
            self.notices.append(notice)
            self.status = (
                RunStatus.INTERRUPTED
                if event.kind == "interrupted"
                else RunStatus.FAILED
            )

    @property
    def ordered_tools(self) -> tuple[ToolActivity, ...]:
        """Tool calls in the order the provider requested them."""

        return tuple(
            sorted(
                self.tools.values(),
                key=lambda tool: tool.sequence,
            )
        )

    @property
    def active_tools(self) -> tuple[ToolActivity, ...]:
        """Tool calls that have not produced a result yet."""

        return tuple(
            tool
            for tool in self.ordered_tools
            if tool.status in {
                ToolStatus.STREAMING,
                ToolStatus.RUNNING,
            }
        )

    @property
    def latest_file_change(self) -> dict[str, str] | None:
        """Most recent confirmed text edit, for the non-blocking diff view."""

        for tool in reversed(self.ordered_tools):
            if tool.file_change is not None:
                return tool.file_change

        return None

    def _start_assistant_message(self) -> None:
        self.transcript.append(
            TranscriptMessage(role="assistant")
        )
        self._assistant_index = len(self.transcript) - 1

    def _assistant_message(self) -> TranscriptMessage:
        if self._assistant_index is None:
            self._start_assistant_message()

        assert self._assistant_index is not None
        return self.transcript[self._assistant_index]

    def _tool_for_event(
        self,
        call_id: str,
        sequence: int,
    ) -> ToolActivity:
        tool = self.tools.get(call_id)

        if tool is None:
            tool = ToolActivity(
                call_id=call_id,
                name="unknown",
                sequence=sequence,
            )
            self.tools[call_id] = tool

        return tool

    def _record_usage(self, usage: Usage) -> None:
        self.usage.add(usage)
        self.completed_turns += 1

        if self.total_cost_usd is None or self.model is None:
            return

        cost = calculate_known_model_cost(self.model, usage)

        if cost is None:
            self.total_cost_usd = None
            return

        self.total_cost_usd += cost.total_usd

    def _update_run_status(
        self,
        event: AssistantEnd,
    ) -> None:
        if event.stop_reason in {"tool_use", "pause_turn"}:
            return

        if event.stop_reason == "interrupted":
            self.status = RunStatus.INTERRUPTED
            return

        self.status = RunStatus.COMPLETED
