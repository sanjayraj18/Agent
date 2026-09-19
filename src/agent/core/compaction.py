from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, cast

from agent.core.ledger import (
    FileEditRecord,
    TaskLedgerSnapshot,
    TodoItem,
    TodoStatus,
)
from agent.events import AssistantEnd, ErrorEvent, TextDelta
from agent.providers.base import (
    EventFactory,
    Message,
    Provider,
    ProviderRequest,
    TextPart,
)


DEFAULT_COMPACTION_MAX_TOKENS = 4_096


class ContextCompactionError(RuntimeError):
    """Raised when the agent cannot create a trustworthy continuation."""


@dataclass(frozen=True, slots=True)
class CompactionInput:
    """
    The bounded information given to the compactor.

    `recent_messages` must be small enough to fit in one request. The loop
    will ensure that it never passes an unbounded full conversation here.
    """

    original_task: str
    previous_summary: str | None
    ledger: TaskLedgerSnapshot
    recent_messages: tuple[Message, ...]

    def __post_init__(self) -> None:
        if not self.original_task.strip():
            raise ContextCompactionError(
                "original_task must not be empty"
            )


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """
    The durable handoff produced after one context window is compacted.

    File edits come from the ledger, never from model-generated text. That
    prevents a summary mistake from rewriting the record of workspace changes.
    """

    summary: str
    todos: tuple[TodoItem, ...]

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ContextCompactionError(
                "compaction summary must not be empty"
            )

        todo_ids = {todo.todo_id for todo in self.todos}

        if len(todo_ids) != len(self.todos):
            raise ContextCompactionError(
                "compaction result contains duplicate todo IDs"
            )


class ContextCompactor(Protocol):
    """The interface the agent loop needs for a summarize-and-continue step."""

    async def compact(
        self,
        source: CompactionInput,
        emit: EventFactory,
    ) -> CompactionResult:
        ...


class ProviderContextCompactor:
    """
    Use the configured LLM provider to produce a small structured handoff.

    This is an internal model call. Its response is not shown as a normal
    assistant answer; the loop will later emit one `context.compacted` event.
    """

    def __init__(
        self,
        provider: Provider,
        request_template: ProviderRequest,
        *,
        max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        self._provider = provider
        self._request_template = request_template
        self._max_tokens = max_tokens

    async def compact(
        self,
        source: CompactionInput,
        emit: EventFactory,
    ) -> CompactionResult:
        request = self._build_request(source)
        text_fragments: list[str] = []
        completed = False

        async for event in self._provider.stream(request, emit):
            if isinstance(event, TextDelta):
                text_fragments.append(event.text)

            elif isinstance(event, ErrorEvent):
                raise ContextCompactionError(
                    "compaction request failed: "
                    f"{event.kind}: {event.message}"
                )

            elif isinstance(event, AssistantEnd):
                if event.stop_reason != "end_turn":
                    raise ContextCompactionError(
                        "compaction ended unexpectedly: "
                        f"{event.stop_reason}"
                    )

                completed = True

        if not completed:
            raise ContextCompactionError(
                "compaction provider ended without assistant.end"
            )

        return parse_compaction_result("".join(text_fragments))

    def _build_request(
        self,
        source: CompactionInput,
    ) -> ProviderRequest:
        return self._request_template.model_copy(
            update={
                "system": _COMPACTION_SYSTEM_PROMPT,
                "stable_context": None,
                "cache_stable_prefix": False,
                "messages": [
                    Message(
                        role="user",
                        content=[
                            TextPart(
                                text=render_compaction_input(source)
                            )
                        ],
                    )
                ],
                "tools": [],
                "max_tokens": min(
                    self._max_tokens,
                    self._request_template.max_tokens,
                ),
                "effort": None,
                "thinking": False,
            }
        )


def render_compaction_input(source: CompactionInput) -> str:
    """Render the bounded task state supplied to the compaction model."""
    previous_summary = source.previous_summary or "(no previous summary)"
    recent_messages = _canonical_json(
        [
            message.model_dump(mode="json")
            for message in source.recent_messages
        ]
    )

    return (
        f"Original task:\n{source.original_task}\n\n"
        f"Previous running summary:\n{previous_summary}\n\n"
        f"Confirmed workspace edits:\n"
        f"{_render_file_edits(source.ledger.file_edits)}\n\n"
        f"Current TODOs:\n"
        f"{_render_todos(source.ledger.todos)}\n\n"
        f"Recent conversation to incorporate:\n"
        f"{recent_messages}"
    )


def parse_compaction_result(raw: str) -> CompactionResult:
    """
    Parse the exact JSON format requested from the compaction model.

    We fail instead of guessing if the model returns malformed state. A wrong
    continuation is worse than stopping and reporting a clear error.
    """
    payload_text = _strip_markdown_fence(raw)

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ContextCompactionError(
            "compaction response was not valid JSON"
        ) from exc

    if not isinstance(payload, dict):
        raise ContextCompactionError(
            "compaction response must be a JSON object"
        )

    summary = payload.get("summary")

    if not isinstance(summary, str):
        raise ContextCompactionError(
            "compaction response needs a string summary"
        )

    raw_todos = payload.get("todos", [])

    if not isinstance(raw_todos, list):
        raise ContextCompactionError(
            "compaction response todos must be a list"
        )

    todos = tuple(
        _parse_todo(raw_todo, index)
        for index, raw_todo in enumerate(raw_todos)
    )

    return CompactionResult(
        summary=summary,
        todos=todos,
    )


def _parse_todo(raw_todo: Any, index: int) -> TodoItem:
    if not isinstance(raw_todo, dict):
        raise ContextCompactionError(
            f"todo at index {index} must be an object"
        )

    todo_id = raw_todo.get("todo_id")
    description = raw_todo.get("description")
    status = raw_todo.get("status", "pending")

    if not isinstance(todo_id, str):
        raise ContextCompactionError(
            f"todo at index {index} needs a string todo_id"
        )

    if not isinstance(description, str):
        raise ContextCompactionError(
            f"todo at index {index} needs a string description"
        )

    if not isinstance(status, str):
        raise ContextCompactionError(
            f"todo at index {index} needs a string status"
        )

    try:
        return TodoItem(
            todo_id=todo_id,
            description=description,
            status=cast(TodoStatus, status),
        )
    except ValueError as exc:
        raise ContextCompactionError(
            f"todo at index {index} is invalid: {exc}"
        ) from exc


def _render_file_edits(
    file_edits: tuple[FileEditRecord, ...],
) -> str:
    if not file_edits:
        return "(none)"

    return "\n".join(
        f"- {edit.tool_name} {edit.path}: {edit.summary}"
        for edit in file_edits
    )


def _render_todos(todos: tuple[TodoItem, ...]) -> str:
    if not todos:
        return "(none)"

    return "\n".join(
        f"- [{todo.status}] {todo.todo_id}: {todo.description}"
        for todo in todos
    )


def _strip_markdown_fence(raw: str) -> str:
    """Accept a fenced JSON response, but nothing more ambiguous."""
    text = raw.strip()

    if not text.startswith("```"):
        return text

    lines = text.splitlines()

    if len(lines) < 3 or not lines[-1].strip() == "```":
        raise ContextCompactionError(
            "compaction response has an incomplete markdown fence"
        )

    return "\n".join(lines[1:-1]).strip()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


_COMPACTION_SYSTEM_PROMPT = """
You compact an agent's working context so it can safely continue one task.

Use only the supplied task, conversation, TODOs, and confirmed workspace edits.
Do not invent file changes, test results, or completed work.

Return only valid JSON in exactly this shape:

{
  "summary": "A concise factual continuation briefing.",
  "todos": [
    {
      "todo_id": "stable-short-id",
      "description": "One remaining task.",
      "status": "pending"
    }
  ]
}

Allowed TODO statuses are: pending, in_progress, done, blocked.
""".strip()