from dataclasses import dataclass, field
from typing import Literal, Sequence

from agent.core.ledger import FileEditRecord, TaskLedgerSnapshot, TodoItem
from agent.events import AssistantEnd, AssistantStart, ContextCompacted, ErrorEvent, Event, TextDelta, ThinkingDelta, ThinkingSignature, ToolCallEnd, ToolCallStart, ToolResult, UserMessage
from agent.providers.base import ContentPart, Message, TextPart, ThinkingPart, ToolResultPart, ToolUsePart


BlockKind = Literal["text", "thinking", "tool_use"]

#One piece/block of an assistant response while we're reconstructing it.
@dataclass
class _AssistantBlock:
    kind : BlockKind
    text_fragments : list[str] = field(default_factory=list)
    signature : str | None = None
    call_id : str | None = None
    name : str | None = None
    arguments : dict[str, object] | None = None


#for collecting blocks
def _get_or_create_block(blocks : dict[int, _AssistantBlock], index : int, kind : BlockKind) -> _AssistantBlock | None:
    block = blocks.get(index)

    if block is None:
        block = _AssistantBlock(kind=kind)
        blocks[index] = block
        return block

    if block.kind != kind:
        return None

    return block


#This function turns those temporary internal blocks into the final Messgaes:
def _assistant_message(blocks : dict[int, _AssistantBlock]) -> Message | None:

    content : list[ContentPart] = []

    for index in sorted(blocks):
        block = blocks[index]

        if block.kind == "text":
            text = "".join(block.text_fragments)
            if text:
                content.append(TextPart(text= text))

        if block.kind == "thinking":
            text = "".join(block.text_fragments)
            if text or block.signature is not None:
                content.append(
                    ThinkingPart(text=text, signature=block.signature)
                )

        elif block.kind == "tool_use":
            if(block.call_id is not None and block.name is not None and block.arguments is not None):
                content.append(
                    ToolUsePart(
                        call_id=block.call_id,
                        name=block.name,
                        arguments=block.arguments,
                    )
                )

    if not content:
        return None

    return Message(role="assistant", content=content)


#important function, will convert events to Message format which model needs
def messages_from_events(
    events: Sequence[Event],
    *,
    prefix_messages: Sequence[Message] = (),
) -> list[Message]:

    messages: list[Message] = list(prefix_messages)
    active_assistant : dict[int, _AssistantBlock] | None = None
    pending_tool_results : list[ContentPart] = []

    def flush_tool_results() -> None:
        nonlocal pending_tool_results

        if pending_tool_results:
            messages.append(
                Message(role="user", content=pending_tool_results)
            )
            pending_tool_results = []


    for event in events:
        if isinstance(event, UserMessage):
            active_assistant = None
            flush_tool_results()
            messages.append(
                Message(
                    role="user",
                    content=[TextPart(text=event.text)],
                )
            )

        elif isinstance(event, AssistantStart):
            flush_tool_results()
            active_assistant = {}


        elif isinstance(event, TextDelta):
            if active_assistant is None:
                continue

            block = _get_or_create_block(
                active_assistant,
                event.index,
                "text",
            )
            if block is not None:
                block.text_fragments.append(event.text)

        elif isinstance(event, ThinkingDelta):
            if active_assistant is None:
                continue

            block = _get_or_create_block(
                active_assistant,
                event.index,
                "thinking",
            )
            if block is not None:
                block.text_fragments.append(event.text)

        elif isinstance(event, ThinkingSignature):
            if active_assistant is None:
                continue

            block = _get_or_create_block(
                active_assistant,
                event.index,
                "thinking",
            )
            if block is not None:
                block.signature = event.signature

        elif isinstance(event, ToolCallStart):
            if active_assistant is None:
                continue

            block = _get_or_create_block(
                active_assistant,
                event.index,
                "tool_use",
            )
            if block is not None:
                block.call_id = event.call_id
                block.name = event.name

        elif isinstance(event, ToolCallEnd):
            if active_assistant is None:
                continue

            block = active_assistant.get(event.index)
            if block is not None and block.kind == "tool_use":
                block.arguments = event.arguments

        elif isinstance(event, AssistantEnd):
            if active_assistant is not None:
                message = _assistant_message(active_assistant)
                if message is not None:
                    messages.append(message)

            active_assistant = None

        elif isinstance(event, ToolResult):
            pending_tool_results.append(
                ToolResultPart(
                    call_id=event.call_id,
                    content=event.content,
                    is_error=event.is_error,
                )
            )

        elif isinstance(event, ContextCompacted):
            active_assistant = None
            flush_tool_results()

        elif isinstance(event, ErrorEvent):
            active_assistant = None

    flush_tool_results()
    return messages


def continuation_message(
    original_task: str,
    summary: str,
    ledger: TaskLedgerSnapshot,
) -> Message:
    """
    Build the compact message that starts a new context window.

    It contains only durable state: the original task, model-produced summary,
    confirmed workspace edits, and explicit TODOs.
    """
    if not original_task.strip():
        raise ValueError("original_task must not be empty")

    if not summary.strip():
        raise ValueError("summary must not be empty")

    text = (
        "You are continuing an existing coding task after the previous "
        "conversation was compacted.\n\n"
        f"Original task:\n{original_task}\n\n"
        f"Continuation summary:\n{summary}\n\n"
        f"Confirmed workspace edits:\n"
        f"{_render_file_edits(ledger.file_edits)}\n\n"
        f"Current TODOs:\n{_render_todos(ledger.todos)}\n\n"
        "Continue from the current workspace. Treat the workspace files as "
        "the source of truth and verify details before changing them."
    )

    return Message(
        role="user",
        content=[TextPart(text=text)],
    )


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
