from agent.core.conversation import messages_from_events
from agent.events import (
    AssistantEnd,
    AssistantStart,
    ErrorEvent,
    TextDelta,
    ThinkingDelta,
    ThinkingSignature,
    ToolCallEnd,
    ToolCallStart,
    ToolResult,
    Usage,
    UserMessage,
)
from agent.providers.base import (
    EventFactory,
    Message,
    TextPart,
    ThinkingPart,
    ToolResultPart,
    ToolUsePart,
)


def _end(
    emit: EventFactory,
    stop_reason: str = "end_turn",
) -> AssistantEnd:
    return emit(
        AssistantEnd,
        stop_reason=stop_reason,
        usage=Usage(),
    )


def test_rebuilds_a_simple_user_and_assistant_turn():
    emit = EventFactory("session-1")

    events = [
        emit(UserMessage, text="What is 2 + 2?"),
        emit(AssistantStart),
        emit(TextDelta, index=0, text="2 + "),
        emit(TextDelta, index=0, text="2 = 4."),
        _end(emit),
    ]

    assert messages_from_events(events) == [
        Message(
            role="user",
            content=[TextPart(text="What is 2 + 2?")],
        ),
        Message(
            role="assistant",
            content=[TextPart(text="2 + 2 = 4.")],
        ),
    ]


def test_preserves_thinking_signature_and_orders_blocks_by_index():
    emit = EventFactory("session-1")

    events = [
        emit(UserMessage, text="Calculate 12 + 7."),
        emit(AssistantStart),
        # These deliberately arrive out of display order.
        emit(TextDelta, index=2, text="The answer is 19."),
        emit(
            ToolCallStart,
            index=1,
            call_id="tool-1",
            name="add_numbers",
        ),
        emit(
            ToolCallEnd,
            index=1,
            call_id="tool-1",
            arguments={"left": 12, "right": 7},
        ),
        emit(ThinkingDelta, index=0, text="I should calculate this."),
        emit(
            ThinkingSignature,
            index=0,
            signature="signed-thinking-block",
        ),
        _end(emit, stop_reason="tool_use"),
    ]

    assert messages_from_events(events) == [
        Message(
            role="user",
            content=[TextPart(text="Calculate 12 + 7.")],
        ),
        Message(
            role="assistant",
            content=[
                ThinkingPart(
                    text="I should calculate this.",
                    signature="signed-thinking-block",
                ),
                ToolUsePart(
                    call_id="tool-1",
                    name="add_numbers",
                    arguments={"left": 12, "right": 7},
                ),
                TextPart(text="The answer is 19."),
            ],
        ),
    ]


def test_groups_parallel_tool_results_into_one_user_message():
    emit = EventFactory("session-1")

    events = [
        emit(UserMessage, text="Add 12 + 7 and 8 + 5."),
        emit(AssistantStart),
        emit(
            ToolCallStart,
            index=0,
            call_id="tool-1",
            name="add_numbers",
        ),
        emit(
            ToolCallEnd,
            index=0,
            call_id="tool-1",
            arguments={"left": 12, "right": 7},
        ),
        emit(
            ToolCallStart,
            index=1,
            call_id="tool-2",
            name="add_numbers",
        ),
        emit(
            ToolCallEnd,
            index=1,
            call_id="tool-2",
            arguments={"left": 8, "right": 5},
        ),
        _end(emit, stop_reason="tool_use"),
        emit(
            ToolResult,
            call_id="tool-1",
            content="19",
        ),
        emit(
            ToolResult,
            call_id="tool-2",
            content="13",
            is_error=False,
        ),
    ]

    assert messages_from_events(events) == [
        Message(
            role="user",
            content=[TextPart(text="Add 12 + 7 and 8 + 5.")],
        ),
        Message(
            role="assistant",
            content=[
                ToolUsePart(
                    call_id="tool-1",
                    name="add_numbers",
                    arguments={"left": 12, "right": 7},
                ),
                ToolUsePart(
                    call_id="tool-2",
                    name="add_numbers",
                    arguments={"left": 8, "right": 5},
                ),
            ],
        ),
        Message(
            role="user",
            content=[
                ToolResultPart(call_id="tool-1", content="19"),
                ToolResultPart(call_id="tool-2", content="13"),
            ],
        ),
    ]


def test_excludes_an_incomplete_assistant_turn():
    emit = EventFactory("session-1")

    events = [
        emit(UserMessage, text="Explain events.py."),
        emit(AssistantStart),
        emit(TextDelta, index=0, text="I started answering but"),
        emit(
            ErrorEvent,
            kind="timeout_error",
            message="Connection timed out",
            retryable=True,
        ),
    ]

    assert messages_from_events(events) == [
        Message(
            role="user",
            content=[TextPart(text="Explain events.py.")],
        ),
    ]