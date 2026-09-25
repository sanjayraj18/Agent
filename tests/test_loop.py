from typing import Any

from agent.core.loop import AgentLoop
from agent.events import (
    AssistantEnd,
    AssistantStart,
    ErrorEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResult,
    Usage,
    UserMessage,
)
from agent.providers.base import (
    EventFactory,
    Message,
    ProviderRequest,
    TextPart,
    ToolResultPart,
    ToolUsePart,
)
from decimal import Decimal
from agent.routing.router import RuleBasedRouter
from agent.tools.base import Tool, ToolExecutionResult
from agent.tools.registry import ToolRegistry


class AddNumbersTool(Tool):
    name = "add_numbers"
    description = "Add two integers."
    input_schema = {
        "type": "object",
        "properties": {
            "left": {"type": "integer"},
            "right": {"type": "integer"},
        },
        "required": ["left", "right"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            content=str(arguments["left"] + arguments["right"])
        )


class TimingProvider:
    """Streams one visible text response for latency measurement tests."""

    name = "timing"

    async def stream(
        self,
        request: ProviderRequest,
        emit: EventFactory,
    ):
        yield emit(AssistantStart)
        yield emit(TextDelta, index=0, text="Measured response.")
        yield emit(
            AssistantEnd,
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
        )


class FixedClock:
    """Returns predetermined monotonic times instead of real time."""

    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        try:
            return next(self._values)
        except StopIteration as exc:
            raise AssertionError("clock was called too many times") from exc


class AddThenAnswerProvider:
    """First turn requests a tool; second turn answers using its result."""

    name = "add-then-answer"

    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    async def stream(
        self,
        request: ProviderRequest,
        emit: EventFactory,
    ):
        self.requests.append(request)

        if len(self.requests) == 1:
            yield emit(AssistantStart)
            yield emit(
                ToolCallStart,
                index=0,
                call_id="call-1",
                name="add_numbers",
            )
            yield emit(
                ToolCallEnd,
                index=0,
                call_id="call-1",
                arguments={"left": 12, "right": 7},
            )
            yield emit(
                AssistantEnd,
                stop_reason="tool_use",
                usage=Usage(),
            )
            return

        yield emit(AssistantStart)
        yield emit(TextDelta, index=0, text="12 + 7 = 19.")
        yield emit(
            AssistantEnd,
            stop_reason="end_turn",
            usage=Usage(),
        )


class PausingProvider:
    """First turn pauses; second turn completes normally."""

    name = "pausing"

    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    async def stream(
        self,
        request: ProviderRequest,
        emit: EventFactory,
    ):
        self.requests.append(request)

        if len(self.requests) == 1:
            yield emit(AssistantStart)
            yield emit(TextDelta, index=0, text="Searching...")
            yield emit(
                AssistantEnd,
                stop_reason="pause_turn",
                usage=Usage(),
            )
            return

        yield emit(AssistantStart)
        yield emit(TextDelta, index=0, text="Search complete.")
        yield emit(
            AssistantEnd,
            stop_reason="end_turn",
            usage=Usage(),
        )


class EndlessToolProvider:
    """Always asks for the same tool, so the iteration cap must stop it."""

    name = "endless-tool"

    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    async def stream(
        self,
        request: ProviderRequest,
        emit: EventFactory,
    ):
        self.requests.append(request)
        call_id = f"call-{len(self.requests)}"

        yield emit(AssistantStart)
        yield emit(
            ToolCallStart,
            index=0,
            call_id=call_id,
            name="add_numbers",
        )
        yield emit(
            ToolCallEnd,
            index=0,
            call_id=call_id,
            arguments={"left": 1, "right": 1},
        )
        yield emit(
            AssistantEnd,
            stop_reason="tool_use",
            usage=Usage(),
        )


class BrokenProvider:
    """Violates the provider contract by ending without AssistantEnd."""

    name = "broken"

    async def stream(
        self,
        request: ProviderRequest,
        emit: EventFactory,
    ):
        yield emit(AssistantStart)
        yield emit(TextDelta, index=0, text="This response never ends.")


def _request_template() -> ProviderRequest:
    return ProviderRequest(
        model="test-model",
        max_tokens=256,
        messages=[],
    )


def _initial_events(emit: EventFactory) -> list[UserMessage]:
    return [
        emit(
            UserMessage,
            text="Add 12 and 7, then explain the answer.",
        )
    ]


async def test_loop_completes_a_tool_using_task():
    emit = EventFactory("session-1")
    provider = AddThenAnswerProvider()

    loop = AgentLoop(
        provider=provider,
        request_template=_request_template(),
        registry=ToolRegistry([AddNumbersTool()]),
    )

    events = [
        event
        async for event in loop.run(
            _initial_events(emit),
            emit,
        )
    ]

    assert [event.type for event in events] == [
        "assistant.start",
        "tool.call_start",
        "tool.call_end",
        "assistant.end",
        "tool.result",
        "assistant.start",
        "assistant.text_delta",
        "assistant.end",
    ]

    assert isinstance(events[4], ToolResult)
    assert events[4].call_id == "call-1"
    assert events[4].content == "19"
    assert events[4].is_error is False

    assert len(provider.requests) == 2
    assert provider.requests[1].messages[-1] == Message(
        role="user",
        content=[
            ToolResultPart(
                call_id="call-1",
                content="19",
            )
        ],
    )


async def test_loop_resumes_pause_turn_without_an_extra_user_message():
    emit = EventFactory("session-1")
    provider = PausingProvider()

    loop = AgentLoop(
        provider=provider,
        request_template=_request_template(),
        registry=ToolRegistry(),
    )

    events = [
        event
        async for event in loop.run(
            _initial_events(emit),
            emit,
        )
    ]

    assert events[-1].type == "assistant.end"
    assert len(provider.requests) == 2

    assert provider.requests[1].messages == [
        Message(
            role="user",
            content=[
                TextPart(
                    text="Add 12 and 7, then explain the answer."
                )
            ],
        ),
        Message(
            role="assistant",
            content=[TextPart(text="Searching...")],
        ),
    ]


async def test_loop_stops_a_runaway_tool_cycle_at_the_iteration_limit():
    emit = EventFactory("session-1")

    loop = AgentLoop(
        provider=EndlessToolProvider(),
        request_template=_request_template(),
        registry=ToolRegistry([AddNumbersTool()]),
        max_iterations=2,
    )

    events = [
        event
        async for event in loop.run(
            _initial_events(emit),
            emit,
        )
    ]

    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].kind == "iteration_limit"
    assert events[-1].retryable is False


async def test_loop_reports_a_provider_that_ends_without_assistant_end():
    emit = EventFactory("session-1")

    loop = AgentLoop(
        provider=BrokenProvider(),
        request_template=_request_template(),
        registry=ToolRegistry(),
    )

    events = [
        event
        async for event in loop.run(
            _initial_events(emit),
            emit,
        )
    ]

    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].kind == "provider_protocol_error"
    assert events[-1].retryable is False


async def test_loop_records_provider_turn_timing():
    emit = EventFactory("session-1")

    clock = FixedClock((100.00, 100.25, 101.75))

    loop = AgentLoop(
        provider=TimingProvider(),
        request_template=_request_template(),
        registry=ToolRegistry(),
        monotonic_clock=clock,
    )

    events = [
        event
        async for event in loop.run(
            _initial_events(emit),
            emit,
        )
    ]

    assert events[-1].type == "assistant.end"

    turn = loop.telemetry.turns[0]

    assert turn.timing is not None
    assert turn.timing.provider_duration_seconds == Decimal("1.75")
    assert turn.timing.time_to_first_output_seconds == Decimal("0.25")


async def test_shadow_routing_records_a_recommendation_without_changing_model():
    """Shadow mode observes a route; the request still uses test-model."""

    emit = EventFactory("session-1")
    provider = AddThenAnswerProvider()

    loop = AgentLoop(
        provider=provider,
        request_template=_request_template(),
        registry=ToolRegistry([AddNumbersTool()]),
        shadow_router=RuleBasedRouter(),
    )

    events = [
        event
        async for event in loop.run(
            _initial_events(emit),
            emit,
        )
    ]

    assert events[-1].type == "assistant.end"
    assert [request.model for request in provider.requests] == [
        "test-model",
        "test-model",
    ]

    assert len(loop.telemetry.turns) == 2
    assert [
        turn.shadow_route.route_id
        for turn in loop.telemetry.turns
        if turn.shadow_route is not None
    ] == [
        "strong-terra-high",
        "strong-terra-high",
    ]
    assert loop.telemetry.shadow_routed_turn_count == 2
    assert loop.telemetry.shadow_model_difference_count == 2
