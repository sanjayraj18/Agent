from agent.core.loop import AgentLoop
from agent.core.retry import RetryPolicy
from agent.events import (
    AssistantEnd,
    AssistantStart,
    ErrorEvent,
    TextDelta,
    Usage,
    UserMessage,
)
from agent.providers.base import (
    EventFactory,
    Message,
    ProviderRequest,
    TextPart,
)
from agent.tools.registry import ToolRegistry


class FailsOnceProvider:
    """Streams a partial answer, fails once, then succeeds on retry."""

    name = "fails-once"

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
                TextDelta,
                index=0,
                text="This partial answer must not enter retry history.",
            )
            yield emit(
                ErrorEvent,
                kind="rate_limit_error",
                message="Try again later",
                retryable=True,
            )
            return

        yield emit(AssistantStart)
        yield emit(TextDelta, index=0, text="Recovered successfully.")
        yield emit(
            AssistantEnd,
            stop_reason="end_turn",
            usage=Usage(),
        )


async def test_loop_retries_with_clean_conversation_history():
    emit = EventFactory("session-1")
    provider = FailsOnceProvider()

    loop = AgentLoop(
        provider=provider,
        request_template=ProviderRequest(
            model="test-model",
            max_tokens=256,
            messages=[],
        ),
        registry=ToolRegistry(),
        retry_policy=RetryPolicy(
            base_delay=0.001,
            jitter_fraction=0,
        ),
    )

    initial_events = [
        emit(UserMessage, text="Please answer this question."),
    ]

    events = [
        event
        async for event in loop.run(initial_events, emit)
    ]

    assert [event.type for event in events] == [
        "assistant.start",
        "assistant.text_delta",
        "error",
        "assistant.start",
        "assistant.text_delta",
        "assistant.end",
    ]

    assert len(provider.requests) == 2

    # The first attempt's partial assistant answer was discarded.
    assert provider.requests[1].messages == [
        Message(
            role="user",
            content=[TextPart(text="Please answer this question.")],
        )
    ]

    assert isinstance(events[-1], AssistantEnd)
    assert events[-1].stop_reason == "end_turn"