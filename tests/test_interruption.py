import asyncio

from agent.core.loop import AgentLoop
from agent.events import AssistantStart, ErrorEvent, UserMessage
from agent.providers.base import EventFactory, ProviderRequest
from agent.tools.registry import ToolRegistry


class BlockingProvider:
    """Starts a stream, then waits forever unless cancelled."""

    name = "blocking"

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def stream(
        self,
        request: ProviderRequest,
        emit: EventFactory,
    ):
        yield emit(AssistantStart)

        self.started.set()
        await asyncio.Event().wait()


async def _collect_events(
    loop: AgentLoop,
    initial_events: list[UserMessage],
    emit: EventFactory,
):
    return [
        event
        async for event in loop.run(initial_events, emit)
    ]


async def test_cancelling_a_stream_produces_a_clean_interrupted_event():
    emit = EventFactory("session-1")
    provider = BlockingProvider()

    loop = AgentLoop(
        provider=provider,
        request_template=ProviderRequest(
            model="test-model",
            max_tokens=256,
            messages=[],
        ),
        registry=ToolRegistry(),
    )

    initial_events = [
        emit(UserMessage, text="Begin a long-running task."),
    ]

    task = asyncio.create_task(
        _collect_events(loop, initial_events, emit)
    )

    await provider.started.wait()
    task.cancel()

    events = await task

    assert [event.type for event in events] == [
        "assistant.start",
        "error",
    ]

    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].kind == "interrupted"
    assert events[-1].retryable is False