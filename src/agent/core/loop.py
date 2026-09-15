from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Sequence

from agent.core.conversation import messages_from_events
from agent.core.retry import RetryPolicy
from agent.events import (
    AssistantEnd,
    ErrorEvent,
    Event,
    ToolCallEnd,
    ToolCallStart,
    ToolResult,
)
from agent.providers.base import (
    EventFactory,
    Provider,
    ProviderRequest,
    ToolUsePart,
)
from agent.tools.dispatcher import ToolDispatcher
from agent.tools.registry import ToolRegistry


@dataclass
class _PendingToolCall:
    call_id: str
    name: str
    arguments: dict[str, object] | None = None


class AgentLoop:
    """The headless engine that drives provider turns and tool calls."""

    def __init__(
        self,
        provider: Provider,
        request_template: ProviderRequest,
        registry: ToolRegistry,
        max_iterations: int = 10,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        self._provider = provider
        self._request_template = request_template
        self._registry = registry
        self._dispatcher = ToolDispatcher(registry)
        self._max_iterations = max_iterations
        self._retry_policy = retry_policy or RetryPolicy()


    async def run(self, initial_events: Sequence[Event],emit: EventFactory) -> AsyncIterator[Event]:
        try:
            async for event in self._run(initial_events, emit):
                yield event

        except asyncio.CancelledError:
            yield emit(
                ErrorEvent,
                kind="interrupted",
                message="agent run was interrupted",
                retryable=False,
            )


    async def _run(
        self,
        initial_events: Sequence[Event],
        emit: EventFactory,
    ) -> AsyncIterator[Event]:
        """Yield every event produced while completing one agent task."""
        history = list(initial_events)

        for _ in range(self._max_iterations):
            retries_completed = 0
            pending_calls: dict[int, _PendingToolCall] = {}
            assistant_end: AssistantEnd | None = None

            while True:
                request = self._request_template.model_copy(
                    update={
                        "messages": messages_from_events(history),
                        "tools": self._registry.specs(),
                    }
                )

                pending_calls = {}
                assistant_end = None
                provider_error: ErrorEvent | None = None

                async for event in self._provider.stream(request, emit):
                    history.append(event)
                    yield event

                    if isinstance(event, ToolCallStart):
                        pending_calls[event.index] = _PendingToolCall(
                            call_id=event.call_id,
                            name=event.name,
                        )

                    elif isinstance(event, ToolCallEnd):
                        pending = pending_calls.get(event.index)
                        if (
                            pending is not None
                            and pending.call_id == event.call_id
                        ):
                            pending.arguments = event.arguments

                    elif isinstance(event, AssistantEnd):
                        assistant_end = event

                    elif isinstance(event, ErrorEvent):
                        provider_error = event
                        break

                if provider_error is None:
                    break

                if not self._retry_policy.should_retry(
                    provider_error,
                    retries_completed,
                ):
                    return

                retries_completed += 1
                delay = self._retry_policy.delay_for(
                    provider_error,
                    retry_number=retries_completed,
                )
                await asyncio.sleep(delay)

            if assistant_end is None:
                error = emit(
                    ErrorEvent,
                    kind="provider_protocol_error",
                    message="provider stream ended without assistant.end",
                    retryable=False,
                )
                history.append(error)
                yield error
                return

            if assistant_end.stop_reason == "tool_use":
                calls: list[ToolUsePart] = []
                incomplete_call = False

                for index in sorted(pending_calls):
                    pending = pending_calls[index]

                    if pending.arguments is None:
                        incomplete_call = True
                        continue

                    calls.append(
                        ToolUsePart(
                            call_id=pending.call_id,
                            name=pending.name,
                            arguments=pending.arguments,
                        )
                    )

                if incomplete_call or not calls:
                    error = emit(
                        ErrorEvent,
                        kind="incomplete_tool_call",
                        message=(
                            "provider stopped for tool use without "
                            "complete tool-call arguments"
                        ),
                        retryable=False,
                    )
                    history.append(error)
                    yield error
                    return

                results = await self._dispatcher.dispatch_all(calls)

                for result in results:
                    tool_result = emit(
                        ToolResult,
                        call_id=result.call_id,
                        content=result.content,
                        is_error=result.is_error,
                    )
                    history.append(tool_result)
                    yield tool_result

                continue

            if assistant_end.stop_reason == "pause_turn":
                # The paused assistant message is already in history.
                # Resume without adding an artificial user message.
                continue

            # end_turn, max_tokens, refusal, and interrupted end this run.
            return

        error = emit(
            ErrorEvent,
            kind="iteration_limit",
            message=(
                f"agent exceeded the iteration limit "
                f"({self._max_iterations})"
            ),
            retryable=False,
        )
        yield error
