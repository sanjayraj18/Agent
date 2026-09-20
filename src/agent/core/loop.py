from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Literal, Sequence, cast

from agent.core.capabilities import (
    ModelCapabilities,
    capabilities_for_model,
)
from agent.core.compaction import (
    CompactionInput,
    ContextCompactionError,
    ContextCompactor,
    ProviderContextCompactor,
)
from agent.core.context import (
    DEFAULT_SAFETY_MARGIN_TOKENS,
    ContextAssessment,
    ContextBudgetError,
    TokenCounter,
    assess_request_context,
)
from agent.core.conversation import (
    continuation_message,
    messages_from_events,
)
from agent.core.ledger import (
    FileEditRecord,
    LedgerError,
    TaskLedger,
    TaskLedgerSnapshot,
)
from agent.core.permission import PermissionPolicy
from agent.core.retry import RetryPolicy
from agent.core.telemetry import SessionTelemetry
from agent.events import (
    AssistantEnd,
    ContextCompacted,
    ErrorEvent,
    Event,
    ToolCallEnd,
    ToolCallStart,
    ToolResult,
    UserMessage,
)
from agent.providers.base import (
    EventFactory,
    Message,
    Provider,
    ProviderRequest,
    ToolResultPart,
    ToolUsePart,
)
from agent.tools.dispatcher import ToolDispatcher
from agent.tools.registry import ToolRegistry


DEFAULT_COMPACTION_RESERVE_TOKENS = 8_192


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
        telemetry: SessionTelemetry | None = None,
        permissions: PermissionPolicy | None = None,
        context_compactor: ContextCompactor | None = None,
        model_capabilities: ModelCapabilities | None = None,
        token_counter: TokenCounter | None = None,
        context_safety_margin_tokens: int = (
            DEFAULT_SAFETY_MARGIN_TOKENS
        ),
        compaction_reserve_tokens: int = (
            DEFAULT_COMPACTION_RESERVE_TOKENS
        ),
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        if context_safety_margin_tokens < 0:
            raise ValueError(
                "context_safety_margin_tokens must not be negative"
            )

        if compaction_reserve_tokens < 1:
            raise ValueError(
                "compaction_reserve_tokens must be at least 1"
            )

        if (
            model_capabilities is not None
            and model_capabilities.model != request_template.model
        ):
            raise ValueError(
                "model_capabilities must match request_template.model"
            )

        self._provider = provider
        self._request_template = request_template
        self._registry = registry
        self._dispatcher = ToolDispatcher(registry, permissions=permissions)
        self._max_iterations = max_iterations
        self._telemetry = telemetry or SessionTelemetry()
        self._retry_policy = retry_policy or RetryPolicy()

        self._model_capabilities = (
            model_capabilities
            or capabilities_for_model(request_template.model)
        )
        self._token_counter = token_counter
        self._context_safety_margin_tokens = (
            context_safety_margin_tokens
        )
        self._compaction_reserve_tokens = compaction_reserve_tokens

        if context_compactor is not None:
            self._context_compactor = context_compactor
        elif self._model_capabilities is not None:
            self._context_compactor = ProviderContextCompactor(
                provider,
                request_template,
            )
        else:
            # Unknown models can still run normally. Context management is
            # disabled because we do not know their safe context boundary.
            self._context_compactor = None

        self._ledger = TaskLedger()

    @property
    def telemetry(self) -> SessionTelemetry:
        """Usage and cost totals collected during this agent run."""
        return self._telemetry

    @property
    def ledger(self) -> TaskLedgerSnapshot:
        """A read-only view of edits and TODOs preserved for this task."""
        return self._ledger.snapshot()

    async def run(
        self,
        initial_events: Sequence[Event],
        emit: EventFactory,
    ) -> AsyncIterator[Event]:
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
        original_task = _original_task(initial_events)
        continuation: Message | None = None
        previous_summary: str | None = None
        self._ledger = TaskLedger()
        untrusted_tool_output_seen = any(
            isinstance(event, ToolResult)
            for event in history
        )

        for _ in range(self._max_iterations):
            retries_completed = 0
            pending_calls: dict[int, _PendingToolCall] = {}
            assistant_end: AssistantEnd | None = None
            force_compaction = False

            while True:
                request = self._build_request(
                    history,
                    continuation,
                )

                try:
                    assessment = self._assess_context(request)
                except ContextBudgetError as exc:
                    error = emit(
                        ErrorEvent,
                        kind="invalid_context_budget",
                        message=str(exc),
                        retryable=False,
                    )
                    yield error
                    return

                recent_messages = tuple(messages_from_events(history))

                if (
                    assessment is not None
                    and self._should_compact(
                        assessment,
                        force_compaction=force_compaction,
                    )
                ):
                    if not recent_messages:
                        error = emit(
                            ErrorEvent,
                            kind="context_unrecoverable",
                            message=(
                                "the static prompt and continuation state "
                                "do not fit in the model context window"
                            ),
                            retryable=False,
                        )
                        yield error
                        return

                    try:
                        compacted_event, continuation, previous_summary = (
                            await self._compact_context(
                                original_task=original_task,
                                previous_summary=previous_summary,
                                recent_messages=recent_messages,
                                previous_assessment=assessment,
                                emit=emit,
                            )
                        )
                    except (
                        ContextCompactionError,
                        ContextBudgetError,
                        LedgerError,
                        ValueError,
                    ) as exc:
                        error = emit(
                            ErrorEvent,
                            kind="context_compaction_failed",
                            message=str(exc),
                            retryable=False,
                        )
                        yield error
                        return

                    # The continuation message replaces the entire old tail.
                    # The workspace and ledger keep the durable task state.
                    history.clear()
                    force_compaction = False
                    yield compacted_event
                    continue

                cache_plan = request.cache_plan
                stable_prefix_fingerprint = (
                    cache_plan.stable_fingerprint
                    if cache_plan is not None
                    else None
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
                        self._telemetry.record_turn(
                            model=request.model,
                            usage=event.usage,
                            stable_prefix_fingerprint=(
                                stable_prefix_fingerprint
                            ),
                        )

                    elif isinstance(event, ErrorEvent):
                        provider_error = event
                        break

                if provider_error is None:
                    break

                if (
                    provider_error.kind == "context_overflow"
                    and assessment is not None
                    and self._context_compactor is not None
                ):
                    # Our estimate was too low or provider rules differ.
                    # Retry only after compaction, never with the same request.
                    force_compaction = True
                    continue

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
                    message=(
                        "provider stream ended without assistant.end"
                    ),
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

                results = await self._dispatcher.dispatch_all(
                    calls,
                    contains_untrusted_content=(
                        untrusted_tool_output_seen
                    ),
                )
                if results:
                    untrusted_tool_output_seen = True

                calls_by_id = {
                    call.call_id: call
                    for call in calls
                }

                for result in results:
                    self._record_successful_file_edit(
                        calls_by_id.get(result.call_id),
                        result,
                    )

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

    def _build_request(
        self,
        history: Sequence[Event],
        continuation: Message | None,
    ) -> ProviderRequest:
        prefix_messages = (
            (continuation,)
            if continuation is not None
            else ()
        )

        return self._request_template.model_copy(
            update={
                "messages": messages_from_events(
                    history,
                    prefix_messages=prefix_messages,
                ),
                "tools": self._registry.specs(),
            }
        )

    def _assess_context(
        self,
        request: ProviderRequest,
    ) -> ContextAssessment | None:
        if self._model_capabilities is None:
            return None

        return assess_request_context(
            request,
            self._model_capabilities,
            self._token_counter,
            safety_margin_tokens=(
                self._context_safety_margin_tokens
            ),
        )

    def _should_compact(
        self,
        assessment: ContextAssessment,
        *,
        force_compaction: bool,
    ) -> bool:
        return (
            force_compaction
            or assessment.needs_compaction
            or assessment.remaining_input_tokens
            < self._compaction_reserve_tokens
        )

    async def _compact_context(
        self,
        *,
        original_task: str,
        previous_summary: str | None,
        recent_messages: tuple[Message, ...],
        previous_assessment: ContextAssessment,
        emit: EventFactory,
    ) -> tuple[ContextCompacted, Message, str]:
        if self._context_compactor is None:
            raise ContextCompactionError(
                "context compaction is unavailable for this model"
            )

        result = await self._context_compactor.compact(
            CompactionInput(
                original_task=original_task,
                previous_summary=previous_summary,
                ledger=self._ledger.snapshot(),
                recent_messages=recent_messages,
            ),
            emit,
        )

        # TODO state is model-generated but schema-validated.
        # File edits stay in our tool-confirmed ledger.
        self._ledger.replace_todos(result.todos)

        continuation = continuation_message(
            original_task,
            result.summary,
            self._ledger.snapshot(),
        )

        compacted_request = self._build_request(
            history=[],
            continuation=continuation,
        )
        compacted_assessment = self._assess_context(compacted_request)

        if (
            compacted_assessment is not None
            and compacted_assessment.needs_compaction
        ):
            raise ContextCompactionError(
                "the compacted continuation still exceeds the model "
                "context window"
            )

        new_input_tokens = (
            compacted_assessment.estimate.total_input_tokens
            if compacted_assessment is not None
            else 0
        )

        event = emit(
            ContextCompacted,
            summary=result.summary,
            previous_input_tokens=(
                previous_assessment.estimate.total_input_tokens
            ),
            new_input_tokens=new_input_tokens,
            discarded_message_count=len(recent_messages),
            preserved_file_edit_count=len(self._ledger.file_edits),
            todo_count=len(self._ledger.todos),
        )

        return event, continuation, result.summary

    def _record_successful_file_edit(
        self,
        call: ToolUsePart | None,
        result: ToolResultPart,
    ) -> None:
        """Record only a tool-confirmed successful write or edit."""
        if (
            call is None
            or result.is_error
            or call.name not in {"write_file", "edit_file"}
        ):
            return

        path = call.arguments.get("path")

        if not isinstance(path, str) or not path.strip():
            return

        summary = result.content.strip() or f"{call.name} completed"

        self._ledger.record_file_edit(
            FileEditRecord(
                call_id=call.call_id,
                tool_name=cast(
                    Literal["write_file", "edit_file"],
                    call.name,
                ),
                path=path,
                summary=summary,
            )
        )


def _original_task(initial_events: Sequence[Event]) -> str:
    """Use the first user request as the stable task identity."""
    for event in initial_events:
        if isinstance(event, UserMessage):
            return event.text

    return "Continue the task using the available workspace state."