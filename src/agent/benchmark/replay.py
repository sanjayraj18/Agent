"""Safe, integrity-checked timeline views of saved benchmark trajectories."""

from __future__ import annotations

from pathlib import Path

from agent.benchmark.models import (
    BenchmarkRunResult,
    ReplayEvent,
    TrajectoryEntry,
    TrajectoryReplay,
)
from agent.benchmark.trajectory import read_trajectory
from agent.events import (
    AssistantEnd,
    AssistantStart,
    ContextCompacted,
    ErrorEvent,
    SessionStarted,
    TextDelta,
    ThinkingDelta,
    ThinkingSignature,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResult,
    UserMessage,
)


def replay_trajectory(
    result: BenchmarkRunResult,
    *,
    trajectories_root: Path,
) -> TrajectoryReplay:
    """
    Read a recorded attempt and produce a privacy-preserving timeline.

    ``read_trajectory`` verifies the stored SHA-256 digest and event count
    before this function summarizes anything.  The original JSONL remains the
    immutable source of truth; this replay intentionally hides private
    thinking and all raw text/tool payloads.
    """
    entries = read_trajectory(trajectories_root, result.trajectory)
    public_events: list[ReplayEvent] = []
    suppressed_private_event_count = 0

    for entry in entries:
        if isinstance(entry.event, (ThinkingDelta, ThinkingSignature)):
            suppressed_private_event_count += 1
            continue

        public_events.append(_summarize_entry(entry))

    return TrajectoryReplay(
        reference=result.trajectory,
        events=tuple(public_events),
        suppressed_private_event_count=suppressed_private_event_count,
    )


def _summarize_entry(entry: TrajectoryEntry) -> ReplayEvent:
    """Describe public event structure without copying its raw payload."""
    event = entry.event

    if isinstance(event, SessionStarted):
        summary = f"Session started for model `{event.model}`."
    elif isinstance(event, UserMessage):
        summary = f"User prompt received ({len(event.text)} characters)."
    elif isinstance(event, AssistantStart):
        summary = "Assistant turn started."
    elif isinstance(event, TextDelta):
        summary = f"Assistant emitted text ({len(event.text)} characters)."
    elif isinstance(event, ToolCallStart):
        summary = f"Tool `{event.name}` call started."
    elif isinstance(event, ToolCallDelta):
        summary = "Tool call arguments streamed."
    elif isinstance(event, ToolCallEnd):
        summary = "Tool call arguments completed."
    elif isinstance(event, ToolResult):
        outcome = "an error" if event.is_error else "success"
        summary = (
            f"Tool returned {outcome} ({len(event.content)} characters)."
        )
    elif isinstance(event, AssistantEnd):
        summary = (
            f"Assistant turn ended: `{event.stop_reason}` "
            f"(input {event.usage.input_tokens}, "
            f"output {event.usage.output_tokens} tokens)."
        )
    elif isinstance(event, ContextCompacted):
        summary = (
            "Context compacted from "
            f"{event.previous_input_tokens} to {event.new_input_tokens} "
            "input tokens."
        )
    elif isinstance(event, ErrorEvent):
        summary = f"Agent reported error kind `{event.kind}`."
    else:  # pragma: no cover - Event is a closed discriminated union.
        raise TypeError(f"unsupported trajectory event {type(event)!r}")

    return ReplayEvent(
        sequence=entry.sequence,
        event_type=entry.event_type,
        summary=summary,
    )
