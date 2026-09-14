from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

from agent.events import AssistantEnd, AssistantStart, ErrorEvent, Event, TextDelta, ThinkingDelta, ToolCallDelta, ToolCallEnd, ToolCallStart, ThinkingSignature, Usage
from agent.providers.base import EventFactory
from agent.providers.sse import SSEFrame


_RETRYABLE = {"overloaded_error", "rate_limit_error", "api_error", "timeout_error"}

_STOP_REASONS = {"end_turn", "tool_use", "max_tokens", "refusal", "pause_turn"}


_STOP_ALIASES = {
    "stop_sequence": "end_turn",
    "model_context_window_exceeded": "max_tokens",
}

_BLOCK_FRAMES = {"content_block_start", "content_block_delta", "content_block_stop"}


@dataclass
class _Block:
    kind: str
    call_id: str = ""
    name: str = ""
    json_parts: list[str] = field(default_factory=list)
    sig_parts: list[str] = field(default_factory=list)


class AnthropicTranslator:
    def __init__(self, emit: EventFactory) -> None:
        self._emit = emit
        self._blocks: dict[int, _Block] = {}
        self._input_tokens = 0
        self._cache_read = 0
        self._cache_create = 0
        self._output_tokens = 0
        self._stop_reason: str | None = None
        self._finished = False


    def push(self, frame: SSEFrame) -> Iterator[Event]:
        if self._finished:
            return
        try:
            payload = json.loads(frame.data)
        except json.JSONDecodeError:
            yield self._fail("malformed_frame", f"unparseable SSE data: {frame.data[:200]!r}")
            return
        yield from self.push_payload(payload)


    def push_payload(self, payload: dict[str, Any]) -> Iterator[Event]:
        if self._finished:
            return

        kind = payload.get("type")

        if kind in _BLOCK_FRAMES:
            index = payload.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                yield self._fail(
                    "malformed_frame", f"{kind} without a valid index: {payload!r}"[:300]
                )
                return
            if kind == "content_block_start":
                yield from self._on_block_start(index, payload)
            elif kind == "content_block_delta":
                yield from self._on_block_delta(index, payload)
            else:
                yield from self._on_block_stop(index)
            return

        if kind == "message_start":
            yield from self._on_message_start(payload)
        elif kind == "message_delta":
            self._on_message_delta(payload)
        elif kind == "message_stop":
            yield from self._on_message_stop()
        elif kind == "error":
            err = payload.get("error") or {}
            etype = err.get("type", "api_error")
            yield self._fail(etype, err.get("message", ""), retryable=etype in _RETRYABLE)


    def close(self) -> Iterator[Event]:
        if self._finished:
            return
        yield self._fail(
            "stream_truncated",
            "stream ended without message_stop",
            retryable=True,
        )


    def _on_message_start(self, payload: dict[str, Any]) -> Iterator[Event]:
        usage = (payload.get("message") or {}).get("usage") or {}
        self._absorb_usage(usage)
        yield self._emit(AssistantStart)


    def _on_block_start(self, index: int, payload: dict[str, Any]) -> Iterator[Event]:
        block = payload.get("content_block") or {}
        kind = block.get("type", "")
        self._blocks[index] = _Block(
            kind=kind,
            call_id=block.get("id", ""),
            name=block.get("name", ""),
        )
        if kind == "tool_use":
            yield self._emit(
                ToolCallStart,
                index=index,
                call_id=block.get("id", ""),
                name=block.get("name", ""),
            )


    def _on_block_delta(self, index: int, payload: dict[str, Any]) -> Iterator[Event]:
        delta = payload.get("delta") or {}
        dtype = delta.get("type")
        block = self._blocks.get(index)

        if dtype == "text_delta":
            yield self._emit(TextDelta, index=index, text=delta.get("text", ""))

        elif dtype == "thinking_delta":
            yield self._emit(ThinkingDelta, index=index, text=delta.get("thinking", ""))

        elif dtype == "signature_delta":
            if block is not None:
                block.sig_parts.append(delta.get("signature", ""))

        elif dtype == "input_json_delta":
            if block is None or block.kind != "tool_use":
                return
            fragment = delta.get("partial_json", "")
            block.json_parts.append(fragment)
            yield self._emit(
                ToolCallDelta,
                index=index,
                call_id=block.call_id,
                partial_json=fragment,
            )


    def _on_block_stop(self, index: int) -> Iterator[Event]:
        block = self._blocks.pop(index, None)
        if block is None:
            return

        if block.kind == "tool_use":
            raw = "".join(block.json_parts).strip()
            if not raw:
                arguments: dict[str, Any] = {}
            else:
                try:
                    arguments = json.loads(raw)
                except json.JSONDecodeError:
                    yield self._fail(
                        "tool_arguments_invalid",
                        f"tool {block.name!r} sent unparseable arguments: {raw[:200]!r}",
                    )
                    return
                if not isinstance(arguments, dict):
                    yield self._fail(
                        "tool_arguments_invalid",
                        f"tool {block.name!r} sent non-object arguments: {raw[:200]!r}",
                    )
                    return
            yield self._emit(
                ToolCallEnd, index=index, call_id=block.call_id, arguments=arguments
            )

        elif block.kind == "thinking":
            signature = "".join(block.sig_parts)
            if signature:
                yield self._emit(ThinkingSignature, index=index, signature=signature)


    def _on_message_delta(self, payload: dict[str, Any]) -> None:
        delta = payload.get("delta") or {}
        if delta.get("stop_reason") is not None:
            self._stop_reason = delta["stop_reason"]
        self._absorb_usage(payload.get("usage") or {})


    def _on_message_stop(self) -> Iterator[Event]:
        self._finished = True
        self._blocks.clear()
        reason = self._stop_reason or "end_turn"
        reason = _STOP_ALIASES.get(reason, reason)
        if reason not in _STOP_REASONS:
            reason = "end_turn"
        yield self._emit(
            AssistantEnd,
            stop_reason=reason,
            usage=Usage(
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                cache_read_input_tokens=self._cache_read,
                cache_creation_input_tokens=self._cache_create,
            ),
        )


    def _absorb_usage(self, usage: dict[str, Any]) -> None:
        for key, attr in (
            ("input_tokens", "_input_tokens"),
            ("output_tokens", "_output_tokens"),
            ("cache_read_input_tokens", "_cache_read"),
            ("cache_creation_input_tokens", "_cache_create"),
        ):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(self, attr, value)


    def _fail(self, kind: str, message: str, retryable: bool = False) -> ErrorEvent:
        self._finished = True
        self._blocks.clear()
        return self._emit(ErrorEvent, kind=kind, message=message, retryable=retryable)
