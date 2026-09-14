from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from datetime import datetime, timezone


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EventBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int
    session_id: str
    ts: datetime = Field(default_factory=_now)


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class SessionStarted(EventBase):
    type: Literal["session.started"] = "session.started"
    cwd: str
    model: str


class UserMessage(EventBase):
    type: Literal["user.message"] = "user.message"
    text: str


class AssistantStart(EventBase):
    type: Literal["assistant.start"] = "assistant.start"


class TextDelta(EventBase):
    type: Literal["assistant.text_delta"] = "assistant.text_delta"
    index: int
    text: str


class ThinkingDelta(EventBase):
    type: Literal["assistant.thinking_delta"] = "assistant.thinking_delta"
    index: int
    text: str


StopReason = Literal[
    "end_turn", "tool_use", "max_tokens", "refusal", "pause_turn", "interrupted"
]

class AssistantEnd(EventBase):
    type: Literal["assistant.end"] = "assistant.end"
    stop_reason: StopReason
    usage: Usage

class ToolCallStart(EventBase):
    type: Literal["tool.call_start"] = "tool.call_start"
    index: int
    call_id: str
    name: str


class ToolCallDelta(EventBase):
    type: Literal["tool.call_delta"] = "tool.call_delta"
    index: int
    call_id: str
    partial_json: str


class ToolCallEnd(EventBase):
    type: Literal["tool.call_end"] = "tool.call_end"
    index: int
    call_id: str
    arguments: dict[str, object]


class ToolResult(EventBase):
    type: Literal["tool.result"] = "tool.result"
    call_id: str
    content: str
    is_error: bool = False


class ErrorEvent(EventBase):
      type: Literal["error"] = "error"
      kind: str
      message: str
      retryable: bool = False
      retry_after: float | None = None


class ThinkingSignature(EventBase):
    type: Literal["assistant.thinking_signature"] = "assistant.thinking_signature"
    index: int
    signature: str


Event = Annotated[
    Union[
        SessionStarted,
        UserMessage,
        AssistantStart,
        TextDelta,
        ThinkingSignature,
        ThinkingDelta,
        AssistantEnd,
        ToolCallStart,
        ToolCallDelta,
        ToolCallEnd,
        ToolResult,
        ErrorEvent,
    ],
    Field(discriminator="type"),
]

EventAdapter: TypeAdapter[Event] = TypeAdapter(Event) # type: ignore

def dumps(event: Event) -> str:
    return EventAdapter.dump_json(event).decode()


def loads(raw: str | bytes) -> Event:
    return EventAdapter.validate_json(raw)