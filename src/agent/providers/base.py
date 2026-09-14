


from typing import Annotated, Any, AsyncIterator, Literal, Protocol, TypeVar, Union, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent.events import Event, EventBase


class TextPart(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["text"] = "text"
    text: str


class ThinkingPart(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["thinking"] = "thinking"
    text: str
    signature: str | None = None


class ToolUsePart(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["tool_use"] = "tool_use"
    call_id: str
    name: str
    arguments: dict[str, Any]


class ToolResultPart(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    content: str
    is_error: bool = False

ContentPart = Annotated[
    Union[TextPart, ThinkingPart, ToolUsePart, ToolResultPart],
    Field(discriminator="type"),
]

class Message(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: Literal["user", "assistant"]
    content: list[ContentPart]


class ToolSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    description: str
    input_schema: dict[str, Any]

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ProviderRequest(BaseModel):
    """One turn's worth of input, in provider-neutral form."""

    model_config = ConfigDict(frozen=True)

    model: str
    messages: list[Message]
    max_tokens: int
    system: str | None = None
    tools: list[ToolSpec] = Field(default_factory=list)
    effort: Effort | None = None
    thinking: bool = True
    # Placeholder. Phase 5 owns real breakpoint placement; for now this means
    # "cache the stable prefix (tools + system)", which is the dominant pattern.
    cache_stable_prefix: bool = False

E = TypeVar("E", bound=EventBase)


class EventFactory:

    __slots__ = ("_session_id", "_seq")

    def __init__(self, session_id : str, start_seq : int = 0) -> None:
        self._session_id = session_id
        self._seq = start_seq

    def __call__(self, event_type: type[E], **fields: Any) -> E:
        self._seq += 1
        return event_type(seq=self._seq, session_id=self._session_id, **fields)

    @property
    def seq(self) -> int:
        return self._seq


@runtime_checkable
class Provider(Protocol):
    """Anything that can turn a ProviderRequest into a stream of Events."""

    name: str

    def stream(
        self,
        request: ProviderRequest,
        emit: EventFactory,
    ) -> AsyncIterator[Event]:
        """Yield events until the turn ends.

        Must terminate with exactly one AssistantEnd or one ErrorEvent.
        Must never raise for a protocol-level failure — failures are events.
        """
        ...