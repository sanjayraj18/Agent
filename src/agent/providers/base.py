from __future__ import annotations

import json
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Literal,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field

from agent.core.prompt_cache import PromptCachePlan, PromptSection
from agent.events import Event, EventBase


class ProviderMetadata(BaseModel):
    """Provider identity kept separate from model-specific capabilities."""

    model_config = ConfigDict(frozen=True)

    provider_id: str = Field(min_length=1)
    base_url: str = Field(min_length=1)


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
    # Tool output can contain file contents, command output, or remote data.
    # Treat it as data by default when it is sent back to an LLM.
    is_untrusted: bool = True
    # This data crosses dispatcher -> agent loop only. ``exclude=True`` means
    # a provider cannot accidentally receive the TUI's before/after diff.
    file_change: dict[str, str] | None = Field(
        default=None,
        exclude=True,
    )


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

    stable_context: str | None = None
    cache_stable_prefix: bool = False

    @property
    def system_prompt(self) -> str | None:
        """Return the stable system-level content ready for a provider."""
        parts = [
            part
            for part in (self.system, self.stable_context)
            if part
        ]
        return "\n\n".join(parts) or None

    @property
    def cache_plan(self) -> PromptCachePlan | None:
        """Build a logical plan used to validate and fingerprint the prefix."""
        if not self.cache_stable_prefix:
            return None

        sections: list[PromptSection] = []

        if self.system:
            sections.append(
                PromptSection(kind="system", content=self.system)
            )

        if self.tools:
            sections.append(
                PromptSection(
                    kind="tools",
                    content=_canonical_json(
                        [
                            tool.model_dump(mode="json")
                            for tool in self.tools
                        ]
                    ),
                )
            )

        if self.stable_context:
            sections.append(
                PromptSection(
                    kind="stable_context",
                    content=self.stable_context,
                )
            )

        sections.append(
            PromptSection(
                kind="dynamic",
                content=_canonical_json(
                    [
                        message.model_dump(mode="json")
                        for message in self.messages
                    ]
                ),
            )
        )

        return PromptCachePlan(sections=tuple(sections))


def _canonical_json(value: object) -> str:
    """Serialize data deterministically without changing meaningful list order."""
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


E = TypeVar("E", bound=EventBase)


class EventFactory:
    __slots__ = ("_session_id", "_seq")

    def __init__(self, session_id: str, start_seq: int = 0) -> None:
        self._session_id = session_id
        self._seq = start_seq

    def __call__(self, event_type: type[E], **fields: Any) -> E:
        self._seq += 1
        return event_type(
            seq=self._seq,
            session_id=self._session_id,
            **fields,
        )

    @property
    def seq(self) -> int:
        return self._seq


@runtime_checkable
class Provider(Protocol):
    """Anything that can turn a ProviderRequest into a stream of events."""

    name: str

    def stream(
        self,
        request: ProviderRequest,
        emit: EventFactory,
    ) -> AsyncIterator[Event]:
        """
        Yield events until the turn ends.

        Must terminate with exactly one AssistantEnd or one ErrorEvent.
        Must never raise for a protocol-level failure.
        """
        ...
