from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.events import Event, dumps, loads


def utc_now() -> datetime:
    """Return one timezone-aware timestamp for persisted records."""

    return datetime.now(timezone.utc)


class SessionStatus(StrEnum):
    """
    Persistent lifecycle state of a session.

    A session becomes READY after a normal agent run. It may be resumed later.
    """

    READY = "ready"
    RUNNING = "running"
    FAILED = "failed"
    ARCHIVED = "archived"


class SessionRecord(BaseModel):
    """
    The durable identity card for one agent session.

    Events themselves are stored separately in the append-only event log.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(min_length=1)
    workspace: str = Field(min_length=1)
    # Store the provider alongside the model: a resumed conversation must use
    # the same wire protocol that produced its historical tool calls/events.
    provider: str = Field(default="anthropic", min_length=1)
    model: str = Field(min_length=1)
    status: SessionStatus = SessionStatus.READY

    title: str | None = None

    # Forked sessions keep a reference to their parent and the exact event
    # sequence they started from.
    parent_session_id: str | None = None
    forked_from_event_seq: int | None = Field(
        default=None,
        ge=1,
    )

    # Lets the database detect missing or out-of-order appended events.
    last_event_seq: int = Field(default=0, ge=0)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    archived_at: datetime | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = value.strip()

        if not normalized:
            raise ValueError("title must not be blank")

        return normalized

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"anthropic", "openai"}:
            raise ValueError("provider must be anthropic or openai")
        return normalized

    @field_validator("parent_session_id")
    @classmethod
    def validate_parent_session_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        if not value.strip():
            raise ValueError("parent_session_id must not be blank")

        return value


class StoredEvent(BaseModel):
    """
    One immutable event-log row.

    `event_json` contains the original validated agent event. We also store
    session_id, sequence, and type in normal SQLite columns so sessions can be
    listed and replayed efficiently without decoding every JSON record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    event_type: str = Field(min_length=1)
    event_json: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def from_event(cls, event: Event) -> StoredEvent:
        """Convert one validated runtime event into a durable log record."""

        return cls(
            session_id=event.session_id,
            sequence=event.seq,
            event_type=event.type,
            event_json=dumps(event),
        )

    def to_event(self) -> Event:
        """
        Rebuild and verify the original event.

        The comparison catches corrupted or mismatched database rows early.
        """

        event = loads(self.event_json)

        if event.session_id != self.session_id:
            raise ValueError(
                "stored event session_id does not match event_json"
            )

        if event.seq != self.sequence:
            raise ValueError(
                "stored event sequence does not match event_json"
            )

        if event.type != self.event_type:
            raise ValueError(
                "stored event type does not match event_json"
            )

        return event
