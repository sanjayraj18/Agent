from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from agent.events import Event
from agent.persistence.database import SqliteDatabase
from agent.persistence.event_log import EventLog, SessionNotFoundError
from agent.persistence.models import (
    SessionRecord,
    SessionStatus,
    StoredEvent,
    utc_now,
)


class SessionStoreError(RuntimeError):
    """Base error for session-record operations."""


class SessionAlreadyExistsError(SessionStoreError):
    """Raised when a caller tries to create a duplicate session ID."""


class SessionForkError(SessionStoreError):
    """Raised when a requested session fork is invalid."""


@dataclass(frozen=True, slots=True)
class LoadedSession:
    """
    One resumable session reconstructed from SQLite.

    `events` is the exact ordered history that AgentLoop needs when resuming.
    """

    session: SessionRecord
    events: tuple[Event, ...]


class SessionStore:
    """
    Manage durable session records and their append-only event histories.

    Session metadata lives in `sessions`.
    Conversation/tool history lives in `session_events`.
    """

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database
        self._event_log = EventLog(database)

    def create(
        self,
        *,
        workspace: Path | str,
        provider: str = "anthropic",
        model: str,
        title: str | None = None,
        session_id: str | None = None,
    ) -> SessionRecord:
        """Create one empty session ready for its first user message."""

        normalized_workspace = self._workspace_string(workspace)

        if not model.strip():
            raise ValueError("model must not be blank")

        record = SessionRecord(
            session_id=session_id or uuid4().hex,
            workspace=normalized_workspace,
            provider=provider,
            model=model,
            title=title,
        )

        with self._database.transaction() as connection:
            self._insert_session(connection, record)

        return record

    def get(self, session_id: str) -> SessionRecord:
        """Load one session record without replaying its events."""

        self._validate_session_id(session_id)

        with self._database.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    session_id,
                    workspace,
                    provider,
                    model,
                    status,
                    title,
                    parent_session_id,
                    forked_from_event_seq,
                    last_event_seq,
                    created_at,
                    updated_at,
                    archived_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            raise SessionNotFoundError(
                f"session does not exist: {session_id}"
            )

        return SessionRecord.model_validate(dict(row))

    def list(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
    ) -> tuple[SessionRecord, ...]:
        """List recent sessions for the TUI session picker."""

        if limit < 1:
            raise ValueError("limit must be at least 1")

        where_clause = ""

        if not include_archived:
            where_clause = "WHERE status != ?"
            parameters: tuple[object, ...] = (
                SessionStatus.ARCHIVED.value,
                limit,
            )
        else:
            parameters = (limit,)

        with self._database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    session_id,
                    workspace,
                    provider,
                    model,
                    status,
                    title,
                    parent_session_id,
                    forked_from_event_seq,
                    last_event_seq,
                    created_at,
                    updated_at,
                    archived_at
                FROM sessions
                {where_clause}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        return tuple(
            SessionRecord.model_validate(dict(row))
            for row in rows
        )

    def load(self, session_id: str) -> LoadedSession:
        """Load session metadata plus its complete validated event history."""

        session = self.get(session_id)
        events = self._event_log.replay(session_id)

        return LoadedSession(
            session=session,
            events=events,
        )

    def archive(self, session_id: str) -> SessionRecord:
        """Hide a session from the normal picker without deleting history."""

        return self._update_status(
            session_id,
            status=SessionStatus.ARCHIVED,
            archived_at=utc_now().isoformat(),
        )

    def unarchive(self, session_id: str) -> SessionRecord:
        """Return an archived session to the normal ready state."""

        return self._update_status(
            session_id,
            status=SessionStatus.READY,
            archived_at=None,
        )

    def set_runtime_status(
        self,
        session_id: str,
        status: Literal[
            SessionStatus.READY,
            SessionStatus.RUNNING,
            SessionStatus.FAILED,
        ],
    ) -> SessionRecord:
        """
        Update the latest live-run status.

        Archiving is deliberately separate so a runtime cannot accidentally
        hide a session from the user.
        """

        if status == SessionStatus.ARCHIVED:
            raise ValueError(
                "use archive() instead of set_runtime_status()"
            )

        return self._update_status(
            session_id,
            status=status,
            archived_at=None,
        )

    def rename(
        self,
        session_id: str,
        title: str | None,
    ) -> SessionRecord:
        """Set or clear the human-friendly title shown in the session picker."""

        self._validate_session_id(session_id)

        if title is not None:
            title = title.strip()

            if not title:
                raise ValueError("title must not be blank")

        with self._database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions
                SET
                    title = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (
                    title,
                    utc_now().isoformat(),
                    session_id,
                ),
            )

            if cursor.rowcount != 1:
                raise SessionNotFoundError(
                    f"session does not exist: {session_id}"
                )

        return self.get(session_id)

    def fork(
        self,
        session_id: str,
        *,
        at_sequence: int | None = None,
        title: str | None = None,
        new_session_id: str | None = None,
    ) -> SessionRecord:
        """
        Create an independent branch from a historical point.

        We copy the parent events through `at_sequence`, changing their
        session_id to the child ID. The new child can then continue with its
        own event sequence and never mutates the parent session.
        """

        parent = self.get(session_id)
        fork_sequence = (
            parent.last_event_seq
            if at_sequence is None
            else at_sequence
        )

        if fork_sequence < 1:
            raise SessionForkError(
                "cannot fork a session without at least one event"
            )

        if fork_sequence > parent.last_event_seq:
            raise SessionForkError(
                "fork sequence exceeds the parent session history"
            )

        child_id = new_session_id or uuid4().hex

        if child_id == parent.session_id:
            raise SessionForkError(
                "forked session must have a new session_id"
            )

        child = SessionRecord(
            session_id=child_id,
            workspace=parent.workspace,
            provider=parent.provider,
            model=parent.model,
            status=SessionStatus.READY,
            title=title,
            parent_session_id=parent.session_id,
            forked_from_event_seq=fork_sequence,
            last_event_seq=fork_sequence,
        )

        with self._database.transaction() as connection:
            source_rows = connection.execute(
                """
                SELECT
                    session_id,
                    sequence,
                    event_type,
                    event_json,
                    created_at
                FROM session_events
                WHERE
                    session_id = ?
                    AND sequence <= ?
                ORDER BY sequence ASC
                """,
                (parent.session_id, fork_sequence),
            ).fetchall()

            if len(source_rows) != fork_sequence:
                raise SessionForkError(
                    "parent event history is incomplete"
                )

            self._insert_session(connection, child)

            for row in source_rows:
                parent_event = StoredEvent.model_validate(dict(row))
                child_event = parent_event.to_event().model_copy(
                    update={"session_id": child_id}
                )
                copied_event = StoredEvent.from_event(
                    child_event
                ).model_copy(
                    update={"created_at": parent_event.created_at}
                )

                connection.execute(
                    """
                    INSERT INTO session_events (
                        session_id,
                        sequence,
                        event_type,
                        event_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        copied_event.session_id,
                        copied_event.sequence,
                        copied_event.event_type,
                        copied_event.event_json,
                        copied_event.created_at.isoformat(),
                    ),
                )

        return child

    def _update_status(
        self,
        session_id: str,
        *,
        status: SessionStatus,
        archived_at: str | None,
    ) -> SessionRecord:
        self._validate_session_id(session_id)

        with self._database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions
                SET
                    status = ?,
                    archived_at = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (
                    status.value,
                    archived_at,
                    utc_now().isoformat(),
                    session_id,
                ),
            )

            if cursor.rowcount != 1:
                raise SessionNotFoundError(
                    f"session does not exist: {session_id}"
                )

        return self.get(session_id)

    @staticmethod
    def _insert_session(
        connection: sqlite3.Connection,
        record: SessionRecord,
    ) -> None:
        try:
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id,
                    workspace,
                    provider,
                    model,
                    status,
                    title,
                    parent_session_id,
                    forked_from_event_seq,
                    last_event_seq,
                    created_at,
                    updated_at,
                    archived_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.workspace,
                    record.provider,
                    record.model,
                    record.status.value,
                    record.title,
                    record.parent_session_id,
                    record.forked_from_event_seq,
                    record.last_event_seq,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    (
                        record.archived_at.isoformat()
                        if record.archived_at is not None
                        else None
                    ),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise SessionAlreadyExistsError(
                f"session already exists: {record.session_id}"
            ) from exc

    @staticmethod
    def _workspace_string(workspace: Path | str) -> str:
        path = Path(workspace).expanduser().resolve()

        if not path.is_dir():
            raise ValueError(
                f"workspace must be an existing directory: {path}"
            )

        return str(path)

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be blank")
