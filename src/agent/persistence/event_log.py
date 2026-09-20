from __future__ import annotations

from collections.abc import Iterable

from agent.events import Event
from agent.persistence.database import SqliteDatabase
from agent.persistence.models import StoredEvent, utc_now


class EventLogError(RuntimeError):
    """Base error for durable agent-event storage."""


class SessionNotFoundError(EventLogError):
    """Raised when an event is written to a nonexistent session."""


class EventSequenceError(EventLogError):
    """Raised when events are missing, duplicated, or out of order."""


class EventLog:
    """
    Append and replay validated agent events.

    The database is the durable source of truth. We never modify an old
    event-log row; correcting behavior always produces a later event.
    """

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    def append(self, event: Event) -> StoredEvent:
        """Append exactly one event in the next expected sequence position."""

        return self.append_many([event])[0]

    def append_many(
        self,
        events: Iterable[Event],
    ) -> tuple[StoredEvent, ...]:
        """
        Atomically append a consecutive event batch for one session.

        If even one event has the wrong sequence number, none of the batch is
        written. This prevents partially saved tool/action histories.
        """

        stored_events = tuple(
            StoredEvent.from_event(event)
            for event in events
        )

        if not stored_events:
            return ()

        session_id = stored_events[0].session_id

        if any(
            event.session_id != session_id
            for event in stored_events
        ):
            raise EventSequenceError(
                "one append batch must contain events from one session"
            )

        with self._database.transaction() as connection:
            row = connection.execute(
                """
                SELECT last_event_seq
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

            if row is None:
                raise SessionNotFoundError(
                    f"session does not exist: {session_id}"
                )

            expected_sequence = int(row["last_event_seq"]) + 1

            for event in stored_events:
                if event.sequence != expected_sequence:
                    raise EventSequenceError(
                        "expected event sequence "
                        f"{expected_sequence} for session {session_id}, "
                        f"got {event.sequence}"
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
                        event.session_id,
                        event.sequence,
                        event.event_type,
                        event.event_json,
                        event.created_at.isoformat(),
                    ),
                )

                expected_sequence += 1

            last_event = stored_events[-1]

            connection.execute(
                """
                UPDATE sessions
                SET
                    last_event_seq = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (
                    last_event.sequence,
                    utc_now().isoformat(),
                    session_id,
                ),
            )

        return stored_events

    def read(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 1_000,
    ) -> tuple[StoredEvent, ...]:
        """
        Return stored events in their original order.

        `after_sequence=0` replays from the beginning. A client that already
        received event 42 asks for events after 42 when reconnecting.
        """

        if not session_id.strip():
            raise ValueError("session_id must not be blank")

        if after_sequence < 0:
            raise ValueError(
                "after_sequence must not be negative"
            )

        if limit < 1:
            raise ValueError("limit must be at least 1")

        with self._database.connection() as connection:
            self._require_session(connection, session_id)

            rows = connection.execute(
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
                    AND sequence > ?
                ORDER BY sequence ASC
                LIMIT ?
                """,
                (
                    session_id,
                    after_sequence,
                    limit,
                ),
            ).fetchall()

        return tuple(
            StoredEvent.model_validate(dict(row))
            for row in rows
        )

    def replay(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
    ) -> tuple[Event, ...]:
        """Rebuild validated runtime events from the durable log."""

        return tuple(
            stored_event.to_event()
            for stored_event in self.read(
                session_id,
                after_sequence=after_sequence,
            )
        )

    def last_sequence(self, session_id: str) -> int:
        """Return the last durable event sequence for one session."""

        if not session_id.strip():
            raise ValueError("session_id must not be blank")

        with self._database.connection() as connection:
            row = connection.execute(
                """
                SELECT last_event_seq
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            raise SessionNotFoundError(
                f"session does not exist: {session_id}"
            )

        return int(row["last_event_seq"])

    @staticmethod
    def _require_session(
        connection,
        session_id: str,
    ) -> None:
        row = connection.execute(
            """
            SELECT 1
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

        if row is None:
            raise SessionNotFoundError(
                f"session does not exist: {session_id}"
            )