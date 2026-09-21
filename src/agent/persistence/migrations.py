from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable

from agent.persistence.database import SqliteDatabase
from agent.persistence.models import SessionStatus, utc_now


class MigrationError(RuntimeError):
    """Raised when the database schema cannot be migrated safely."""


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """Describes what `migrate()` changed."""

    previous_version: int
    current_version: int
    applied_versions: tuple[int, ...]


Migration = Callable[[sqlite3.Connection], None]

LATEST_SCHEMA_VERSION = 2

_SESSION_STATUS_VALUES = ", ".join(
    f"'{status.value}'"
    for status in SessionStatus
)


def migrate(database: SqliteDatabase) -> MigrationResult:
    """
    Upgrade the session database to the latest known schema.

    Migrations run in one atomic transaction. Either every required table and
    index is created, or SQLite rolls the whole operation back.
    """

    with database.transaction() as connection:
        _create_migration_table(connection)

        previous_version = _current_version(connection)

        if previous_version > LATEST_SCHEMA_VERSION:
            raise MigrationError(
                "session database is newer than this agent version: "
                f"{previous_version} > {LATEST_SCHEMA_VERSION}"
            )

        applied_versions: list[int] = []

        for version in range(
            previous_version + 1,
            LATEST_SCHEMA_VERSION + 1,
        ):
            migration = _MIGRATIONS.get(version)

            if migration is None:
                raise MigrationError(
                    f"missing migration for schema version {version}"
                )

            migration(connection)
            _record_migration(connection, version)
            applied_versions.append(version)

    return MigrationResult(
        previous_version=previous_version,
        current_version=LATEST_SCHEMA_VERSION,
        applied_versions=tuple(applied_versions),
    )


def _create_migration_table(
    connection: sqlite3.Connection,
) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY CHECK(version > 0),
            applied_at TEXT NOT NULL
        )
        """
    )


def _current_version(
    connection: sqlite3.Connection,
) -> int:
    row = connection.execute(
        """
        SELECT COALESCE(MAX(version), 0) AS version
        FROM schema_migrations
        """
    ).fetchone()

    assert row is not None
    return int(row["version"])


def _record_migration(
    connection: sqlite3.Connection,
    version: int,
) -> None:
    connection.execute(
        """
        INSERT INTO schema_migrations (
            version,
            applied_at
        )
        VALUES (?, ?)
        """,
        (version, utc_now().isoformat()),
    )


def _migration_1_create_session_tables(
    connection: sqlite3.Connection,
) -> None:
    """
    Create the first append-only session/event-log schema.

    Events are never updated in place. A later phase can add snapshots for
    speed, but the event log remains the durable source of truth.
    """

    connection.execute(
        f"""
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            workspace TEXT NOT NULL,
            model TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK(status IN ({_SESSION_STATUS_VALUES})),
            title TEXT,
            parent_session_id TEXT
                REFERENCES sessions(session_id)
                ON DELETE RESTRICT,
            forked_from_event_seq INTEGER
                CHECK(
                    forked_from_event_seq IS NULL
                    OR forked_from_event_seq > 0
                ),
            last_event_seq INTEGER NOT NULL DEFAULT 0
                CHECK(last_event_seq >= 0),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            archived_at TEXT
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE session_events (
            session_id TEXT NOT NULL
                REFERENCES sessions(session_id)
                ON DELETE CASCADE,
            sequence INTEGER NOT NULL CHECK(sequence > 0),
            event_type TEXT NOT NULL,
            event_json TEXT NOT NULL,
            created_at TEXT NOT NULL,

            PRIMARY KEY (session_id, sequence)
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_sessions_updated_at
        ON sessions(updated_at DESC)
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_sessions_parent
        ON sessions(parent_session_id)
        """
    )


def _migration_2_add_session_provider(
    connection: sqlite3.Connection,
) -> None:
    """Give existing Anthropic-only sessions an explicit provider identity."""

    connection.execute(
        """
        ALTER TABLE sessions
        ADD COLUMN provider TEXT NOT NULL DEFAULT 'anthropic'
        CHECK(provider IN ('anthropic', 'openai'))
        """
    )

_MIGRATIONS: dict[int, Migration] = {
    1: _migration_1_create_session_tables,
    2: _migration_2_add_session_provider,
}
