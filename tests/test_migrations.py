from __future__ import annotations

import pytest

from agent.persistence.database import SqliteDatabase
from agent.persistence.migrations import (
    LATEST_SCHEMA_VERSION,
    MigrationError,
    migrate,
)


def test_first_migration_creates_the_session_schema(tmp_path):
    database = SqliteDatabase(tmp_path / "sessions.sqlite3")

    result = migrate(database)

    assert result.previous_version == 0
    assert result.current_version == LATEST_SCHEMA_VERSION
    assert result.applied_versions == (1,)

    with database.connection() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {"schema_migrations", "sessions", "session_events"} <= tables


def test_migration_is_idempotent_for_an_current_database(tmp_path):
    database = SqliteDatabase(tmp_path / "sessions.sqlite3")
    migrate(database)

    result = migrate(database)

    assert result.previous_version == LATEST_SCHEMA_VERSION
    assert result.applied_versions == ()


def test_refuses_a_database_created_by_a_newer_agent(tmp_path):
    database = SqliteDatabase(tmp_path / "sessions.sqlite3")
    migrate(database)

    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO schema_migrations (version, applied_at) "
            "VALUES (?, ?)",
            (LATEST_SCHEMA_VERSION + 1, "2030-01-01T00:00:00+00:00"),
        )

    with pytest.raises(MigrationError, match="newer"):
        migrate(database)
