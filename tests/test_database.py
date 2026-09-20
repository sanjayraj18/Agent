from __future__ import annotations

import sqlite3

import pytest

from agent.persistence.database import SqliteDatabase


def test_transaction_commits_and_creates_the_database_parent(tmp_path):
    database = SqliteDatabase(tmp_path / "state" / "sessions.sqlite3")

    with database.transaction() as connection:
        connection.execute("CREATE TABLE sample (value TEXT NOT NULL)")
        connection.execute("INSERT INTO sample (value) VALUES ('saved')")

    with database.connection() as connection:
        row = connection.execute("SELECT value FROM sample").fetchone()

    assert database.path.exists()
    assert row is not None
    assert row["value"] == "saved"


def test_transaction_rolls_back_everything_on_error(tmp_path):
    database = SqliteDatabase(tmp_path / "sessions.sqlite3")

    with database.transaction() as connection:
        connection.execute("CREATE TABLE sample (value TEXT NOT NULL)")

    with pytest.raises(sqlite3.IntegrityError):
        with database.transaction() as connection:
            connection.execute("INSERT INTO sample (value) VALUES ('kept')")
            connection.execute("INSERT INTO sample (value) VALUES (NULL)")

    with database.connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM sample"
        ).fetchone()

    assert count is not None
    assert count["count"] == 0


def test_rejects_a_directory_as_the_database_path(tmp_path):
    with pytest.raises(ValueError, match="must be a file"):
        SqliteDatabase(tmp_path)
