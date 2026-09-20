from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class DatabaseError(RuntimeError):
    """Raised when the SQLite database cannot be opened or configured."""


class SqliteDatabase:
    """
    Small SQLite connection factory for the persistence layer.

    Each operation gets its own connection. WAL mode allows one writer and
    multiple readers on the same local database file, which is a good fit for
    one agent runtime plus one or more attached TUI clients.
    """

    def __init__(
        self,
        path: Path,
        *,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must not be negative")

        resolved_path = path.expanduser().resolve()

        if resolved_path.exists() and resolved_path.is_dir():
            raise ValueError(
                "database path must be a file, not a directory: "
                f"{resolved_path}"
            )

        self._path = resolved_path
        self._busy_timeout_ms = busy_timeout_ms

    @property
    def path(self) -> Path:
        """The resolved location of the SQLite database file."""

        return self._path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """
        Open a normal read/write connection and close it afterwards.

        Use this for reads or SQLite operations that manage their own
        transaction. Use `transaction()` for one atomic write operation.
        """

        connection = self._open()

        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """
        Run one atomic write transaction.

        `BEGIN IMMEDIATE` reserves the single SQLite writer early. Instead of
        two simultaneous agent actions writing conflicting events, one waits
        up to `busy_timeout_ms`; then SQLite raises an error safely.
        """

        connection = self._open()
        began = False

        try:
            connection.execute("BEGIN IMMEDIATE")
            began = True

            yield connection

        except BaseException:
            if began:
                connection.rollback()
            raise

        else:
            connection.commit()

        finally:
            connection.close()

    def _open(self) -> sqlite3.Connection:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)

            connection = sqlite3.connect(
                self._path,
                isolation_level=None,
                timeout=self._busy_timeout_ms / 1_000,
            )
            connection.row_factory = sqlite3.Row

            # Apply these settings to every new connection.
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(
                f"PRAGMA busy_timeout = {self._busy_timeout_ms}"
            )

            return connection

        except sqlite3.Error as exc:
            raise DatabaseError(
                "could not open SQLite session database at "
                f"{self._path}: {exc}"
            ) from exc