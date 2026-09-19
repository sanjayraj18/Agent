from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TodoStatus = Literal[
    "pending",
    "in_progress",
    "done",
    "blocked",
]

_VALID_TODO_STATUSES: set[str] = {
    "pending",
    "in_progress",
    "done",
    "blocked",
}


class LedgerError(ValueError):
    """Raised when task state would become ambiguous or inconsistent."""


@dataclass(frozen=True, slots=True)
class TodoItem:
    """One explicit piece of work that survives conversation compaction."""

    todo_id: str
    description: str
    status: TodoStatus = "pending"

    def __post_init__(self) -> None:
        if not self.todo_id.strip():
            raise LedgerError("todo_id must not be empty")

        if not self.description.strip():
            raise LedgerError("todo description must not be empty")

        if self.status not in _VALID_TODO_STATUSES:
            raise LedgerError(f"invalid todo status: {self.status!r}")


@dataclass(frozen=True, slots=True)
class FileEditRecord:
    """
    A durable record that a workspace-changing tool completed successfully.

    This stores a short fact about the edit, not the file contents. The
    workspace file itself remains the source of truth.
    """

    call_id: str
    tool_name: Literal["write_file", "edit_file"]
    path: str
    summary: str

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise LedgerError("file edit call_id must not be empty")

        if not self.path.strip():
            raise LedgerError("file edit path must not be empty")

        if not self.summary.strip():
            raise LedgerError("file edit summary must not be empty")


@dataclass(frozen=True, slots=True)
class TaskLedgerSnapshot:
    """An immutable ledger view supplied to the context compactor."""

    todos: tuple[TodoItem, ...]
    file_edits: tuple[FileEditRecord, ...]


@dataclass(slots=True)
class TaskLedger:

    _todos: dict[str, TodoItem] = field(default_factory=dict)
    _file_edits: list[FileEditRecord] = field(default_factory=list)

    @property
    def todos(self) -> tuple[TodoItem, ...]:
        return tuple(self._todos.values())

    @property
    def file_edits(self) -> tuple[FileEditRecord, ...]:
        return tuple(self._file_edits)

    def add_todo(self, todo: TodoItem) -> None:
        if todo.todo_id in self._todos:
            raise LedgerError(f"todo already exists: {todo.todo_id!r}")

        self._todos[todo.todo_id] = todo

    def replace_todos(self, todos: tuple[TodoItem, ...]) -> None:
        replacement = {todo.todo_id: todo for todo in todos}

        if len(replacement) != len(todos):
            raise LedgerError("todo IDs must be unique")

        self._todos = replacement

    def update_todo(
        self,
        todo_id: str,
        status: TodoStatus,
    ) -> TodoItem:
        todo = self._todos.get(todo_id)

        if todo is None:
            raise LedgerError(f"unknown todo: {todo_id!r}")

        updated = TodoItem(
            todo_id=todo.todo_id,
            description=todo.description,
            status=status,
        )
        self._todos[todo_id] = updated
        return updated

    def record_file_edit(self, edit: FileEditRecord) -> None:
        for existing in self._file_edits:
            if existing.call_id != edit.call_id:
                continue

            if existing == edit:
                return

            raise LedgerError(
                f"file edit call_id was reused: {edit.call_id!r}"
            )

        self._file_edits.append(edit)

    def snapshot(self) -> TaskLedgerSnapshot:
        return TaskLedgerSnapshot(
            todos=self.todos,
            file_edits=self.file_edits,
        )
