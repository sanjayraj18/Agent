"""Load benchmark tasks from version-controlled JSON definitions."""

from __future__ import annotations

import json
from pathlib import Path

from agent.benchmark.models import BenchmarkTask


class BenchmarkCatalogError(ValueError):
    """A benchmark task definition is missing or invalid."""


class BenchmarkCatalog:
    """The read-only catalog of reproducible benchmark tasks."""

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()
        self._tasks_directory = self._root / "tasks"

    @property
    def root(self) -> Path:
        return self._root

    def list_tasks(self) -> tuple[BenchmarkTask, ...]:
        if not self._tasks_directory.is_dir():
            raise BenchmarkCatalogError(
                f"benchmark task directory does not exist: {self._tasks_directory}"
            )

        return tuple(
            self._load_path(path)
            for path in sorted(self._tasks_directory.glob("*.json"))
        )

    def load_task(self, task_id: str) -> BenchmarkTask:
        if not task_id or task_id != task_id.strip():
            raise BenchmarkCatalogError("task_id must be a non-empty string")

        path = self._tasks_directory / f"{task_id}.json"
        task = self._load_path(path)

        if task.task_id != task_id:
            raise BenchmarkCatalogError(
                f"{path}: task_id {task.task_id!r} does not match its filename"
            )

        return task

    def fixture_path(self, task: BenchmarkTask) -> Path:
        candidate = (self._root / task.fixture).resolve()

        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            raise BenchmarkCatalogError(
                f"task {task.task_id!r} fixture escapes the catalog"
            ) from exc

        if not candidate.is_dir():
            raise BenchmarkCatalogError(
                f"task {task.task_id!r} fixture is not a directory: {candidate}"
            )

        return candidate

    def _load_path(self, path: Path) -> BenchmarkTask:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise BenchmarkCatalogError(f"benchmark task not found: {path}") from exc
        except OSError as exc:
            raise BenchmarkCatalogError(f"could not read benchmark task {path}: {exc}") from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BenchmarkCatalogError(f"{path}: invalid JSON ({exc})") from exc

        if not isinstance(decoded, dict):
            raise BenchmarkCatalogError(f"{path}: task definition must be an object")

        try:
            return BenchmarkTask.model_validate(decoded)
        except Exception as exc:
            raise BenchmarkCatalogError(f"{path}: invalid task definition ({exc})") from exc
