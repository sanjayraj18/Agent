"""Load and validate benchmark tasks from version-controlled JSON definitions."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from agent.benchmark.models import BenchmarkTask


TASK_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


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
        """Load every task and validate its complete on-disk contract."""
        if not self._tasks_directory.is_dir():
            raise BenchmarkCatalogError(
                "benchmark task directory does not exist: "
                f"{self._tasks_directory}"
            )

        tasks: list[BenchmarkTask] = []

        for path in sorted(self._tasks_directory.glob("*.json")):
            task = self._load_path(path)
            self._validate_task_path(path, task)
            self.fixture_path(task)
            tasks.append(task)

        return tuple(tasks)

    def load_task(self, task_id: str) -> BenchmarkTask:
        """
        Load one task by ID.

        The task ID is validated before becoming part of a filesystem path.
        This prevents values such as '../somewhere-else' from being used.
        """
        if (
            not isinstance(task_id, str)
            or not TASK_ID_PATTERN.fullmatch(task_id)
        ):
            raise BenchmarkCatalogError(
                "task_id must contain only lowercase letters, numbers, "
                "hyphens, and underscores"
            )

        path = self._tasks_directory / f"{task_id}.json"
        task = self._load_path(path)

        self._validate_task_path(path, task)
        self.fixture_path(task)

        return task

    def fixture_path(self, task: BenchmarkTask) -> Path:
        """Return the validated fixture directory for one task."""
        candidate = (self._root / task.fixture).resolve()

        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            raise BenchmarkCatalogError(
                f"task {task.task_id!r} fixture escapes the catalog"
            ) from exc

        if not candidate.is_dir():
            raise BenchmarkCatalogError(
                f"task {task.task_id!r} fixture is not a directory: "
                f"{candidate}"
            )

        return candidate

    def _validate_task_path(
        self,
        path: Path,
        task: BenchmarkTask,
    ) -> None:
        """Ensure the JSON filename agrees with the task's declared ID."""
        if task.task_id != path.stem:
            raise BenchmarkCatalogError(
                f"{path}: task_id {task.task_id!r} does not match "
                f"its filename {path.stem!r}"
            )

    def _load_path(self, path: Path) -> BenchmarkTask:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise BenchmarkCatalogError(
                f"benchmark task not found: {path}"
            ) from exc
        except OSError as exc:
            raise BenchmarkCatalogError(
                f"could not read benchmark task {path}: {exc}"
            ) from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BenchmarkCatalogError(
                f"{path}: invalid JSON ({exc})"
            ) from exc

        if not isinstance(decoded, dict):
            raise BenchmarkCatalogError(
                f"{path}: task definition must be a JSON object"
            )

        try:
            return BenchmarkTask.model_validate(decoded)
        except ValidationError as exc:
            raise BenchmarkCatalogError(
                f"{path}: invalid task definition ({exc})"
            ) from exc