from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolExecutionResult
from agent.tools.workspace import Workspace, WorkspacePathError


class GrepTool(Tool):
    """Search UTF-8 workspace files with a regular expression."""

    name = "grep"
    description = (
        "Search UTF-8 text files in the workspace with a regular expression. "
        "Returns matching lines as path:line_number: text. Binary files and "
        ".git, .venv, node_modules, and __pycache__ directories are skipped."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression to search for.",
            },
            "path": {
                "type": "string",
                "description": "Workspace-relative file or directory to search.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "description": "Maximum matching lines to return.",
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    _IGNORED_DIRECTORIES = {
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
    }

    def __init__(
        self,
        workspace: Workspace,
        max_results: int = 200,
        max_file_bytes: int = 1_000_000,
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        if max_file_bytes < 1:
            raise ValueError("max_file_bytes must be at least 1")
        self._workspace = workspace
        self._max_results = max_results
        self._max_file_bytes = max_file_bytes

    async def execute(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        unexpected = set(arguments) - {"pattern", "path", "max_results"}
        if unexpected:
            return ToolExecutionResult(
                content=f"unexpected arguments: {', '.join(sorted(unexpected))}",
                is_error=True,
            )

        pattern_value = arguments.get("pattern")
        if not isinstance(pattern_value, str) or not pattern_value:
            return ToolExecutionResult(
                content="pattern must be a non-empty string",
                is_error=True,
            )
        if "\x00" in pattern_value:
            return ToolExecutionResult(
                content="pattern must not contain a null byte",
                is_error=True,
            )
        try:
            pattern = re.compile(pattern_value)
        except re.error as exc:
            return ToolExecutionResult(
                content=f"invalid regular expression: {exc}",
                is_error=True,
            )

        path_value = arguments.get("path", ".")
        if not isinstance(path_value, str):
            return ToolExecutionResult(content="path must be a string", is_error=True)
        try:
            target = self._workspace.resolve(path_value)
        except WorkspacePathError as exc:
            return ToolExecutionResult(content=str(exc), is_error=True)
        if not target.exists():
            return ToolExecutionResult(
                content=f"path does not exist: {path_value}",
                is_error=True,
            )
        if not target.is_file() and not target.is_dir():
            return ToolExecutionResult(
                content=f"path is neither a file nor a directory: {path_value}",
                is_error=True,
            )

        if self._is_ignored(Path(self._workspace.relative(target))):
            return ToolExecutionResult(content="no matches")

        limit = self._result_limit(arguments.get("max_results", self._max_results))
        if isinstance(limit, ToolExecutionResult):
            return limit

        matches: list[str] = []
        truncated = False
        try:
            for file_path in self._files_in(target):
                relative_path = self._workspace.relative(file_path)
                for line_number, line in self._matching_lines(file_path, pattern):
                    if len(matches) >= limit:
                        truncated = True
                        break
                    matches.append(
                        f"{relative_path}:{line_number}: {line.rstrip(chr(10) + chr(13))}"
                    )
                if truncated:
                    break
        except OSError as exc:
            return ToolExecutionResult(
                content=f"could not search files: {exc}",
                is_error=True,
            )

        if not matches:
            return ToolExecutionResult(content="no matches")

        content = "\n".join(matches)
        if truncated:
            content += f"\n[results truncated at {limit} matches]"
        return ToolExecutionResult(content=content)

    def _result_limit(self, value: object) -> int | ToolExecutionResult:
        if not isinstance(value, int) or isinstance(value, bool):
            return ToolExecutionResult(
                content="max_results must be an integer",
                is_error=True,
            )
        if value < 1 or value > self._max_results:
            return ToolExecutionResult(
                content=(
                    f"max_results must be between 1 and {self._max_results}"
                ),
                is_error=True,
            )
        return value

    def _files_in(self, target: Path) -> Iterator[Path]:
        if target.is_file():
            yield target
            return

        for root, directory_names, file_names in os.walk(target, followlinks=False):
            root_path = Path(root)
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name not in self._IGNORED_DIRECTORIES
            )
            for name in sorted(file_names):
                candidate = root_path / name
                try:
                    resolved = self._workspace.resolve(
                        candidate.relative_to(self._workspace.root).as_posix()
                    )
                except WorkspacePathError:
                    continue
                if resolved.is_file():
                    yield resolved

    def _matching_lines(
        self,
        path: Path,
        pattern: re.Pattern[str],
    ) -> Iterator[tuple[int, str]]:
        try:
            if path.stat().st_size > self._max_file_bytes:
                return
            content = path.read_bytes()
        except OSError:
            return

        if b"\x00" in content:
            return
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return

        for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
            if pattern.search(line):
                yield line_number, line

    @classmethod
    def _is_ignored(cls, path: Path) -> bool:
        return any(part in cls._IGNORED_DIRECTORIES for part in path.parts)
