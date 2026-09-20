from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolExecutionResult
from agent.tools.changes import FileChange
from agent.tools.workspace import Workspace, WorkspacePathError


class EditFileTool(Tool):
    """Apply one exact text replacement to a UTF-8 workspace file."""

    name = "edit_file"
    description = (
        "Replace exactly one occurrence of old_text with new_text in a UTF-8 "
        "workspace file. This tool fails if old_text is missing or occurs more "
        "than once, so it never silently edits an ambiguous location."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: Workspace, max_bytes: int = 1_000_000) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be at least 1")
        self._workspace = workspace
        self._max_bytes = max_bytes

    async def execute(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        unexpected = set(arguments) - {"path", "old_text", "new_text"}
        if unexpected:
            return ToolExecutionResult(
                content=f"unexpected arguments: {', '.join(sorted(unexpected))}",
                is_error=True,
            )

        path_value = arguments.get("path")
        old_text = arguments.get("old_text")
        new_text = arguments.get("new_text")
        if not isinstance(path_value, str):
            return ToolExecutionResult(content="path must be a string", is_error=True)
        if not isinstance(old_text, str):
            return ToolExecutionResult(content="old_text must be a string", is_error=True)
        if not isinstance(new_text, str):
            return ToolExecutionResult(content="new_text must be a string", is_error=True)
        if not old_text:
            return ToolExecutionResult(content="old_text must not be empty", is_error=True)

        try:
            path = self._workspace.resolve(path_value)
        except WorkspacePathError as exc:
            return ToolExecutionResult(content=str(exc), is_error=True)

        if not path.exists():
            return ToolExecutionResult(
                content=f"file does not exist: {path_value}",
                is_error=True,
            )
        if not path.is_file():
            return ToolExecutionResult(
                content=f"path is not a regular file: {path_value}",
                is_error=True,
            )

        try:
            if path.stat().st_size > self._max_bytes:
                return ToolExecutionResult(
                    content=f"file exceeds the {self._max_bytes}-byte edit limit",
                    is_error=True,
                )
            raw = path.read_bytes()
        except OSError as exc:
            return ToolExecutionResult(
                content=f"could not read file: {exc}",
                is_error=True,
            )

        if b"\x00" in raw:
            return ToolExecutionResult(
                content="binary files cannot be edited as text",
                is_error=True,
            )

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return ToolExecutionResult(
                content="file is not valid UTF-8 text",
                is_error=True,
            )

        occurrences = self._occurrence_count(text, old_text)
        if occurrences == 0:
            return ToolExecutionResult(
                content="old_text was not found in the file",
                is_error=True,
            )
        if occurrences > 1:
            return ToolExecutionResult(
                content=f"old_text occurs {occurrences} times; provide a unique match",
                is_error=True,
            )

        updated = text.replace(old_text, new_text, 1)
        try:
            encoded = updated.encode("utf-8")
        except UnicodeEncodeError:
            return ToolExecutionResult(
                content="new_text must be valid UTF-8 text",
                is_error=True,
            )
        if len(encoded) > self._max_bytes:
            return ToolExecutionResult(
                content=f"edited file exceeds the {self._max_bytes}-byte limit",
                is_error=True,
            )

        try:
            self._atomic_write(path, encoded)
        except OSError as exc:
            return ToolExecutionResult(
                content=f"could not write file: {exc}",
                is_error=True,
            )

        relative_path = self._workspace.relative(path)
        return ToolExecutionResult(
            content=f"edited {relative_path} (replaced 1 occurrence)",
            file_change=FileChange(
                path=relative_path,
                before=text,
                after=updated,
                operation="updated",
            ),
        )

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".agent-edit-",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(data)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _occurrence_count(text: str, needle: str) -> int:
        """Count matches including overlapping ones to avoid ambiguous edits."""
        count = 0
        start = 0
        while (index := text.find(needle, start)) != -1:
            count += 1
            start = index + 1
        return count
