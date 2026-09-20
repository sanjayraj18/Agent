from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from agent.tools.base import Tool, ToolExecutionResult
from agent.tools.changes import FileChange
from agent.tools.workspace import Workspace, WorkspacePathError


class WriteFileTool(Tool):
    """Create or replace a UTF-8 text file inside the workspace."""

    name = "write_file"
    description = (
        "Create or replace a UTF-8 text file inside the workspace. "
        "Set create_parents to true only when missing parent directories should "
        "be created. Set overwrite to false to fail if the path already exists."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "overwrite": {"type": "boolean"},
            "create_parents": {"type": "boolean"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: Workspace, max_bytes: int = 1_000_000) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be at least 1")
        self._workspace = workspace
        self._max_bytes = max_bytes

    async def execute(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        unexpected = set(arguments) - {
            "path",
            "content",
            "overwrite",
            "create_parents",
        }
        if unexpected:
            return self._error_unexpected(unexpected)

        path_value = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(path_value, str):
            return ToolExecutionResult(content="path must be a string", is_error=True)
        if not isinstance(content, str):
            return ToolExecutionResult(content="content must be a string", is_error=True)

        overwrite = self._boolean(arguments.get("overwrite", True), "overwrite")
        if isinstance(overwrite, ToolExecutionResult):
            return overwrite
        create_parents = self._boolean(
            arguments.get("create_parents", False),
            "create_parents",
        )
        if isinstance(create_parents, ToolExecutionResult):
            return create_parents

        try:
            data = content.encode("utf-8")
        except UnicodeEncodeError:
            return ToolExecutionResult(
                content="content must be valid UTF-8 text",
                is_error=True,
            )

        if len(data) > self._max_bytes:
            return ToolExecutionResult(
                content=f"content exceeds the {self._max_bytes}-byte write limit",
                is_error=True,
            )

        try:
            path = self._workspace.resolve(path_value)
        except WorkspacePathError as exc:
            return ToolExecutionResult(content=str(exc), is_error=True)

        existing_text = self._existing_text(
            path,
            path_value,
            overwrite=overwrite,
        )
        if isinstance(existing_text, ToolExecutionResult):
            return existing_text

        before, operation = existing_text

        try:
            if not path.parent.exists():
                if not create_parents:
                    return ToolExecutionResult(
                        content=f"parent directory does not exist: "
                        f"{self._workspace.relative(path.parent)}",
                        is_error=True,
                    )
                path.parent.mkdir(parents=True, exist_ok=True)

            if not path.parent.is_dir():
                return ToolExecutionResult(
                    content="parent path is not a directory",
                    is_error=True,
                )

            self._atomic_write(path, data)
        except OSError as exc:
            return ToolExecutionResult(
                content=f"could not write file: {exc}",
                is_error=True,
            )

        relative_path = self._workspace.relative(path)
        return ToolExecutionResult(
            content=f"wrote {len(data)} bytes to {relative_path}",
            file_change=FileChange(
                path=relative_path,
                before=before,
                after=content,
                operation=operation,
            ),
        )

    def _existing_text(
        self,
        path: Path,
        path_value: str,
        *,
        overwrite: bool,
    ) -> tuple[str, Literal["created", "updated"]] | ToolExecutionResult:
        """Read the old text before replacement so a truthful diff exists."""

        if not path.exists():
            return "", "created"

        if not path.is_file():
            return ToolExecutionResult(
                content=f"path is not a regular file: {path_value}",
                is_error=True,
            )

        if not overwrite:
            return ToolExecutionResult(
                content=f"file already exists: {path_value}",
                is_error=True,
            )

        try:
            if path.stat().st_size > self._max_bytes:
                return ToolExecutionResult(
                    content=(
                        "existing file exceeds the "
                        f"{self._max_bytes}-byte diff limit"
                    ),
                    is_error=True,
                )
            raw = path.read_bytes()
        except OSError as exc:
            return ToolExecutionResult(
                content=f"could not read existing file: {exc}",
                is_error=True,
            )

        if b"\x00" in raw:
            return ToolExecutionResult(
                content=(
                    "existing file is binary; refusing to overwrite it "
                    "without a text diff"
                ),
                is_error=True,
            )

        try:
            return raw.decode("utf-8"), "updated"
        except UnicodeDecodeError:
            return ToolExecutionResult(
                content=(
                    "existing file is not valid UTF-8 text; refusing to "
                    "overwrite it without a text diff"
                ),
                is_error=True,
            )

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".agent-write-",
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
    def _boolean(value: object, name: str) -> bool | ToolExecutionResult:
        if not isinstance(value, bool):
            return ToolExecutionResult(
                content=f"{name} must be a boolean",
                is_error=True,
            )
        return value

    @staticmethod
    def _error_unexpected(arguments: set[str]) -> ToolExecutionResult:
        return ToolExecutionResult(
            content=f"unexpected arguments: {', '.join(sorted(arguments))}",
            is_error=True,
        )
