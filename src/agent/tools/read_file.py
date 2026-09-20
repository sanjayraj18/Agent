from __future__ import annotations

from typing import Any

from agent.security import SecretScanner
from agent.tools.base import Tool, ToolExecutionResult
from agent.tools.workspace import Workspace, WorkspacePathError


class ReadFileTool(Tool):
    """Read a UTF-8 text file inside one workspace."""

    name = "read_file"
    description = (
        "Read a UTF-8 text file from the workspace. "
        "Use start_line and end_line to read a specific line range. "
        "The result includes line numbers. "
        "Potential secrets are redacted before content is returned. "
        "This tool cannot read files outside the workspace."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path to the file.",
            },
            "start_line": {
                "type": "integer",
                "minimum": 1,
                "description": "First line to read, starting at 1.",
            },
            "end_line": {
                "type": "integer",
                "minimum": 1,
                "description": "Last line to read, inclusive.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace: Workspace,
        max_bytes: int = 100_000,
        max_lines: int = 2_000,
        secret_scanner: SecretScanner | None = None,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be at least 1")

        if max_lines < 1:
            raise ValueError("max_lines must be at least 1")

        self._workspace = workspace
        self._max_bytes = max_bytes
        self._max_lines = max_lines
        self._secret_scanner = secret_scanner or SecretScanner()

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        unexpected_arguments = set(arguments) - {
            "path",
            "start_line",
            "end_line",
        }
        if unexpected_arguments:
            names = ", ".join(sorted(unexpected_arguments))
            return ToolExecutionResult(
                content=f"unexpected arguments: {names}",
                is_error=True,
            )

        path_value = arguments.get("path")
        if not isinstance(path_value, str):
            return ToolExecutionResult(
                content="path must be a string",
                is_error=True,
            )

        start_line = self._line_number(
            arguments.get("start_line", 1),
            "start_line",
        )
        if isinstance(start_line, ToolExecutionResult):
            return start_line

        end_value = arguments.get("end_line")
        end_line: int | None = None

        if end_value is not None:
            parsed_end_line = self._line_number(
                end_value,
                "end_line",
            )
            if isinstance(parsed_end_line, ToolExecutionResult):
                return parsed_end_line
            end_line = parsed_end_line

        if end_line is not None and end_line < start_line:
            return ToolExecutionResult(
                content="end_line must not be less than start_line",
                is_error=True,
            )

        try:
            path = self._workspace.resolve(path_value)
        except WorkspacePathError as exc:
            return ToolExecutionResult(
                content=str(exc),
                is_error=True,
            )

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
            file_size = path.stat().st_size

            with path.open("rb") as file:
                raw = file.read(self._max_bytes)
        except OSError as exc:
            return ToolExecutionResult(
                content=f"could not read file: {exc}",
                is_error=True,
            )

        if b"\x00" in raw:
            return ToolExecutionResult(
                content="binary files cannot be read as text",
                is_error=True,
            )

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return ToolExecutionResult(
                content="file is not valid UTF-8 text",
                is_error=True,
            )

        lines = text.splitlines()

        if start_line > len(lines) and lines:
            return ToolExecutionResult(
                content=(
                    f"start_line {start_line} is beyond the end of the "
                    f"readable file ({len(lines)} lines)"
                ),
                is_error=True,
            )

        requested_end = end_line or (
            start_line + self._max_lines - 1
        )
        limited_end = min(
            requested_end,
            start_line + self._max_lines - 1,
            len(lines),
        )

        selected_lines = lines[start_line - 1 : limited_end]
        selected_text = "\n".join(selected_lines)
        scan = self._secret_scanner.scan(selected_text)

        safe_lines = (
            scan.redacted_text.split("\n")
            if selected_lines
            else []
        )

        rendered_lines = [
            f"{line_number:>6}: {line}"
            for line_number, line in enumerate(
                safe_lines,
                start=start_line,
            )
        ]

        notes: list[str] = []

        if scan.was_redacted:
            noun = (
                "secret"
                if scan.redaction_count == 1
                else "secrets"
            )
            notes.append(
                f"{scan.redaction_count} potential {noun} redacted"
            )

        if limited_end < len(lines) and (
            end_line is None or end_line > limited_end
        ):
            notes.append(
                f"line output truncated at line {limited_end}"
            )

        if file_size > self._max_bytes:
            notes.append(
                f"file output truncated after {self._max_bytes} bytes"
            )

        content = "\n".join(rendered_lines)

        if notes:
            suffix = "\n".join(f"[{note}]" for note in notes)
            content = f"{content}\n{suffix}" if content else suffix

        return ToolExecutionResult(content=content)

    @staticmethod
    def _line_number(
        value: object,
        name: str,
    ) -> int | ToolExecutionResult:
        if not isinstance(value, int) or isinstance(value, bool):
            return ToolExecutionResult(
                content=f"{name} must be an integer",
                is_error=True,
            )

        if value < 1:
            return ToolExecutionResult(
                content=f"{name} must be at least 1",
                is_error=True,
            )

        return value