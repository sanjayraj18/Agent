from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolExecutionResult
from agent.tools.workspace import Workspace, WorkspacePathError


class GlobTool(Tool):
    """Find workspace files and directories using a glob pattern."""

    name = "glob"
    description = (
        "Find files and directories inside the workspace using a glob pattern. "
        "Examples: 'src/**/*.py', 'tests/test_*.py', or '*.toml'. "
        "Results are workspace-relative paths. "
        "The tool ignores .git, .venv, node_modules, and __pycache__."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "A workspace-relative glob pattern.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "description": "Maximum number of paths to return.",
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
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be at least 1")

        self._workspace = workspace
        self._max_results = max_results

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        unexpected_arguments = set(arguments) - {
            "pattern",
            "max_results",
        }
        if unexpected_arguments:
            names = ", ".join(sorted(unexpected_arguments))
            return ToolExecutionResult(
                content=f"unexpected arguments: {names}",
                is_error=True,
            )

        pattern = arguments.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return ToolExecutionResult(
                content="pattern must be a non-empty string",
                is_error=True,
            )

        if "\x00" in pattern:
            return ToolExecutionResult(
                content="pattern must not contain a null byte",
                is_error=True,
            )

        pattern_path = Path(pattern)

        if pattern_path.is_absolute():
            return ToolExecutionResult(
                content="absolute patterns are not allowed",
                is_error=True,
            )

        if ".." in pattern_path.parts:
            return ToolExecutionResult(
                content="pattern must not contain '..'",
                is_error=True,
            )

        result_limit = arguments.get(
            "max_results",
            self._max_results,
        )

        if (
            not isinstance(result_limit, int)
            or isinstance(result_limit, bool)
        ):
            return ToolExecutionResult(
                content="max_results must be an integer",
                is_error=True,
            )

        if result_limit < 1 or result_limit > self._max_results:
            return ToolExecutionResult(
                content=(
                    f"max_results must be between 1 and "
                    f"{self._max_results}"
                ),
                is_error=True,
            )

        matches: list[str] = []
        truncated = False

        try:
            for match in self._workspace.root.glob(pattern):
                relative_match = match.relative_to(self._workspace.root)

                if self._is_ignored(relative_match):
                    continue

                try:
                    self._workspace.resolve(relative_match.as_posix())
                except WorkspacePathError:
                    # Do not expose symlinks that lead outside the workspace.
                    continue

                if len(matches) >= result_limit:
                    truncated = True
                    break

                display_path = relative_match.as_posix()

                if match.is_dir():
                    display_path += "/"

                matches.append(display_path)

        except (OSError, ValueError) as exc:
            return ToolExecutionResult(
                content=f"could not evaluate glob pattern: {exc}",
                is_error=True,
            )

        matches.sort()

        if not matches:
            return ToolExecutionResult(content="no matches")

        content = "\n".join(matches)

        if truncated:
            content += (
                f"\n[results truncated at {result_limit} paths]"
            )

        return ToolExecutionResult(content=content)

    @classmethod
    def _is_ignored(cls, path: Path) -> bool:
        return any(
            part in cls._IGNORED_DIRECTORIES
            for part in path.parts
        )