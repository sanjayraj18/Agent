from __future__ import annotations

import asyncio
from typing import Any

from agent.tools.base import Tool, ToolExecutionResult
from agent.tools.processes import (
    ProcessRegistry,
    ProcessSnapshot,
    read_limited_stream,
)
from agent.tools.workspace import Workspace


class BashTool(Tool):
    """Run bounded shell commands in the workspace."""

    name = "bash"
    description = (
        "Run a shell command with the workspace as its working directory. "
        "Use action='run' to execute a command, optionally with background=true. "
        "Use action='status' or action='stop' with a job_id to inspect or stop "
        "a background command. Foreground commands have a timeout and bounded output."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["run", "status", "stop"]},
            "command": {"type": "string"},
            "background": {"type": "boolean"},
            "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 600},
            "job_id": {"type": "string"},
        },
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace: Workspace,
        registry: ProcessRegistry | None = None,
        max_output_bytes: int = 100_000,
        max_timeout_seconds: float = 600,
    ) -> None:
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be at least 1")
        if max_timeout_seconds <= 0:
            raise ValueError("max_timeout_seconds must be greater than zero")
        self._workspace = workspace
        self._registry = registry or ProcessRegistry(max_output_bytes)
        self._max_output_bytes = max_output_bytes
        self._max_timeout_seconds = max_timeout_seconds

    async def execute(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        unexpected = set(arguments) - {
            "action",
            "command",
            "background",
            "timeout_seconds",
            "job_id",
        }
        if unexpected:
            return ToolExecutionResult(
                content=f"unexpected arguments: {', '.join(sorted(unexpected))}",
                is_error=True,
            )

        action = arguments.get("action", "run")
        if action not in {"run", "status", "stop"}:
            return ToolExecutionResult(
                content="action must be one of: run, status, stop",
                is_error=True,
            )

        if action == "run":
            return await self._run(arguments)
        return await self._manage(action, arguments)

    async def _run(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolExecutionResult(
                content="command must be a non-empty string",
                is_error=True,
            )
        if "\x00" in command:
            return ToolExecutionResult(
                content="command must not contain a null byte",
                is_error=True,
            )

        background = arguments.get("background", False)
        if not isinstance(background, bool):
            return ToolExecutionResult(
                content="background must be a boolean",
                is_error=True,
            )

        timeout = self._timeout(arguments.get("timeout_seconds", 30.0))
        if isinstance(timeout, ToolExecutionResult):
            return timeout

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self._workspace.root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            return ToolExecutionResult(
                content=f"could not start command: {exc}",
                is_error=True,
            )

        if background:
            snapshot = await self._registry.register(command, process)
            return ToolExecutionResult(
                content=f"started background job {snapshot.job_id} (pid {snapshot.pid})"
            )

        assert process.stdout is not None
        reader_task = asyncio.create_task(
            read_limited_stream(process.stdout, self._max_output_bytes)
        )
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            process.kill()
            await process.wait()
        output, truncated = await reader_task
        rendered = self._render_output(output, truncated)

        if timed_out:
            return ToolExecutionResult(
                content=f"command timed out after {timeout:g} seconds\n{rendered}",
                is_error=True,
            )
        if process.returncode != 0:
            return ToolExecutionResult(
                content=f"command exited with status {process.returncode}\n{rendered}",
                is_error=True,
            )
        return ToolExecutionResult(content=rendered)

    async def _manage(
        self,
        action: str,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        job_id = arguments.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            return ToolExecutionResult(
                content="job_id must be a non-empty string",
                is_error=True,
            )

        if action == "stop":
            snapshot = await self._registry.stop(job_id)
        else:
            snapshot = self._registry.snapshot(job_id)

        if snapshot is None:
            return ToolExecutionResult(
                content=f"unknown background job: {job_id}",
                is_error=True,
            )
        return ToolExecutionResult(content=self._render_snapshot(snapshot))

    def _timeout(self, value: object) -> float | ToolExecutionResult:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return ToolExecutionResult(
                content="timeout_seconds must be a number",
                is_error=True,
            )
        timeout = float(value)
        if timeout <= 0 or timeout > self._max_timeout_seconds:
            return ToolExecutionResult(
                content=f"timeout_seconds must be between 0 and "
                f"{self._max_timeout_seconds:g}",
                is_error=True,
            )
        return timeout

    @staticmethod
    def _render_output(output: bytes, truncated: bool) -> str:
        text = output.decode("utf-8", errors="replace").rstrip("\n")
        if not text:
            text = "[no output]"
        if truncated:
            text += "\n[output truncated]"
        return text

    @classmethod
    def _render_snapshot(cls, snapshot: ProcessSnapshot) -> str:
        details = (
            f"job {snapshot.job_id}: {snapshot.status} "
            f"(pid {snapshot.pid}, exit={snapshot.return_code})"
        )
        return (
            f"{details}\n"
            f"{cls._render_output(snapshot.output.encode(), snapshot.output_truncated)}"
        )
