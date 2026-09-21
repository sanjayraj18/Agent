"""Independent grading of an agent-modified benchmark workspace."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from agent.benchmark.images import CommandOutput
from agent.benchmark.isolation import IsolatedWorkspace
from agent.benchmark.models import (
    BenchmarkTask,
    CommandSpec,
    VerificationResult,
)


CommandRunner = Callable[[CommandSpec, Path], Awaitable[CommandOutput]]
_MAX_CAPTURED_OUTPUT = 12_000


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    passed: bool
    changed_paths: tuple[str, ...]
    verification: tuple[VerificationResult, ...]
    violation_message: str | None = None


async def evaluate_workspace(
    task: BenchmarkTask,
    isolated_workspace: IsolatedWorkspace,
    command_runner: CommandRunner,
) -> EvaluationOutcome:
    """Run task checks after enforcing its allowed-file boundary."""
    changed_paths = isolated_workspace.changed_paths()
    violation = _allowed_path_violation(task, changed_paths)

    if violation is not None:
        return EvaluationOutcome(
            passed=False,
            changed_paths=changed_paths,
            verification=(),
            violation_message=violation,
        )

    results: list[VerificationResult] = []
    for command in task.verification:
        output = await command_runner(command, isolated_workspace.workspace)
        results.append(
            VerificationResult(
                command=command,
                exit_code=output.exit_code,
                passed=not output.timed_out and output.exit_code == 0,
                stdout_tail=_tail(output.stdout),
                stderr_tail=_tail(output.stderr),
            )
        )

    return EvaluationOutcome(
        passed=bool(results) and all(result.passed for result in results),
        changed_paths=changed_paths,
        verification=tuple(results),
    )


def _allowed_path_violation(
    task: BenchmarkTask,
    changed_paths: tuple[str, ...],
) -> str | None:
    if not task.allowed_changed_paths:
        return None

    disallowed = [
        path
        for path in changed_paths
        if not any(_is_allowed(path, allowed) for allowed in task.allowed_changed_paths)
    ]

    if not disallowed:
        return None

    return "task changed files outside allowed_changed_paths: " + ", ".join(disallowed)


def _is_allowed(path: str, allowed: str) -> bool:
    return path == allowed or path.startswith(f"{allowed.rstrip('/')}/")


def _tail(value: str) -> str:
    if len(value) <= _MAX_CAPTURED_OUTPUT:
        return value
    return "[output truncated]\n" + value[-_MAX_CAPTURED_OUTPUT:]
