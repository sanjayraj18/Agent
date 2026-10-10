"""Independent grading of an agent-modified benchmark workspace.

    Final workspace evidence:
    - Which files changed?
    - Did verification commands pass?

    Trajectory evidence:
    - Which tools did the agent call?
    - Did it satisfy each milestone?
    - Did it call a forbidden tool?
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from agent.benchmark.images import CommandOutput
from agent.benchmark.isolation import IsolatedWorkspace
from agent.benchmark.models import (
    BenchmarkTask,
    CommandSpec,
    EvaluationResult,
    EvaluationViolation,
    EvaluationViolationKind,
    MilestoneResult,
    TrajectoryEntry,
    VerificationResult,
)
from agent.events import ToolCallStart


CommandRunner = Callable[[CommandSpec, Path], Awaitable[CommandOutput]]
_MAX_CAPTURED_OUTPUT = 12_000


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """
    Compatibility wrapper around the full deterministic evaluation receipt.

    The runner currently uses these properties. In the next file, it will
    persist `evaluation` directly into BenchmarkRunResult.
    """

    evaluation: EvaluationResult

    @property
    def passed(self) -> bool:
        return self.evaluation.passed

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return self.evaluation.changed_paths

    @property
    def verification(self) -> tuple[VerificationResult, ...]:
        return self.evaluation.verification

    @property
    def violation_message(self) -> str | None:
        if not self.evaluation.violations:
            return None

        return self.evaluation.violations[0].message


async def evaluate_workspace(
    task: BenchmarkTask,
    isolated_workspace: IsolatedWorkspace,
    command_runner: CommandRunner,
    trajectory: tuple[TrajectoryEntry, ...] = (),
) -> EvaluationOutcome:
    """
    Grade a completed attempt using deterministic public evidence.

    Policy violations skip final verification because the attempt has already
    failed its safety contract. Missing milestones still allow verification to
    run, so the report can show both behavioral and final-state evidence.
    """
    changed_paths = isolated_workspace.changed_paths()
    tool_calls = _tool_calls_from_trajectory(trajectory)

    milestones, milestone_violations = _evaluate_milestones(
        task,
        tool_calls,
    )

    policy_violations: list[EvaluationViolation] = []

    changed_path_violation = _allowed_path_violation(task, changed_paths)
    if changed_path_violation is not None:
        policy_violations.append(
            EvaluationViolation(
                kind=EvaluationViolationKind.FILE_BOUNDARY,
                message=changed_path_violation,
            )
        )

    policy_violations.extend(
        _forbidden_tool_violations(
            task,
            tool_calls,
        )
    )

    violations = [
        *policy_violations,
        *milestone_violations,
    ]

    # A boundary breach or forbidden tool use is a hard policy failure.
    # Do not spend more time running final verification after one occurs.
    if policy_violations:
        return EvaluationOutcome(
            evaluation=EvaluationResult(
                passed=False,
                changed_paths=changed_paths,
                milestones=milestones,
                violations=tuple(violations),
            )
        )

    verification = await _run_verification(
        task,
        isolated_workspace.workspace,
        command_runner,
    )

    for result in verification:
        if not result.passed:
            violations.append(
                EvaluationViolation(
                    kind=EvaluationViolationKind.VERIFICATION,
                    message=_verification_failure_message(result),
                )
            )

    return EvaluationOutcome(
        evaluation=EvaluationResult(
            passed=not violations,
            changed_paths=changed_paths,
            milestones=milestones,
            violations=tuple(violations),
            verification=verification,
        )
    )


def _tool_calls_from_trajectory(
    trajectory: tuple[TrajectoryEntry, ...],
) -> tuple[ToolCallStart, ...]:
    """Extract only the public tool-call events needed for Phase 2."""
    return tuple(
        entry.event
        for entry in trajectory
        if isinstance(entry.event, ToolCallStart)
    )


def _evaluate_milestones(
    task: BenchmarkTask,
    tool_calls: tuple[ToolCallStart, ...],
) -> tuple[
    tuple[MilestoneResult, ...],
    tuple[EvaluationViolation, ...],
]:
    results: list[MilestoneResult] = []
    violations: list[EvaluationViolation] = []

    for milestone in task.milestones:
        matching_call = next(
            (
                call
                for call in tool_calls
                if call.name in milestone.tool_names
            ),
            None,
        )

        if matching_call is not None:
            results.append(
                MilestoneResult(
                    milestone_id=milestone.milestone_id,
                    kind=milestone.kind,
                    passed=True,
                    observed_tool_name=matching_call.name,
                    observed_sequence=matching_call.seq,
                )
            )
            continue

        results.append(
            MilestoneResult(
                milestone_id=milestone.milestone_id,
                kind=milestone.kind,
                passed=False,
            )
        )
        violations.append(
            EvaluationViolation(
                kind=EvaluationViolationKind.MISSING_MILESTONE,
                message=(
                    f"required milestone {milestone.milestone_id!r} "
                    "was not observed"
                ),
            )
        )

    return tuple(results), tuple(violations)


def _forbidden_tool_violations(
    task: BenchmarkTask,
    tool_calls: tuple[ToolCallStart, ...],
) -> tuple[EvaluationViolation, ...]:
    forbidden = set(task.forbidden_tool_names)

    if not forbidden:
        return ()

    return tuple(
        EvaluationViolation(
            kind=EvaluationViolationKind.FORBIDDEN_TOOL,
            message=f"task called forbidden tool {call.name!r}",
            event_sequence=call.seq,
        )
        for call in tool_calls
        if call.name in forbidden
    )


async def _run_verification(
    task: BenchmarkTask,
    workspace: Path,
    command_runner: CommandRunner,
) -> tuple[VerificationResult, ...]:
    results: list[VerificationResult] = []

    for command in task.verification:
        output = await command_runner(command, workspace)
        results.append(
            VerificationResult(
                command=command,
                exit_code=output.exit_code,
                passed=not output.timed_out and output.exit_code == 0,
                stdout_tail=_tail(output.stdout),
                stderr_tail=_tail(output.stderr),
            )
        )

    return tuple(results)


def _allowed_path_violation(
    task: BenchmarkTask,
    changed_paths: tuple[str, ...],
) -> str | None:
    if not task.allowed_changed_paths:
        return None

    disallowed = [
        path
        for path in changed_paths
        if not any(
            _is_allowed(path, allowed)
            for allowed in task.allowed_changed_paths
        )
    ]

    if not disallowed:
        return None

    return (
        "task changed files outside allowed_changed_paths: "
        + ", ".join(disallowed)
    )


def _verification_failure_message(result: VerificationResult) -> str:
    command = " ".join(result.command.argv)
    return (
        "verification command failed: "
        f"{command!r} exited with code {result.exit_code}"
    )


def _is_allowed(path: str, allowed: str) -> bool:
    return path == allowed or path.startswith(f"{allowed.rstrip('/')}/")


def _tail(value: str) -> str:
    if len(value) <= _MAX_CAPTURED_OUTPUT:
        return value

    return "[output truncated]\n" + value[-_MAX_CAPTURED_OUTPUT:]