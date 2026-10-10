from pathlib import Path

from agent.benchmark.evaluator import evaluate_workspace
from agent.benchmark.images import CommandOutput
from agent.benchmark.isolation import create_isolated_workspace
from agent.benchmark.models import (
    BenchmarkTask,
    CommandSpec,
    EvaluationViolationKind,
    MilestoneKind,
    MilestoneSpec,
    TrajectoryEntry,
)
from agent.events import ToolCallStart
from agent.providers.base import EventFactory


def _task(
    *,
    milestones: tuple[MilestoneSpec, ...] = (),
    forbidden_tool_names: tuple[str, ...] = (),
    allowed_changed_paths: tuple[str, ...] = ("implementation.py",),
) -> BenchmarkTask:
    return BenchmarkTask(
        task_id="task-one",
        title="Task one",
        prompt="Fix it",
        category="bug_fix",
        fixture="fixture",
        verification=(CommandSpec(argv=("verify",)),),
        allowed_changed_paths=allowed_changed_paths,
        milestones=milestones,
        forbidden_tool_names=forbidden_tool_names,
    )


def _isolated_workspace(tmp_path: Path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "implementation.py").write_text("old", encoding="utf-8")
    return create_isolated_workspace(
        fixture,
        run_root=tmp_path / "runs",
        run_id="run-001",
    )


def _tool_trajectory(*tool_names: str) -> tuple[TrajectoryEntry, ...]:
    emit = EventFactory("session")
    entries: list[TrajectoryEntry] = []

    for index, tool_name in enumerate(tool_names):
        event = emit(
            ToolCallStart,
            index=index,
            call_id=f"call-{index}",
            name=tool_name,
        )
        entries.append(
            TrajectoryEntry(
                sequence=event.seq,
                event_type=event.type,
                event=event,
            )
        )

    return tuple(entries)


async def _successful_command(*_args: object) -> CommandOutput:
    return CommandOutput(exit_code=0, stdout="verified", stderr="")


async def test_evaluator_records_public_evidence_for_a_passed_milestone(
    tmp_path: Path,
):
    isolated = _isolated_workspace(tmp_path)
    task = _task(
        milestones=(
            MilestoneSpec(
                milestone_id="inspect-workspace",
                description="Inspect before editing.",
                kind=MilestoneKind.TOOL_CALLED,
                tool_names=("read_file", "grep"),
            ),
        ),
    )

    try:
        outcome = await evaluate_workspace(
            task,
            isolated,
            _successful_command,
            trajectory=_tool_trajectory("read_file"),
        )
    finally:
        isolated.cleanup()

    assert outcome.passed is True
    assert outcome.evaluation.milestones[0].passed is True
    assert outcome.evaluation.milestones[0].observed_tool_name == "read_file"
    assert outcome.evaluation.milestones[0].observed_sequence == 1


async def test_evaluator_fails_a_missing_milestone_but_runs_verification(
    tmp_path: Path,
):
    isolated = _isolated_workspace(tmp_path)
    task = _task(
        milestones=(
            MilestoneSpec(
                milestone_id="inspect-workspace",
                description="Inspect before editing.",
                kind=MilestoneKind.TOOL_CALLED,
                tool_names=("read_file",),
            ),
        ),
    )

    try:
        outcome = await evaluate_workspace(
            task,
            isolated,
            _successful_command,
        )
    finally:
        isolated.cleanup()

    assert outcome.passed is False
    assert outcome.verification[0].passed is True
    assert outcome.evaluation.violations[0].kind is (
        EvaluationViolationKind.MISSING_MILESTONE
    )


async def test_evaluator_rejects_a_forbidden_tool_without_verification(
    tmp_path: Path,
):
    isolated = _isolated_workspace(tmp_path)
    task = _task(forbidden_tool_names=("bash",))

    async def should_not_run(*_args: object) -> CommandOutput:
        raise AssertionError("verification must not run after a policy violation")

    try:
        outcome = await evaluate_workspace(
            task,
            isolated,
            should_not_run,
            trajectory=_tool_trajectory("bash"),
        )
    finally:
        isolated.cleanup()

    assert outcome.passed is False
    assert outcome.verification == ()
    assert outcome.evaluation.violations[0].kind is (
        EvaluationViolationKind.FORBIDDEN_TOOL
    )
    assert outcome.evaluation.violations[0].event_sequence == 1


async def test_evaluator_rejects_changes_outside_the_task_boundary(
    tmp_path: Path,
):
    isolated = _isolated_workspace(tmp_path)
    (isolated.workspace / "tests.py").write_text("changed", encoding="utf-8")

    async def should_not_run(*_args: object) -> CommandOutput:
        raise AssertionError("verification must not run after a boundary violation")

    try:
        outcome = await evaluate_workspace(
            _task(),
            isolated,
            should_not_run,
        )
    finally:
        isolated.cleanup()

    assert outcome.passed is False
    assert outcome.verification == ()
    assert outcome.evaluation.violations[0].kind is (
        EvaluationViolationKind.FILE_BOUNDARY
    )


async def test_evaluator_records_a_failed_verification_command(tmp_path: Path):
    isolated = _isolated_workspace(tmp_path)

    async def failing_command(*_args: object) -> CommandOutput:
        return CommandOutput(exit_code=7, stdout="", stderr="test failed")

    try:
        outcome = await evaluate_workspace(
            _task(),
            isolated,
            failing_command,
        )
    finally:
        isolated.cleanup()

    assert outcome.passed is False
    assert outcome.verification[0].passed is False
    assert outcome.evaluation.violations[0].kind is (
        EvaluationViolationKind.VERIFICATION
    )
