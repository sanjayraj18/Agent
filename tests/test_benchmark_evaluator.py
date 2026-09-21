from pathlib import Path

from agent.benchmark.evaluator import evaluate_workspace
from agent.benchmark.images import CommandOutput
from agent.benchmark.isolation import create_isolated_workspace
from agent.benchmark.models import BenchmarkTask, CommandSpec


async def test_evaluator_rejects_changes_outside_the_task_boundary(tmp_path: Path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "implementation.py").write_text("old", encoding="utf-8")
    isolated = create_isolated_workspace(
        fixture,
        run_root=tmp_path / "runs",
        run_id="run-001",
    )
    (isolated.workspace / "tests.py").write_text("changed", encoding="utf-8")
    task = BenchmarkTask(
        task_id="task-one",
        title="Task one",
        prompt="Fix it",
        fixture="fixture",
        verification=[CommandSpec(argv=("unused",))],
        allowed_changed_paths=("implementation.py",),
    )

    async def should_not_run(*_args: object) -> CommandOutput:
        raise AssertionError("verification must not run after a boundary violation")

    outcome = await evaluate_workspace(task, isolated, should_not_run)

    assert outcome.passed is False
    assert outcome.violation_message is not None
    isolated.cleanup()
