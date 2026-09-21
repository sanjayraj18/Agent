from pathlib import Path
import sys

from agent.benchmark.images import LocalCommandExecutor
from agent.benchmark.models import BenchmarkRunConfig, BenchmarkTask, CommandSpec, RunStatus
from agent.benchmark.runner import BenchmarkRunner
from agent.benchmark.trajectory import read_trajectory
from agent.events import AssistantEnd, AssistantStart, Usage
from agent.providers.base import EventFactory


def _image() -> str:
    return "example.test/python@sha256:" + "a" * 64


async def test_runner_repeats_clean_attempts_and_preserves_trajectories(
    tmp_path: Path,
):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "answer.txt").write_text("wrong", encoding="utf-8")
    task = BenchmarkTask(
        task_id="fix-answer",
        title="Fix answer",
        prompt="Fix answer.txt",
        fixture="fixture",
        verification=[
            CommandSpec(
                argv=(
                    sys.executable,
                    "-c",
                    "from pathlib import Path; assert Path('answer.txt').read_text() == 'right'",
                )
            )
        ],
        allowed_changed_paths=("answer.txt",),
    )
    config = BenchmarkRunConfig(
        provider="openai",
        model="gpt-5.6-terra",
        agent_revision="a" * 40,
        container_image=_image(),
        attempts=3,
        parallelism=2,
    )

    async def agent_attempt(workspace: Path, _prompt: str):
        (workspace / "answer.txt").write_text("right", encoding="utf-8")
        emit = EventFactory("session")
        yield emit(AssistantStart)
        yield emit(
            AssistantEnd,
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=3),
        )

    results_root = tmp_path / "results"
    runner = BenchmarkRunner(
        fixture_root=tmp_path,
        workspace_root=results_root / "workspaces",
        results_root=results_root / "trajectories",
        agent_attempt=agent_attempt,
        command_runner=LocalCommandExecutor().run,
    )

    results = await runner.run_task(task, config)

    assert [result.status for result in results] == [RunStatus.PASSED] * 3
    assert (fixture / "answer.txt").read_text(encoding="utf-8") == "wrong"
    assert not any((results_root / "workspaces").iterdir())
    assert all(result.metrics.turns == 1 for result in results)
    assert all(
        len(read_trajectory(results_root / "trajectories", result.trajectory)) == 2
        for result in results
    )
