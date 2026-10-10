from decimal import Decimal
from pathlib import Path
import sys

from agent.benchmark.images import LocalCommandExecutor
from agent.benchmark.models import (
    BenchmarkRunConfig,
    BenchmarkTask,
    CommandSpec,
    MilestoneKind,
    MilestoneSpec,
    RunStatus,
)
from agent.benchmark.runner import BenchmarkRunner, _metrics
from agent.benchmark.trajectory import read_trajectory
from agent.events import AssistantEnd, AssistantStart, ToolCallStart, Usage
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
        category="bug_fix",
        fixture="fixture",
        verification=(
            CommandSpec(
                argv=(
                    sys.executable,
                    "-c",
                    "from pathlib import Path; assert Path('answer.txt').read_text() == 'right'",
                )
            ),
        ),
        allowed_changed_paths=("answer.txt",),
        milestones=(
            MilestoneSpec(
                milestone_id="inspect-workspace",
                description="Inspect the workspace before editing.",
                kind=MilestoneKind.TOOL_CALLED,
                tool_names=("read_file",),
            ),
        ),
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
            ToolCallStart,
            index=0,
            call_id="read-answer",
            name="read_file",
        )
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
        result.metrics.executed_turns[0].model == "gpt-5.6-terra"
        for result in results
    )
    assert all(
        len(read_trajectory(results_root / "trajectories", result.trajectory)) == 3
        for result in results
    )
    assert all(result.evaluation is not None for result in results)
    assert all(result.evaluation.passed for result in results if result.evaluation)
    assert all(
        result.evaluation.milestones[0].observed_tool_name == "read_file"
        for result in results
        if result.evaluation is not None
    )


def test_metrics_prices_the_actual_live_routed_model_not_the_starting_model():
    emit = EventFactory("session")
    assistant_end = emit(
        AssistantEnd,
        stop_reason="end_turn",
        usage=Usage(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        ),
        executed_provider="openai",
        executed_model="gpt-5.6-luna",
        executed_route_id="economy-luna-medium",
    )

    metrics = _metrics(
        [assistant_end],
        duration_seconds=Decimal("1"),
        provider="openai",
        configured_model="gpt-5.6-terra",
        configured_route_id="strong-terra-high",
    )

    assert metrics.executed_turns[0].model == "gpt-5.6-luna"
    assert metrics.executed_turns[0].route_id == "economy-luna-medium"
    assert metrics.cost_usd == Decimal("1.4")


def test_metrics_uses_the_fixed_configuration_when_legacy_events_lack_metadata():
    emit = EventFactory("session")
    assistant_end = emit(
        AssistantEnd,
        stop_reason="end_turn",
        usage=Usage(input_tokens=1_000_000, output_tokens=1_000_000),
    )

    metrics = _metrics(
        [assistant_end],
        duration_seconds=Decimal("1"),
        provider="openai",
        configured_model="gpt-5.6-terra",
        configured_route_id="strong-terra-high",
    )

    assert metrics.executed_turns[0].model == "gpt-5.6-terra"
    assert metrics.executed_turns[0].route_id == "strong-terra-high"
    assert metrics.cost_usd == Decimal("14")


def test_metrics_marks_a_run_cost_unknown_when_an_executed_model_has_no_price():
    emit = EventFactory("session")
    assistant_end = emit(
        AssistantEnd,
        stop_reason="end_turn",
        usage=Usage(input_tokens=100),
        executed_provider="openai",
        executed_model="not-in-our-pricing-catalog",
        executed_route_id="experimental",
    )

    metrics = _metrics(
        [assistant_end],
        duration_seconds=Decimal("1"),
        provider="openai",
        configured_model="gpt-5.6-terra",
        configured_route_id="strong-terra-high",
    )

    assert metrics.executed_turns[0].cost_usd is None
    assert metrics.cost_usd is None
