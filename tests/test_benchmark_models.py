from decimal import Decimal

import pytest

from agent.benchmark.models import (
    BenchmarkRunConfig,
    BenchmarkTask,
    CommandSpec,
    ExecutedTurn,
    MilestoneKind,
    MilestoneSpec,
    RunMetrics,
    ScoreboardRow,
    TaskCategory,
    TrajectoryEntry,
)
from agent.events import Usage, UserMessage
from agent.providers.base import EventFactory


def _image() -> str:
    return "example.test/python@sha256:" + "a" * 64


def _task_contract(**overrides: object) -> BenchmarkTask:
    values: dict[str, object] = {
        "task_id": "fix-add-bug",
        "title": "Fix a simple addition bug",
        "prompt": "Fix the bug and run the tests.",
        "category": "bug_fix",
        "fixture": "fixtures/fix-add-bug",
        "verification": [{"argv": ["python", "-m", "pytest", "-q"]}],
    }
    values.update(overrides)
    return BenchmarkTask.model_validate(values)


def test_task_contract_keeps_its_category_and_public_milestones():
    task = _task_contract(
        allowed_changed_paths=("src/math_utils.py",),
        milestones=(
            MilestoneSpec(
                milestone_id="inspect-workspace",
                description="Inspect the workspace before editing.",
                kind=MilestoneKind.TOOL_CALLED,
                tool_names=("read_file", "grep"),
            ),
        ),
        forbidden_tool_names=("network_fetch",),
    )

    assert task.category is TaskCategory.BUG_FIX
    assert task.allowed_changed_paths == ("src/math_utils.py",)
    assert task.milestones[0].tool_names == ("read_file", "grep")
    assert task.forbidden_tool_names == ("network_fetch",)


def test_task_contract_rejects_duplicate_milestone_ids():
    milestone = {
        "milestone_id": "inspect-workspace",
        "description": "Inspect the workspace.",
        "kind": "tool_called",
        "tool_names": ["read_file"],
    }

    with pytest.raises(ValueError, match="milestone_id values must be unique"):
        _task_contract(milestones=(milestone, milestone))


def test_task_contract_rejects_a_milestone_that_requires_a_forbidden_tool():
    with pytest.raises(
        ValueError,
        match="milestone cannot require a forbidden tool",
    ):
        _task_contract(
            milestones=(
                {
                    "milestone_id": "inspect-workspace",
                    "description": "Inspect the workspace.",
                    "kind": "tool_called",
                    "tool_names": ["read_file"],
                },
            ),
            forbidden_tool_names=("read_file",),
        )


def test_task_contract_rejects_duplicate_allowed_changed_paths():
    with pytest.raises(ValueError, match="allowed_changed_paths must be unique"):
        _task_contract(
            allowed_changed_paths=("src/math_utils.py", "src/math_utils.py"),
        )


def test_config_requires_three_attempts_and_a_pinned_image():
    with pytest.raises(ValueError, match="greater than or equal to 3"):
        BenchmarkRunConfig(
            provider="openai",
            model="gpt-5.6-terra",
            agent_revision="a" * 40,
            container_image=_image(),
            attempts=2,
        )


def test_config_fingerprint_is_stable_for_equal_configuration():
    common = {
        "provider": "openai",
        "model": "gpt-5.6-terra",
        "agent_revision": "a" * 40,
        "container_image": _image(),
        "agent_settings": {"effort": "high"},
    }
    assert BenchmarkRunConfig(**common).fingerprint == BenchmarkRunConfig(
        **common
    ).fingerprint


def test_config_fingerprint_changes_for_a_different_routing_strategy():
    fixed = BenchmarkRunConfig(
        provider="openai",
        model="gpt-5.6-terra",
        agent_revision="a" * 40,
        container_image=_image(),
        strategy_id="fixed",
    )
    live = BenchmarkRunConfig(
        provider="openai",
        model="gpt-5.6-terra",
        agent_revision="a" * 40,
        container_image=_image(),
        strategy_id="live",
    )

    assert fixed.fingerprint != live.fingerprint


def test_run_metrics_preserves_the_actual_model_used_for_each_turn():
    luna_turn = ExecutedTurn(
        provider="openai",
        model="gpt-5.6-luna",
        route_id="economy-luna-medium",
        usage=Usage(
            input_tokens=100,
            output_tokens=10,
            cache_read_input_tokens=5,
            cache_creation_input_tokens=2,
        ),
        cost_usd=Decimal("0.003"),
    )
    terra_turn = ExecutedTurn(
        provider="openai",
        model="gpt-5.6-terra",
        route_id="strong-terra-high",
        usage=Usage(
            input_tokens=200,
            output_tokens=20,
            cache_read_input_tokens=6,
            cache_creation_input_tokens=3,
        ),
        cost_usd=Decimal("0.020"),
    )

    metrics = RunMetrics(
        duration_seconds=Decimal("1.2"),
        input_tokens=300,
        output_tokens=30,
        cache_read_tokens=11,
        cache_creation_tokens=5,
        turns=2,
        executed_turns=(luna_turn, terra_turn),
        cost_usd=Decimal("0.023"),
    )

    assert [turn.model for turn in metrics.executed_turns] == [
        "gpt-5.6-luna",
        "gpt-5.6-terra",
    ]
    assert metrics.cost_usd == Decimal("0.023")


def test_run_metrics_rejects_a_summary_that_disagrees_with_turn_receipts():
    turn = ExecutedTurn(
        provider="openai",
        model="gpt-5.6-luna",
        route_id="economy-luna-medium",
        usage=Usage(input_tokens=100, output_tokens=10),
        cost_usd=Decimal("0.003"),
    )

    with pytest.raises(ValueError, match="input_tokens"):
        RunMetrics(
            duration_seconds=Decimal("1"),
            input_tokens=101,
            output_tokens=10,
            turns=1,
            executed_turns=(turn,),
            cost_usd=Decimal("0.003"),
        )


def test_run_metrics_marks_total_cost_unknown_when_a_turn_price_is_unknown():
    turn = ExecutedTurn(
        provider="openai",
        model="future-model",
        usage=Usage(input_tokens=100),
        cost_usd=None,
    )

    metrics = RunMetrics(
        duration_seconds=Decimal("1"),
        input_tokens=100,
        turns=1,
        executed_turns=(turn,),
        cost_usd=None,
    )

    assert metrics.cost_usd is None


def test_trajectory_entry_rejects_a_mismatched_sequence():
    event = EventFactory("session")(UserMessage, text="hello")

    with pytest.raises(ValueError, match="sequence"):
        TrajectoryEntry(
            sequence=2,
            event_type="user.message",
            event=event,
        )


def test_scoreboard_row_requires_the_mathematically_correct_pass_rate():
    with pytest.raises(ValueError, match="pass_rate"):
        ScoreboardRow(
            task_id="fix-add-bug",
            provider="openai",
            model="gpt-5.6-terra",
            agent_revision="a" * 40,
            container_image=_image(),
            config_fingerprint="b" * 64,
            attempts=3,
            passed_attempts=2,
            pass_rate=Decimal("1"),
            mean_duration_seconds=Decimal("1"),
            duration_standard_deviation_seconds=Decimal("0"),
            mean_tokens=Decimal("1"),
            token_standard_deviation=Decimal("0"),
            mean_turns=Decimal("1"),
        )


def test_command_spec_uses_argv_not_an_empty_shell_command():
    with pytest.raises(ValueError, match="at least 1 item"):
        CommandSpec(argv=())
