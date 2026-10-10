from datetime import datetime, timezone
from decimal import Decimal

from agent.benchmark.models import (
    BenchmarkRunConfig,
    BenchmarkRunResult,
    CommandSpec,
    ExecutedTurn,
    RunMetrics,
    RunStatus,
    TrajectoryReference,
    VerificationResult,
)
from agent.benchmark.scoreboard import build_scoreboard_row
from agent.events import Usage


def _image() -> str:
    return "example.test/python@sha256:" + "a" * 64


def _result(attempt: int, status: RunStatus) -> BenchmarkRunResult:
    verification = (
        VerificationResult(
            command=CommandSpec(argv=("python", "-V")),
            exit_code=0,
            passed=True,
        ),
    ) if status is RunStatus.PASSED else ()
    now = datetime.now(timezone.utc)
    return BenchmarkRunResult(
        run_id=f"run-{attempt}",
        task_id="fix-add-bug",
        attempt=attempt,
        status=status,
        started_at=now,
        finished_at=now,
        metrics=RunMetrics(
            duration_seconds=Decimal(attempt),
            input_tokens=10,
            output_tokens=5,
            turns=2,
            cost_usd=Decimal("0.01"),
        ),
        trajectory=TrajectoryReference(
            relative_path=f"run-{attempt}/trajectory.jsonl",
            event_count=0,
            sha256="b" * 64,
        ),
        verification=verification,
    )


def _metrics_for(*executed_turns: ExecutedTurn) -> RunMetrics:
    costs = [turn.cost_usd for turn in executed_turns]
    total_cost = (
        None
        if any(cost is None for cost in costs)
        else sum(
            (cost for cost in costs if cost is not None),
            start=Decimal("0"),
        )
    )

    return RunMetrics(
        duration_seconds=Decimal("1"),
        input_tokens=sum(
            turn.usage.input_tokens for turn in executed_turns
        ),
        output_tokens=sum(
            turn.usage.output_tokens for turn in executed_turns
        ),
        cache_read_tokens=sum(
            turn.usage.cache_read_input_tokens
            for turn in executed_turns
        ),
        cache_creation_tokens=sum(
            turn.usage.cache_creation_input_tokens
            for turn in executed_turns
        ),
        turns=len(executed_turns),
        executed_turns=executed_turns,
        cost_usd=total_cost,
    )


def test_scoreboard_calculates_pass_rate_and_variance():
    config = BenchmarkRunConfig(
        provider="openai",
        model="gpt-5.6-terra",
        agent_revision="a" * 40,
        container_image=_image(),
    )
    row = build_scoreboard_row(
        (
            _result(1, RunStatus.PASSED),
            _result(2, RunStatus.FAILED),
            _result(3, RunStatus.PASSED),
        ),
        config,
    )

    assert row.pass_rate == Decimal(2) / Decimal(3)
    assert row.reliability is not None
    assert row.reliability.pass_one == Decimal(2) / Decimal(3)
    assert row.reliability.points[0].k == 3
    assert row.reliability.points[0].pass_all == Decimal("0")
    assert row.reliability.points[0].pass_at_least_one == Decimal("1")
    assert row.duration_standard_deviation_seconds > 0
    assert row.mean_cost_usd == Decimal("0.01")


def test_scoreboard_reports_the_actual_model_mix_for_a_live_strategy():
    luna_turn = ExecutedTurn(
        provider="openai",
        model="gpt-5.6-luna",
        route_id="economy-luna-medium",
        usage=Usage(input_tokens=100, output_tokens=10),
        cost_usd=Decimal("0.003"),
    )
    terra_turn = ExecutedTurn(
        provider="openai",
        model="gpt-5.6-terra",
        route_id="strong-terra-high",
        usage=Usage(input_tokens=200, output_tokens=20),
        cost_usd=Decimal("0.020"),
    )
    results = (
        _result(1, RunStatus.PASSED).model_copy(
            update={"metrics": _metrics_for(luna_turn, terra_turn)}
        ),
        _result(2, RunStatus.PASSED).model_copy(
            update={"metrics": _metrics_for(luna_turn)}
        ),
        _result(3, RunStatus.PASSED).model_copy(
            update={"metrics": _metrics_for(terra_turn)}
        ),
    )
    config = BenchmarkRunConfig(
        provider="openai",
        model="gpt-5.6-terra",
        route_id="strong-terra-high",
        strategy_id="live",
        agent_revision="a" * 40,
        container_image=_image(),
    )

    row = build_scoreboard_row(results, config)

    assert row.strategy_id == "live"
    assert [
        (entry.model, entry.route_id, entry.turns, entry.cost_usd)
        for entry in row.model_mix
    ] == [
        (
            "gpt-5.6-luna",
            "economy-luna-medium",
            2,
            Decimal("0.006"),
        ),
        (
            "gpt-5.6-terra",
            "strong-terra-high",
            2,
            Decimal("0.040"),
        ),
    ]
