from decimal import Decimal

import pytest

from agent.benchmark.comparison import (
    BaselineRelation,
    ComparisonError,
    compare_scoreboard_rows,
)
from agent.benchmark.models import ScoreboardRow


def _image() -> str:
    return "example.test/python@sha256:" + "b" * 64


def _row(
    strategy_id: str,
    *,
    passed_attempts: int = 3,
    mean_cost_usd: Decimal | None = Decimal("0.02"),
    mean_duration_seconds: Decimal = Decimal("14"),
    agent_revision: str = "a" * 40,
) -> ScoreboardRow:
    return ScoreboardRow(
        task_id="fix-add-bug",
        provider="openai",
        model="gpt-5.6-terra",
        route_id=strategy_id,
        strategy_id=strategy_id,
        agent_revision=agent_revision,
        container_image=_image(),
        config_fingerprint="c" * 64,
        attempts=3,
        passed_attempts=passed_attempts,
        pass_rate=Decimal(passed_attempts) / Decimal(3),
        mean_duration_seconds=mean_duration_seconds,
        duration_standard_deviation_seconds=Decimal("0"),
        mean_tokens=Decimal("100"),
        token_standard_deviation=Decimal("0"),
        mean_turns=Decimal("2"),
        mean_cost_usd=mean_cost_usd,
        cost_standard_deviation_usd=(
            None if mean_cost_usd is None else Decimal("0")
        ),
    )


def test_comparison_identifies_a_candidate_that_dominates_the_baseline():
    comparison = compare_scoreboard_rows(
        (
            _row("strong-terra-high"),
            _row(
                "economy-luna-medium",
                mean_cost_usd=Decimal("0.002"),
                mean_duration_seconds=Decimal("11"),
            ),
        ),
        baseline_strategy_id="strong-terra-high",
    )

    candidate = comparison.candidates[0]
    assert candidate.relation_to_baseline is BaselineRelation.DOMINATES
    assert candidate.mean_cost_delta_usd == Decimal("-0.018")
    assert candidate.cost_savings_fraction == Decimal("0.9")
    assert candidate.mean_duration_delta_seconds == Decimal("-3")
    assert candidate.pareto_optimal is True


def test_comparison_labels_a_quality_cost_speed_trade_off_honestly():
    comparison = compare_scoreboard_rows(
        (
            _row("strong-terra-high"),
            _row(
                "economy-luna-medium",
                passed_attempts=2,
                mean_cost_usd=Decimal("0.002"),
                mean_duration_seconds=Decimal("11"),
            ),
        ),
        baseline_strategy_id="strong-terra-high",
    )

    candidate = comparison.candidates[0]
    assert candidate.relation_to_baseline is BaselineRelation.TRADE_OFF
    assert candidate.pass_rate_delta == Decimal("-1") / Decimal("3")
    assert candidate.pareto_optimal is True


def test_comparison_is_inconclusive_when_any_price_is_unknown():
    comparison = compare_scoreboard_rows(
        (
            _row("strong-terra-high"),
            _row("experimental", mean_cost_usd=None),
        ),
        baseline_strategy_id="strong-terra-high",
    )

    candidate = comparison.candidates[0]
    assert candidate.relation_to_baseline is BaselineRelation.UNKNOWN_COST
    assert candidate.mean_cost_delta_usd is None
    assert candidate.pareto_optimal is None


def test_comparison_rejects_evidence_from_a_different_agent_revision():
    with pytest.raises(ComparisonError, match="agent revisions"):
        compare_scoreboard_rows(
            (
                _row("strong-terra-high"),
                _row(
                    "economy-luna-medium",
                    agent_revision="c" * 40,
                ),
            ),
            baseline_strategy_id="strong-terra-high",
        )
