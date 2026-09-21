"""Aggregate repeated benchmark attempts into reproducible measurements."""

from __future__ import annotations

from decimal import Decimal
from math import sqrt
from typing import Iterable

from agent.benchmark.models import (
    BenchmarkRunConfig,
    BenchmarkRunResult,
    RunStatus,
    ScoreboardRow,
)


class ScoreboardError(ValueError):
    """The supplied attempts cannot form one comparable scoreboard row."""


def build_scoreboard_row(
    results: Iterable[BenchmarkRunResult],
    config: BenchmarkRunConfig,
) -> ScoreboardRow:
    """Aggregate exactly one task/configuration group."""
    attempts = tuple(results)
    if not attempts:
        raise ScoreboardError("at least one result is required")

    task_ids = {result.task_id for result in attempts}
    if len(task_ids) != 1:
        raise ScoreboardError("scoreboard rows may contain only one task")

    passed = sum(result.status is RunStatus.PASSED for result in attempts)
    costs = [result.metrics.cost_usd for result in attempts]
    known_costs = [cost for cost in costs if cost is not None]

    # A partial pricing catalog must not become a deceptively low average.
    mean_cost = _mean(known_costs) if len(known_costs) == len(costs) else None
    cost_deviation = (
        _population_standard_deviation(known_costs)
        if len(known_costs) == len(costs)
        else None
    )

    durations = [result.metrics.duration_seconds for result in attempts]
    token_counts = [Decimal(result.metrics.total_tokens) for result in attempts]
    turns = [Decimal(result.metrics.turns) for result in attempts]

    return ScoreboardRow(
        task_id=attempts[0].task_id,
        provider=config.provider,
        model=config.model,
        agent_revision=config.agent_revision,
        container_image=config.container_image,
        config_fingerprint=config.fingerprint,
        attempts=len(attempts),
        passed_attempts=passed,
        pass_rate=Decimal(passed) / Decimal(len(attempts)),
        mean_duration_seconds=_mean(durations),
        duration_standard_deviation_seconds=(
            _population_standard_deviation(durations)
        ),
        mean_tokens=_mean(token_counts),
        token_standard_deviation=_population_standard_deviation(token_counts),
        mean_turns=_mean(turns),
        mean_cost_usd=mean_cost,
        cost_standard_deviation_usd=cost_deviation,
    )


def _mean(values: list[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")
    return sum(values, start=Decimal("0")) / Decimal(len(values))


def _population_standard_deviation(values: list[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")
    average = _mean(values)
    variance = sum(
        ((value - average) ** 2 for value in values),
        start=Decimal("0"),
    ) / Decimal(len(values))
    # Decimal does not guarantee a context precision appropriate for an
    # arbitrary sqrt. A float is sufficient for a report-only deviation.
    return Decimal(str(sqrt(float(variance))))
