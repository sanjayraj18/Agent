"""Aggregate repeated benchmark attempts into reproducible measurements."""

from __future__ import annotations

from decimal import Decimal
from math import sqrt
from typing import Iterable

from agent.benchmark.models import (
    BenchmarkRunConfig,
    BenchmarkRunResult,
    ExecutedTurn,
    ModelMixEntry,
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
        route_id=config.route_id,
        strategy_id=config.strategy_id,
        model_mix=_build_model_mix(attempts),
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


def _build_model_mix(
    results: tuple[BenchmarkRunResult, ...],
) -> tuple[ModelMixEntry, ...]:
    """
    Group real LLM turns by provider, model, and route.

    This is based on execution receipts, not the benchmark's starting model.
    """
    grouped: dict[
        tuple[str, str, str | None],
        list[ExecutedTurn],
    ] = {}

    for result in results:
        for turn in result.metrics.executed_turns:
            key = (
                turn.provider,
                turn.model,
                turn.route_id,
            )
            grouped.setdefault(key, []).append(turn)

    entries: list[ModelMixEntry] = []

    for (provider, model, route_id), turns in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2] or "",
        ),
    ):
        costs = [turn.cost_usd for turn in turns]
        total_cost = (
            None
            if any(cost is None for cost in costs)
            else sum(
                (cost for cost in costs if cost is not None),
                start=Decimal("0"),
            )
        )

        entries.append(
            ModelMixEntry(
                provider=provider,
                model=model,
                route_id=route_id,
                turns=len(turns),
                input_tokens=sum(
                    turn.usage.input_tokens
                    for turn in turns
                ),
                output_tokens=sum(
                    turn.usage.output_tokens
                    for turn in turns
                ),
                cache_read_tokens=sum(
                    turn.usage.cache_read_input_tokens
                    for turn in turns
                ),
                cache_creation_tokens=sum(
                    turn.usage.cache_creation_input_tokens
                    for turn in turns
                ),
                cost_usd=total_cost,
            )
        )

    return tuple(entries)
