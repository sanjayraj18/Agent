"""Fair comparison of benchmark strategies such as fixed and live routing."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from agent.benchmark.models import ScoreboardRow


class ComparisonError(ValueError):
    """The supplied scoreboard rows cannot be compared fairly."""


class BaselineRelation(StrEnum):
    """How one candidate performs relative to the chosen baseline."""

    DOMINATES = "dominates_baseline"
    DOMINATED = "dominated_by_baseline"
    TRADE_OFF = "trade_off"
    EQUIVALENT = "equivalent"
    UNKNOWN_COST = "unknown_cost"


class CandidateComparison(BaseModel):
    """One strategy and its measured difference from the baseline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: ScoreboardRow

    # Positive means the candidate passed more often.
    pass_rate_delta: Decimal

    # Negative means the candidate is cheaper.
    mean_cost_delta_usd: Decimal | None = None

    # Positive means money was saved; negative means it cost more.
    cost_savings_fraction: Decimal | None = None

    # Negative means the candidate is faster.
    mean_duration_delta_seconds: Decimal

    relation_to_baseline: BaselineRelation

    # None means an unknown price prevents an honest Pareto conclusion.
    pareto_optimal: bool | None


class BenchmarkComparison(BaseModel):
    """A baseline row plus comparable candidate strategies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=3)
    baseline: ScoreboardRow
    candidates: tuple[CandidateComparison, ...] = Field(min_length=1)


def compare_scoreboard_rows(
    rows: Iterable[ScoreboardRow],
    *,
    baseline_strategy_id: str,
) -> BenchmarkComparison:
    """
    Compare several strategies measured against the same benchmark task.

    The rows must use the same task, provider, agent revision, container image,
    and attempt count. Different routes/models are intentional: that is the
    decision being measured.
    """
    candidates = tuple(rows)

    if len(candidates) < 2:
        raise ComparisonError("at least two scoreboard rows are required")

    normalized_baseline_id = baseline_strategy_id.strip()
    if not normalized_baseline_id:
        raise ComparisonError("baseline_strategy_id must not be blank")

    strategy_ids = [row.strategy_id for row in candidates]
    if len(strategy_ids) != len(set(strategy_ids)):
        raise ComparisonError(
            "every compared row must have a unique strategy_id"
        )

    try:
        baseline = next(
            row
            for row in candidates
            if row.strategy_id == normalized_baseline_id
        )
    except StopIteration as exc:
        raise ComparisonError(
            f"baseline strategy was not found: {normalized_baseline_id!r}"
        ) from exc

    _require_same_evaluation_basis(candidates, baseline)

    comparisons = tuple(
        _compare_candidate(
            candidate,
            baseline=baseline,
            all_rows=candidates,
        )
        for candidate in candidates
        if candidate is not baseline
    )

    return BenchmarkComparison(
        task_id=baseline.task_id,
        baseline=baseline,
        candidates=comparisons,
    )


def _require_same_evaluation_basis(
    rows: tuple[ScoreboardRow, ...],
    baseline: ScoreboardRow,
) -> None:
    """Reject comparisons where the evidence came from different conditions."""
    for row in rows:
        if row.task_id != baseline.task_id:
            raise ComparisonError(
                "cannot compare rows from different benchmark tasks"
            )

        if row.provider != baseline.provider:
            raise ComparisonError(
                "cannot compare rows from different providers"
            )

        if row.agent_revision != baseline.agent_revision:
            raise ComparisonError(
                "cannot compare rows from different agent revisions"
            )

        if row.container_image != baseline.container_image:
            raise ComparisonError(
                "cannot compare rows from different container images"
            )

        if row.attempts != baseline.attempts:
            raise ComparisonError(
                "cannot compare rows with different attempt counts"
            )


def _compare_candidate(
    candidate: ScoreboardRow,
    *,
    baseline: ScoreboardRow,
    all_rows: tuple[ScoreboardRow, ...],
) -> CandidateComparison:
    return CandidateComparison(
        candidate=candidate,
        pass_rate_delta=candidate.pass_rate - baseline.pass_rate,
        mean_cost_delta_usd=_cost_delta(candidate, baseline),
        cost_savings_fraction=_cost_savings_fraction(
            candidate,
            baseline,
        ),
        mean_duration_delta_seconds=(
            candidate.mean_duration_seconds
            - baseline.mean_duration_seconds
        ),
        relation_to_baseline=_relation_to_baseline(
            candidate,
            baseline,
        ),
        pareto_optimal=_is_pareto_optimal(candidate, all_rows),
    )


def _cost_delta(
    candidate: ScoreboardRow,
    baseline: ScoreboardRow,
) -> Decimal | None:
    if (
        candidate.mean_cost_usd is None
        or baseline.mean_cost_usd is None
    ):
        return None

    return candidate.mean_cost_usd - baseline.mean_cost_usd


def _cost_savings_fraction(
    candidate: ScoreboardRow,
    baseline: ScoreboardRow,
) -> Decimal | None:
    if (
        candidate.mean_cost_usd is None
        or baseline.mean_cost_usd is None
        or baseline.mean_cost_usd == 0
    ):
        return None

    return (
        baseline.mean_cost_usd - candidate.mean_cost_usd
    ) / baseline.mean_cost_usd


def _relation_to_baseline(
    candidate: ScoreboardRow,
    baseline: ScoreboardRow,
) -> BaselineRelation:
    if (
        candidate.mean_cost_usd is None
        or baseline.mean_cost_usd is None
    ):
        return BaselineRelation.UNKNOWN_COST

    if _dominates(candidate, baseline):
        return BaselineRelation.DOMINATES

    if _dominates(baseline, candidate):
        return BaselineRelation.DOMINATED

    if (
        candidate.pass_rate == baseline.pass_rate
        and candidate.mean_cost_usd == baseline.mean_cost_usd
        and (
            candidate.mean_duration_seconds
            == baseline.mean_duration_seconds
        )
    ):
        return BaselineRelation.EQUIVALENT

    return BaselineRelation.TRADE_OFF


def _is_pareto_optimal(
    candidate: ScoreboardRow,
    rows: tuple[ScoreboardRow, ...],
) -> bool | None:
    """
    Return whether no other strategy is better on quality, cost, and speed.

    Unknown prices make this deliberately inconclusive instead of pretending
    that a partial cost catalog is sufficient evidence.
    """
    if any(row.mean_cost_usd is None for row in rows):
        return None

    return not any(
        other is not candidate and _dominates(other, candidate)
        for other in rows
    )


def _dominates(
    left: ScoreboardRow,
    right: ScoreboardRow,
) -> bool:
    """Return whether left is no worse everywhere and better somewhere."""
    if left.mean_cost_usd is None or right.mean_cost_usd is None:
        return False

    no_worse = (
        left.pass_rate >= right.pass_rate
        and left.mean_cost_usd <= right.mean_cost_usd
        and left.mean_duration_seconds <= right.mean_duration_seconds
    )
    strictly_better_somewhere = (
        left.pass_rate > right.pass_rate
        or left.mean_cost_usd < right.mean_cost_usd
        or left.mean_duration_seconds < right.mean_duration_seconds
    )

    return no_worse and strictly_better_somewhere
