"""Policy-based acceptance decisions for measured agent benchmark evidence."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from agent.benchmark.models import (
    DiagnosisCategory,
    GateCheck,
    GateDecision,
    GatePolicy,
    GateRule,
    QualitySnapshot,
    RegressionGateResult,
    ScoreboardRow,
)
from agent.benchmark.report import read_diagnosis_report, read_scoreboard


class GateError(ValueError):
    """Saved evidence or policy cannot support a fair gate decision."""


def load_gate_policy(path: Path) -> GatePolicy:
    """Load an explicit JSON gate policy instead of hiding limits in code."""
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GateError(f"could not read gate policy {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GateError(f"{path}: invalid gate policy JSON ({exc})") from exc

    try:
        return GatePolicy.model_validate(decoded)
    except Exception as exc:
        raise GateError(f"{path}: invalid gate policy ({exc})") from exc


def load_quality_snapshot(
    *,
    scoreboard_path: Path,
    diagnosis_path: Path,
) -> QualitySnapshot:
    """Pair one diagnosis report with its matching scoreboard row by fingerprint."""
    diagnosis_report = read_diagnosis_report(diagnosis_path)
    rows = read_scoreboard(scoreboard_path)
    matches: tuple[ScoreboardRow, ...] = tuple(
        row
        for row in rows
        if row.config_fingerprint == diagnosis_report.config_fingerprint
    )

    match = next(iter(matches), None)
    if match is None:
        raise GateError(
            "no scoreboard row matches the diagnosis report configuration "
            f"fingerprint in {diagnosis_path}"
        )

    if len(matches) > 1:
        raise GateError(
            "more than one scoreboard row matches the diagnosis report "
            "configuration fingerprint"
        )

    return QualitySnapshot(
        scoreboard=match,
        diagnosis_report=diagnosis_report,
    )


def evaluate_regression_gate(
    *,
    baseline: QualitySnapshot,
    candidate: QualitySnapshot,
    policy: GatePolicy,
) -> RegressionGateResult:
    """Accept only candidates that satisfy every explicit quality rule."""
    _require_comparable_evidence(baseline, candidate)

    checks = (
        _minimum_pass_rate_check(candidate.scoreboard, policy),
        _strict_reliability_check(
            baseline.scoreboard,
            candidate.scoreboard,
            policy,
        ),
        _policy_violation_check(baseline, candidate, policy),
        _known_cost_check(baseline.scoreboard, candidate.scoreboard, policy),
        _cost_increase_check(
            baseline.scoreboard,
            candidate.scoreboard,
            policy,
        ),
        _duration_increase_check(
            baseline.scoreboard,
            candidate.scoreboard,
            policy,
        ),
    )
    decision = (
        GateDecision.ACCEPTED
        if all(check.passed for check in checks)
        else GateDecision.REJECTED
    )

    return RegressionGateResult(
        policy=policy,
        baseline=baseline,
        candidate=candidate,
        decision=decision,
        checks=checks,
    )


def _require_comparable_evidence(
    baseline: QualitySnapshot,
    candidate: QualitySnapshot,
) -> None:
    """Reject an apparent comparison made from different task conditions."""
    baseline_row = baseline.scoreboard
    candidate_row = candidate.scoreboard

    if baseline_row.task_id != candidate_row.task_id:
        raise GateError("cannot gate evidence from different benchmark tasks")

    if baseline_row.container_image != candidate_row.container_image:
        raise GateError("cannot gate evidence from different container images")

    if baseline_row.attempts != candidate_row.attempts:
        raise GateError("cannot gate evidence with different attempt counts")


def _minimum_pass_rate_check(
    candidate: ScoreboardRow,
    policy: GatePolicy,
) -> GateCheck:
    passed = candidate.pass_rate >= policy.minimum_pass_rate
    return GateCheck(
        rule=GateRule.MINIMUM_PASS_RATE,
        passed=passed,
        baseline_value="not applicable",
        candidate_value=_percentage(candidate.pass_rate),
        requirement=f">= {_percentage(policy.minimum_pass_rate)}",
        message=(
            "candidate pass rate meets the minimum"
            if passed
            else "candidate pass rate is below the minimum"
        ),
    )


def _strict_reliability_check(
    baseline: ScoreboardRow,
    candidate: ScoreboardRow,
    policy: GatePolicy,
) -> GateCheck:
    baseline_value = _strict_reliability(baseline, policy.strict_reliability_k)
    candidate_value = _strict_reliability(candidate, policy.strict_reliability_k)
    requirement = (
        "drop <= "
        f"{_percentage(policy.maximum_strict_reliability_drop)}"
    )

    if baseline_value is None or candidate_value is None:
        return GateCheck(
            rule=GateRule.STRICT_RELIABILITY,
            passed=False,
            baseline_value=_percentage_or_missing(baseline_value),
            candidate_value=_percentage_or_missing(candidate_value),
            requirement=requirement,
            message=(
                f"both rows must record pass^{policy.strict_reliability_k} "
                "before strict reliability can be gated"
            ),
        )

    drop = baseline_value - candidate_value
    passed = drop <= policy.maximum_strict_reliability_drop
    return GateCheck(
        rule=GateRule.STRICT_RELIABILITY,
        passed=passed,
        baseline_value=_percentage(baseline_value),
        candidate_value=_percentage(candidate_value),
        requirement=requirement,
        message=(
            "candidate strict reliability is within the allowed drop"
            if passed
            else "candidate strict reliability dropped too far"
        ),
    )


def _policy_violation_check(
    baseline: QualitySnapshot,
    candidate: QualitySnapshot,
    policy: GatePolicy,
) -> GateCheck:
    baseline_count = _policy_violation_count(baseline)
    candidate_count = _policy_violation_count(candidate)

    if not policy.require_zero_policy_violations:
        return GateCheck(
            rule=GateRule.ZERO_POLICY_VIOLATIONS,
            passed=True,
            baseline_value=str(baseline_count),
            candidate_value=str(candidate_count),
            requirement="not enforced by this policy",
            message="policy-violation count is recorded but not enforced",
        )

    passed = candidate_count == 0
    return GateCheck(
        rule=GateRule.ZERO_POLICY_VIOLATIONS,
        passed=passed,
        baseline_value=str(baseline_count),
        candidate_value=str(candidate_count),
        requirement="candidate must equal 0",
        message=(
            "candidate has no policy violations"
            if passed
            else "candidate recorded one or more policy violations"
        ),
    )


def _known_cost_check(
    baseline: ScoreboardRow,
    candidate: ScoreboardRow,
    policy: GatePolicy,
) -> GateCheck:
    baseline_cost = baseline.mean_cost_usd
    candidate_cost = candidate.mean_cost_usd

    if not policy.require_known_cost:
        return GateCheck(
            rule=GateRule.KNOWN_COST,
            passed=True,
            baseline_value=_cost(baseline_cost),
            candidate_value=_cost(candidate_cost),
            requirement="not enforced by this policy",
            message="cost availability is recorded but not enforced",
        )

    passed = baseline_cost is not None and candidate_cost is not None
    return GateCheck(
        rule=GateRule.KNOWN_COST,
        passed=passed,
        baseline_value=_cost(baseline_cost),
        candidate_value=_cost(candidate_cost),
        requirement="both costs must be known",
        message=(
            "both benchmark rows have known cost"
            if passed
            else "at least one benchmark row has unknown cost"
        ),
    )


def _cost_increase_check(
    baseline: ScoreboardRow,
    candidate: ScoreboardRow,
    policy: GatePolicy,
) -> GateCheck:
    threshold = policy.maximum_cost_increase_fraction
    if threshold is None:
        return GateCheck(
            rule=GateRule.COST_INCREASE,
            passed=True,
            baseline_value=_cost(baseline.mean_cost_usd),
            candidate_value=_cost(candidate.mean_cost_usd),
            requirement="not configured",
            message="cost increase is not limited by this policy",
        )

    increase = _relative_increase(
        baseline.mean_cost_usd,
        candidate.mean_cost_usd,
    )
    if increase is None:
        return GateCheck(
            rule=GateRule.COST_INCREASE,
            passed=False,
            baseline_value=_cost(baseline.mean_cost_usd),
            candidate_value=_cost(candidate.mean_cost_usd),
            requirement=f"increase <= {_percentage(threshold)}",
            message="cost increase cannot be evaluated from the recorded costs",
        )

    passed = increase <= threshold
    return GateCheck(
        rule=GateRule.COST_INCREASE,
        passed=passed,
        baseline_value=_cost(baseline.mean_cost_usd),
        candidate_value=_cost(candidate.mean_cost_usd),
        requirement=f"increase <= {_percentage(threshold)}",
        message=(
            "candidate cost increase is within the budget"
            if passed
            else "candidate cost increase exceeds the budget"
        ),
    )


def _duration_increase_check(
    baseline: ScoreboardRow,
    candidate: ScoreboardRow,
    policy: GatePolicy,
) -> GateCheck:
    threshold = policy.maximum_duration_increase_fraction
    if threshold is None:
        return GateCheck(
            rule=GateRule.DURATION_INCREASE,
            passed=True,
            baseline_value=_seconds(baseline.mean_duration_seconds),
            candidate_value=_seconds(candidate.mean_duration_seconds),
            requirement="not configured",
            message="duration increase is not limited by this policy",
        )

    increase = _relative_increase(
        baseline.mean_duration_seconds,
        candidate.mean_duration_seconds,
    )
    if increase is None:
        return GateCheck(
            rule=GateRule.DURATION_INCREASE,
            passed=False,
            baseline_value=_seconds(baseline.mean_duration_seconds),
            candidate_value=_seconds(candidate.mean_duration_seconds),
            requirement=f"increase <= {_percentage(threshold)}",
            message="duration increase cannot be evaluated from a zero baseline",
        )

    passed = increase <= threshold
    return GateCheck(
        rule=GateRule.DURATION_INCREASE,
        passed=passed,
        baseline_value=_seconds(baseline.mean_duration_seconds),
        candidate_value=_seconds(candidate.mean_duration_seconds),
        requirement=f"increase <= {_percentage(threshold)}",
        message=(
            "candidate duration increase is within the budget"
            if passed
            else "candidate duration increase exceeds the budget"
        ),
    )


def _strict_reliability(row: ScoreboardRow, k: int) -> Decimal | None:
    if row.reliability is None:
        return None

    for point in row.reliability.points:
        if point.k == k:
            return point.pass_all

    return None


def _policy_violation_count(snapshot: QualitySnapshot) -> int:
    return next(
        (
            pattern.count
            for pattern in snapshot.diagnosis_report.summary.failure_patterns
            if pattern.category is DiagnosisCategory.POLICY_VIOLATION
        ),
        0,
    )


def _relative_increase(
    baseline: Decimal | None,
    candidate: Decimal | None,
) -> Decimal | None:
    if baseline is None or candidate is None:
        return None

    if baseline == 0:
        return Decimal("0") if candidate == 0 else None

    return (candidate - baseline) / baseline


def _percentage(value: Decimal) -> str:
    return f"{value:.2%}"


def _percentage_or_missing(value: Decimal | None) -> str:
    return "not recorded" if value is None else _percentage(value)


def _cost(value: Decimal | None) -> str:
    return "unknown" if value is None else f"${value:.5f}"


def _seconds(value: Decimal) -> str:
    return f"{value:.2f}s"
