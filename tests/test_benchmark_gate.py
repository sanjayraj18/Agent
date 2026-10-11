import json
from decimal import Decimal
from pathlib import Path

import pytest

from agent.benchmark.diagnosis import build_diagnosis_report
from agent.benchmark.cli import evaluate_saved_regression_gate
from agent.benchmark.gate import (
    GateError,
    evaluate_regression_gate,
    load_gate_policy,
    load_quality_snapshot,
)
from agent.benchmark.models import (
    DiagnosisCategory,
    GateDecision,
    GatePolicy,
    GateRule,
    QualitySnapshot,
    ReliabilityPoint,
    ReliabilitySummary,
    RunDiagnosis,
    RunStatus,
    ScoreboardRow,
)
from agent.benchmark.report import (
    write_diagnosis_report,
    write_scoreboard,
)


def _policy(**overrides: object) -> GatePolicy:
    values: dict[str, object] = {
        "policy_id": "test-quality-policy",
        "minimum_pass_rate": Decimal("0.50"),
        "strict_reliability_k": 3,
        "maximum_strict_reliability_drop": Decimal("0.05"),
        "require_zero_policy_violations": True,
        "require_known_cost": True,
        "maximum_cost_increase_fraction": Decimal("0.20"),
        "maximum_duration_increase_fraction": None,
    }
    values.update(overrides)
    return GatePolicy.model_validate(values)


def _diagnosis(
    *,
    run_id: str,
    category: DiagnosisCategory,
    task_id: str,
) -> RunDiagnosis:
    status = (
        RunStatus.PASSED
        if category is DiagnosisCategory.PASSED
        else RunStatus.FAILED
    )
    return RunDiagnosis(
        run_id=run_id,
        task_id=task_id,
        attempt=int(run_id.rsplit("-", maxsplit=1)[1]),
        run_status=status,
        category=category,
        summary=f"{category.value} summary",
        evidence=(
            {
                "source": "run_status",
                "message": f"run status: {status.value}",
            },
        ),
        suggested_next_step="Inspect recorded evidence.",
    )


def _snapshot(
    *,
    fingerprint: str,
    passed_attempts: int = 3,
    strict_reliability: Decimal = Decimal("1"),
    mean_cost: Decimal | None = Decimal("0.020"),
    task_id: str = "fix-add-bug",
    failure_category: DiagnosisCategory = DiagnosisCategory.VERIFICATION_FAILED,
    container_image: str | None = None,
) -> QualitySnapshot:
    attempts = 3
    pass_one = Decimal(passed_attempts) / Decimal(attempts)
    pass_at_three = Decimal("1") if passed_attempts else Decimal("0")
    scoreboard = ScoreboardRow(
        task_id=task_id,
        provider="openai",
        model="gpt-5.6-terra",
        route_id="strong-terra-high",
        strategy_id=f"candidate-{fingerprint[0]}",
        agent_revision="a" * 40,
        container_image=(
            container_image
            or "example.test/python@sha256:" + "b" * 64
        ),
        config_fingerprint=fingerprint,
        attempts=attempts,
        passed_attempts=passed_attempts,
        pass_rate=pass_one,
        reliability=ReliabilitySummary(
            attempts=attempts,
            passed_attempts=passed_attempts,
            pass_one=pass_one,
            points=(
                ReliabilityPoint(
                    k=3,
                    pass_all=strict_reliability,
                    pass_at_least_one=pass_at_three,
                ),
            ),
        ),
        mean_duration_seconds=Decimal("2"),
        duration_standard_deviation_seconds=Decimal("0"),
        mean_tokens=Decimal("100"),
        token_standard_deviation=Decimal("0"),
        mean_turns=Decimal("2"),
        mean_cost_usd=mean_cost,
        cost_standard_deviation_usd=Decimal("0") if mean_cost is not None else None,
    )
    diagnoses = tuple(
        _diagnosis(
            run_id=f"run-{attempt}",
            task_id=task_id,
            category=(
                DiagnosisCategory.PASSED
                if attempt <= passed_attempts
                else failure_category
            ),
        )
        for attempt in range(1, attempts + 1)
    )
    return QualitySnapshot(
        scoreboard=scoreboard,
        diagnosis_report=build_diagnosis_report(
            diagnoses,
            config_fingerprint=fingerprint,
        ),
    )


def test_regression_gate_accepts_a_candidate_that_meets_every_rule():
    result = evaluate_regression_gate(
        baseline=_snapshot(fingerprint="a" * 64),
        candidate=_snapshot(
            fingerprint="c" * 64,
            mean_cost=Decimal("0.018"),
        ),
        policy=_policy(),
    )

    assert result.decision is GateDecision.ACCEPTED
    assert all(check.passed for check in result.checks)


def test_regression_gate_rejects_a_strict_reliability_regression():
    result = evaluate_regression_gate(
        baseline=_snapshot(fingerprint="a" * 64),
        candidate=_snapshot(
            fingerprint="c" * 64,
            passed_attempts=2,
            strict_reliability=Decimal("0"),
        ),
        policy=_policy(),
    )

    assert result.decision is GateDecision.REJECTED
    assert [
        check.rule
        for check in result.checks
        if not check.passed
    ] == [GateRule.STRICT_RELIABILITY]


def test_regression_gate_rejects_candidate_policy_violations():
    result = evaluate_regression_gate(
        baseline=_snapshot(
            fingerprint="a" * 64,
            passed_attempts=2,
            strict_reliability=Decimal("0"),
        ),
        candidate=_snapshot(
            fingerprint="c" * 64,
            passed_attempts=2,
            strict_reliability=Decimal("0"),
            failure_category=DiagnosisCategory.POLICY_VIOLATION,
        ),
        policy=_policy(),
    )

    failed_rules = {
        check.rule
        for check in result.checks
        if not check.passed
    }
    assert failed_rules == {GateRule.ZERO_POLICY_VIOLATIONS}


def test_regression_gate_rejects_unknown_candidate_cost():
    result = evaluate_regression_gate(
        baseline=_snapshot(fingerprint="a" * 64),
        candidate=_snapshot(fingerprint="c" * 64, mean_cost=None),
        policy=_policy(),
    )

    failed_rules = {
        check.rule
        for check in result.checks
        if not check.passed
    }
    assert GateRule.KNOWN_COST in failed_rules
    assert GateRule.COST_INCREASE in failed_rules


def test_regression_gate_rejects_incomparable_task_evidence():
    with pytest.raises(GateError, match="different benchmark tasks"):
        evaluate_regression_gate(
            baseline=_snapshot(fingerprint="a" * 64),
            candidate=_snapshot(
                fingerprint="c" * 64,
                task_id="repair-label-parser",
            ),
            policy=_policy(),
        )


def test_load_quality_snapshot_pairs_saved_evidence_by_fingerprint(
    tmp_path: Path,
):
    snapshot = _snapshot(fingerprint="a" * 64)
    scoreboard_path = tmp_path / "scoreboard.md"
    diagnosis_path = tmp_path / "diagnosis.md"
    write_scoreboard(scoreboard_path, (snapshot.scoreboard,))
    write_diagnosis_report(diagnosis_path, snapshot.diagnosis_report)

    loaded = load_quality_snapshot(
        scoreboard_path=scoreboard_path.with_suffix(".json"),
        diagnosis_path=diagnosis_path.with_suffix(".json"),
    )

    assert loaded == snapshot


def test_saved_gate_writes_a_machine_readable_and_markdown_decision(
    tmp_path: Path,
):
    baseline = _snapshot(fingerprint="a" * 64)
    candidate = _snapshot(
        fingerprint="c" * 64,
        mean_cost=Decimal("0.018"),
    )
    baseline_scoreboard = tmp_path / "baseline-scoreboard.md"
    baseline_diagnosis = tmp_path / "baseline-diagnosis.md"
    candidate_scoreboard = tmp_path / "candidate-scoreboard.md"
    candidate_diagnosis = tmp_path / "candidate-diagnosis.md"
    policy_path = tmp_path / "policy.json"
    output_path = tmp_path / "gate.md"

    write_scoreboard(baseline_scoreboard, (baseline.scoreboard,))
    write_diagnosis_report(baseline_diagnosis, baseline.diagnosis_report)
    write_scoreboard(candidate_scoreboard, (candidate.scoreboard,))
    write_diagnosis_report(candidate_diagnosis, candidate.diagnosis_report)
    policy_path.write_text(
        json.dumps(_policy().model_dump(mode="json")),
        encoding="utf-8",
    )

    execution = evaluate_saved_regression_gate(
        baseline_scoreboard_path=baseline_scoreboard.with_suffix(".json"),
        baseline_diagnosis_path=baseline_diagnosis.with_suffix(".json"),
        candidate_scoreboard_path=candidate_scoreboard.with_suffix(".json"),
        candidate_diagnosis_path=candidate_diagnosis.with_suffix(".json"),
        policy_path=policy_path,
        output_path=output_path,
    )

    assert execution.result.decision is GateDecision.ACCEPTED
    assert output_path.with_suffix(".json").exists()
    assert "Decision: **ACCEPTED**" in output_path.read_text(encoding="utf-8")
    assert load_gate_policy(policy_path) == _policy()
