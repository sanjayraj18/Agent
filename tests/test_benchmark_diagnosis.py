from datetime import datetime, timezone
from decimal import Decimal

from agent.benchmark.diagnosis import diagnose_run
from agent.benchmark.models import (
    BenchmarkRunResult,
    CommandSpec,
    DiagnosisCategory,
    DiagnosisEvidenceSource,
    EvaluationResult,
    EvaluationViolation,
    EvaluationViolationKind,
    RunMetrics,
    RunStatus,
    TrajectoryReference,
    VerificationResult,
)


def _result(
    *,
    status: RunStatus,
    error_message: str | None = None,
    evaluation: EvaluationResult | None = None,
    verification: tuple[VerificationResult, ...] = (),
) -> BenchmarkRunResult:
    if status is RunStatus.PASSED and not verification:
        verification = (
            VerificationResult(
                command=CommandSpec(argv=("python", "-m", "pytest", "-q")),
                exit_code=0,
                passed=True,
            ),
        )

    now = datetime.now(timezone.utc)
    return BenchmarkRunResult(
        run_id="fix-add-bug-001",
        task_id="fix-add-bug",
        attempt=1,
        status=status,
        started_at=now,
        finished_at=now,
        metrics=RunMetrics(duration_seconds=Decimal("1")),
        trajectory=TrajectoryReference(
            relative_path="fix-add-bug-001/trajectory.jsonl",
            event_count=3,
            sha256="a" * 64,
        ),
        verification=verification,
        error_message=error_message,
        evaluation=evaluation,
    )


def _failed_verification() -> VerificationResult:
    return VerificationResult(
        command=CommandSpec(argv=("python", "-m", "pytest", "-q")),
        exit_code=1,
        passed=False,
    )


def test_diagnose_run_marks_a_passing_run_as_passed():
    diagnosis = diagnose_run(_result(status=RunStatus.PASSED))

    assert diagnosis.category is DiagnosisCategory.PASSED
    assert diagnosis.evidence[0].source is DiagnosisEvidenceSource.RUN_STATUS


def test_diagnose_run_prioritizes_timeout_over_other_possible_evidence():
    diagnosis = diagnose_run(
        _result(
            status=RunStatus.TIMED_OUT,
            error_message="agent attempt exceeded 30 seconds",
        )
    )

    assert diagnosis.category is DiagnosisCategory.TIMED_OUT
    assert diagnosis.evidence[-1].source is DiagnosisEvidenceSource.ERROR_MESSAGE


def test_diagnose_run_classifies_execution_errors_without_guessing_the_cause():
    diagnosis = diagnose_run(
        _result(
            status=RunStatus.ERROR,
            error_message="ConnectError: provider request failed",
        )
    )

    assert diagnosis.category is DiagnosisCategory.EXECUTION_ERROR
    assert "stopped before deterministic evaluation" in diagnosis.summary


def test_diagnose_run_prioritizes_policy_violation_over_failed_verification():
    violation = EvaluationViolation(
        kind=EvaluationViolationKind.FORBIDDEN_TOOL,
        message="task called forbidden tool 'network_fetch'",
        event_sequence=4,
    )
    diagnosis = diagnose_run(
        _result(
            status=RunStatus.FAILED,
            evaluation=EvaluationResult(
                passed=False,
                violations=(violation,),
            ),
            verification=(_failed_verification(),),
        )
    )

    assert diagnosis.category is DiagnosisCategory.POLICY_VIOLATION
    assert diagnosis.evidence[0].event_sequence == 4


def test_diagnose_run_identifies_a_missing_milestone():
    violation = EvaluationViolation(
        kind=EvaluationViolationKind.MISSING_MILESTONE,
        message="required milestone 'inspect-workspace' was not observed",
    )
    diagnosis = diagnose_run(
        _result(
            status=RunStatus.FAILED,
            evaluation=EvaluationResult(
                passed=False,
                violations=(violation,),
            ),
        )
    )

    assert diagnosis.category is DiagnosisCategory.MISSING_MILESTONE


def test_diagnose_run_identifies_failed_verification():
    diagnosis = diagnose_run(
        _result(
            status=RunStatus.FAILED,
            verification=(_failed_verification(),),
        )
    )

    assert diagnosis.category is DiagnosisCategory.VERIFICATION_FAILED
    assert "python -m pytest -q" in diagnosis.evidence[0].message


def test_diagnose_run_keeps_an_underspecified_failure_honest():
    diagnosis = diagnose_run(_result(status=RunStatus.FAILED))

    assert diagnosis.category is DiagnosisCategory.UNKNOWN_FAILURE
