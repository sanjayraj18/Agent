"""Deterministically explain completed benchmark attempts from saved evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from decimal import Decimal

from agent.benchmark.models import (
    BenchmarkRunResult,
    DiagnosisCategory,
    DiagnosisEvidence,
    DiagnosisEvidenceSource,
    DiagnosisSummary,
    EvaluationViolation,
    EvaluationViolationKind,
    FailurePattern,
    RunDiagnosis,
    RunStatus,
    VerificationResult,
)


class DiagnosisError(ValueError):
    """The supplied diagnosis records cannot form one trustworthy summary."""


def diagnose_run(result: BenchmarkRunResult) -> RunDiagnosis:
    """
    Classify one completed run without changing it or calling a model.

    The ordering is deliberate: a timeout or execution error happened before
    evaluation; policy violations take precedence over a later test failure.
    """
    if result.status is RunStatus.PASSED:
        return _diagnosis(
            result,
            category=DiagnosisCategory.PASSED,
            summary="The run satisfied every evaluation rule.",
            evidence=(_status_evidence(result),),
            suggested_next_step="No corrective action is needed.",
        )

    if result.status is RunStatus.TIMED_OUT:
        return _diagnosis(
            result,
            category=DiagnosisCategory.TIMED_OUT,
            summary="The agent did not finish before the task timeout.",
            evidence=_status_and_error_evidence(result),
            suggested_next_step=(
                "Inspect the trajectory near its final event and determine "
                "whether the agent was stalled or the timeout is too small."
            ),
        )

    if result.status is RunStatus.ERROR:
        return _diagnosis(
            result,
            category=DiagnosisCategory.EXECUTION_ERROR,
            summary="The attempt stopped before deterministic evaluation.",
            evidence=_status_and_error_evidence(result),
            suggested_next_step=(
                "Inspect the recorded error message and the final trajectory "
                "event before changing agent behavior."
            ),
        )

    policy_violation = _first_policy_violation(result)
    if policy_violation is not None:
        return _diagnosis(
            result,
            category=DiagnosisCategory.POLICY_VIOLATION,
            summary="The run violated the task's safety or file-boundary policy.",
            evidence=(_violation_evidence(policy_violation),),
            suggested_next_step=(
                "Inspect the policy violation and the linked tool event before "
                "changing prompts or permissions."
            ),
        )

    missing_milestone = _first_missing_milestone(result)
    if missing_milestone is not None:
        return _diagnosis(
            result,
            category=DiagnosisCategory.MISSING_MILESTONE,
            summary="The run omitted required observable task behavior.",
            evidence=(_violation_evidence(missing_milestone),),
            suggested_next_step=(
                "Inspect the required milestone and trajectory to see why the "
                "agent skipped it."
            ),
        )

    failed_verification = _first_failed_verification(result)
    if failed_verification is not None:
        return _diagnosis(
            result,
            category=DiagnosisCategory.VERIFICATION_FAILED,
            summary="The workspace did not satisfy a verification command.",
            evidence=(_verification_evidence(failed_verification),),
            suggested_next_step=(
                "Read the failed verification output and compare it with the "
                "files changed during the run."
            ),
        )

    return _diagnosis(
        result,
        category=DiagnosisCategory.UNKNOWN_FAILURE,
        summary="The run failed without enough recorded evidence for a narrower cause.",
        evidence=_status_and_error_evidence(result),
        suggested_next_step=(
            "Inspect the full trajectory and add deterministic evidence before "
            "drawing a stronger conclusion."
        ),
    )


def summarize_diagnoses(
    diagnoses: Iterable[RunDiagnosis],
) -> DiagnosisSummary:
    """Aggregate repeated attempts of exactly one task into failure patterns."""
    values = tuple(diagnoses)
    if not values:
        raise DiagnosisError("at least one diagnosis is required")

    task_ids = {diagnosis.task_id for diagnosis in values}
    if len(task_ids) != 1:
        raise DiagnosisError("a diagnosis summary may contain only one task")

    run_ids = [diagnosis.run_id for diagnosis in values]
    if len(run_ids) != len(set(run_ids)):
        raise DiagnosisError("a diagnosis summary cannot contain duplicate run IDs")

    passed_runs = sum(
        diagnosis.category is DiagnosisCategory.PASSED
        for diagnosis in values
    )
    failed_runs = len(values) - passed_runs
    counts = Counter(
        diagnosis.category
        for diagnosis in values
        if diagnosis.category is not DiagnosisCategory.PASSED
    )

    patterns = tuple(
        FailurePattern(
            category=category,
            count=count,
            share_of_failed_runs=Decimal(count) / Decimal(failed_runs),
            share_of_all_runs=Decimal(count) / Decimal(len(values)),
        )
        for category, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0].value),
        )
    )

    return DiagnosisSummary(
        task_id=values[0].task_id,
        total_runs=len(values),
        passed_runs=passed_runs,
        failed_runs=failed_runs,
        failure_patterns=patterns,
    )


def _diagnosis(
    result: BenchmarkRunResult,
    *,
    category: DiagnosisCategory,
    summary: str,
    evidence: tuple[DiagnosisEvidence, ...],
    suggested_next_step: str,
) -> RunDiagnosis:
    return RunDiagnosis(
        run_id=result.run_id,
        task_id=result.task_id,
        attempt=result.attempt,
        run_status=result.status,
        category=category,
        summary=summary,
        evidence=evidence,
        suggested_next_step=suggested_next_step,
    )


def _status_evidence(result: BenchmarkRunResult) -> DiagnosisEvidence:
    return DiagnosisEvidence(
        source=DiagnosisEvidenceSource.RUN_STATUS,
        message=f"run status: {result.status.value}",
    )


def _status_and_error_evidence(
    result: BenchmarkRunResult,
) -> tuple[DiagnosisEvidence, ...]:
    evidence = [_status_evidence(result)]
    if result.error_message is not None and result.error_message.strip():
        evidence.append(
            DiagnosisEvidence(
                source=DiagnosisEvidenceSource.ERROR_MESSAGE,
                message=result.error_message,
            )
        )
    return tuple(evidence)


def _first_policy_violation(
    result: BenchmarkRunResult,
) -> EvaluationViolation | None:
    if result.evaluation is None:
        return None

    return next(
        (
            violation
            for violation in result.evaluation.violations
            if violation.kind
            in {
                EvaluationViolationKind.FILE_BOUNDARY,
                EvaluationViolationKind.FORBIDDEN_TOOL,
            }
        ),
        None,
    )


def _first_missing_milestone(
    result: BenchmarkRunResult,
) -> EvaluationViolation | None:
    if result.evaluation is None:
        return None

    return next(
        (
            violation
            for violation in result.evaluation.violations
            if violation.kind is EvaluationViolationKind.MISSING_MILESTONE
        ),
        None,
    )


def _first_failed_verification(
    result: BenchmarkRunResult,
) -> VerificationResult | None:
    return next(
        (verification for verification in result.verification if not verification.passed),
        None,
    )


def _violation_evidence(
    violation: EvaluationViolation,
) -> DiagnosisEvidence:
    return DiagnosisEvidence(
        source=DiagnosisEvidenceSource.EVALUATION_VIOLATION,
        message=violation.message,
        event_sequence=violation.event_sequence,
    )


def _verification_evidence(
    verification: VerificationResult,
) -> DiagnosisEvidence:
    command = " ".join(verification.command.argv)
    return DiagnosisEvidence(
        source=DiagnosisEvidenceSource.VERIFICATION,
        message=(
            f"verification command {command!r} exited with code "
            f"{verification.exit_code}"
        ),
    )
