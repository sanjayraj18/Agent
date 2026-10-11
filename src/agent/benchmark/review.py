"""Build a focused engineering review from immutable benchmark evidence."""

from __future__ import annotations

from pathlib import Path

from agent.benchmark.models import (
    BenchmarkRunResult,
    DiagnosisReport,
    QualityReview,
)
from agent.benchmark.replay import replay_trajectory
from agent.benchmark.trajectory import TrajectoryError


class ReviewError(ValueError):
    """Saved benchmark artifacts cannot support one trustworthy review."""


def build_quality_review(
    *,
    result: BenchmarkRunResult,
    diagnosis_report: DiagnosisReport,
    trajectories_root: Path,
) -> QualityReview:
    """
    Join one run, its deterministic diagnosis, and its verified timeline.

    This function is intentionally read-only: it makes no model request, does
    not invoke a tool, and never changes the benchmark workspace.
    """
    if diagnosis_report.summary.task_id != result.task_id:
        raise ReviewError(
            "diagnosis report task ID does not match the selected run"
        )

    match = next(
        (
            diagnosis
            for diagnosis in diagnosis_report.diagnoses
            if diagnosis.run_id == result.run_id
        ),
        None,
    )
    if match is None:
        raise ReviewError(
            "diagnosis report does not contain the selected run "
            f"{result.run_id!r}"
        )

    try:
        replay = replay_trajectory(
            result,
            trajectories_root=trajectories_root,
        )
    except TrajectoryError as exc:
        raise ReviewError(
            f"could not replay trajectory for {result.run_id!r}: {exc}"
        ) from exc

    return QualityReview(
        run=result,
        diagnosis=match,
        replay=replay,
    )
