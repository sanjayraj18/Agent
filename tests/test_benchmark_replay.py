from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from agent.benchmark.diagnosis import build_diagnosis_report, diagnose_run
from agent.benchmark.models import (
    BenchmarkRunResult,
    EvaluationResult,
    EvaluationViolation,
    EvaluationViolationKind,
    RunMetrics,
    RunStatus,
)
from agent.benchmark.replay import replay_trajectory
from agent.benchmark.review import ReviewError, build_quality_review
from agent.benchmark.trajectory import TrajectoryWriter
from agent.events import (
    AssistantStart,
    ErrorEvent,
    ThinkingDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResult,
    UserMessage,
)
from agent.providers.base import EventFactory


def _recorded_run(tmp_path: Path) -> tuple[BenchmarkRunResult, Path]:
    trajectories_root = tmp_path / "trajectories"
    writer = TrajectoryWriter(trajectories_root, "fix-add-bug-001")
    emit = EventFactory("session")
    events = (
        emit(UserMessage, text="private user prompt"),
        emit(AssistantStart),
        emit(ThinkingDelta, index=0, text="private reasoning"),
        emit(
            ToolCallStart,
            index=0,
            call_id="call-1",
            name="edit_file",
        ),
        emit(
            ToolCallEnd,
            index=0,
            call_id="call-1",
            arguments={"path": "private.txt"},
        ),
        emit(
            ToolResult,
            call_id="call-1",
            content="private tool output",
            is_error=True,
        ),
        emit(
            ErrorEvent,
            kind="tool_failure",
            message="private runtime error",
        ),
    )
    for event in events:
        writer.append(event)
    reference = writer.close()
    now = datetime.now(timezone.utc)
    result = BenchmarkRunResult(
        run_id="fix-add-bug-001",
        task_id="fix-add-bug",
        attempt=1,
        status=RunStatus.FAILED,
        started_at=now,
        finished_at=now,
        metrics=RunMetrics(duration_seconds=Decimal("1")),
        trajectory=reference,
        evaluation=EvaluationResult(
            passed=False,
            violations=(
                EvaluationViolation(
                    kind=EvaluationViolationKind.FORBIDDEN_TOOL,
                    message="task called forbidden tool 'edit_file'",
                    event_sequence=4,
                ),
            ),
        ),
    )
    return result, trajectories_root


def test_replay_verifies_the_trajectory_and_suppresses_private_payloads(
    tmp_path: Path,
):
    result, trajectories_root = _recorded_run(tmp_path)

    replay = replay_trajectory(result, trajectories_root=trajectories_root)

    assert replay.reference == result.trajectory
    assert replay.suppressed_private_event_count == 1
    assert [event.sequence for event in replay.events] == [1, 2, 4, 5, 6, 7]
    assert replay.events[2].summary == "Tool `edit_file` call started."
    rendered = "\n".join(event.summary for event in replay.events)
    assert "private user prompt" not in rendered
    assert "private reasoning" not in rendered
    assert "private.txt" not in rendered
    assert "private tool output" not in rendered
    assert "private runtime error" not in rendered


def test_quality_review_links_diagnosis_evidence_to_the_safe_timeline(
    tmp_path: Path,
):
    result, trajectories_root = _recorded_run(tmp_path)
    report = build_diagnosis_report(
        (diagnose_run(result),),
        config_fingerprint="a" * 64,
    )

    review = build_quality_review(
        result=result,
        diagnosis_report=report,
        trajectories_root=trajectories_root,
    )

    assert review.diagnosis.evidence[0].event_sequence == 4
    assert review.replay.events[2].sequence == 4


def test_quality_review_rejects_a_tampered_trajectory(tmp_path: Path):
    result, trajectories_root = _recorded_run(tmp_path)
    report = build_diagnosis_report(
        (diagnose_run(result),),
        config_fingerprint="a" * 64,
    )
    trajectory_path = trajectories_root / result.trajectory.relative_path
    trajectory_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ReviewError, match="SHA-256"):
        build_quality_review(
            result=result,
            diagnosis_report=report,
            trajectories_root=trajectories_root,
        )
