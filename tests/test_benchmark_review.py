from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from agent.benchmark.cli import review_saved_benchmark_run
from agent.benchmark.diagnosis import build_diagnosis_report, diagnose_run
from agent.benchmark.models import BenchmarkRunResult, RunMetrics, RunStatus
from agent.benchmark.report import (
    read_run_results,
    write_diagnosis_report,
    write_run_results,
)
from agent.benchmark.trajectory import TrajectoryWriter
from agent.events import AssistantStart, ThinkingDelta, UserMessage
from agent.providers.base import EventFactory


def test_saved_review_loads_the_exact_run_and_writes_safe_artifacts(
    tmp_path: Path,
):
    trajectories_root = tmp_path / "trajectories"
    writer = TrajectoryWriter(trajectories_root, "fix-add-bug-001")
    emit = EventFactory("session")
    writer.append(emit(UserMessage, text="sensitive benchmark prompt"))
    writer.append(emit(AssistantStart))
    writer.append(emit(ThinkingDelta, index=0, text="private thought"))
    reference = writer.close()
    now = datetime.now(timezone.utc)
    result = BenchmarkRunResult(
        run_id="fix-add-bug-001",
        task_id="fix-add-bug",
        attempt=1,
        status=RunStatus.ERROR,
        started_at=now,
        finished_at=now,
        metrics=RunMetrics(duration_seconds=Decimal("1")),
        trajectory=reference,
        error_message="agent process exited unexpectedly",
    )
    diagnosis_report = build_diagnosis_report(
        (diagnose_run(result),),
        config_fingerprint="a" * 64,
    )
    results_path = tmp_path / "results.json"
    diagnosis_path = tmp_path / "diagnosis.md"
    output_path = tmp_path / "review.md"
    write_run_results(results_path, (result,))
    write_diagnosis_report(diagnosis_path, diagnosis_report)

    execution = review_saved_benchmark_run(
        results_path=results_path,
        diagnosis_path=diagnosis_path.with_suffix(".json"),
        trajectories_root=trajectories_root,
        run_id=result.run_id,
        output_path=output_path,
    )

    markdown = output_path.read_text(encoding="utf-8")
    assert execution.review.run == result
    assert read_run_results(results_path) == (result,)
    assert output_path.with_suffix(".json").exists()
    assert "# Agent benchmark quality review" in markdown
    assert "Private reasoning events omitted: 1." in markdown
    assert "sensitive benchmark prompt" not in markdown
    assert "private thought" not in markdown
