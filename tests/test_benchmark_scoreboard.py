from datetime import datetime, timezone
from decimal import Decimal

from agent.benchmark.models import (
    BenchmarkRunConfig,
    BenchmarkRunResult,
    CommandSpec,
    RunMetrics,
    RunStatus,
    TrajectoryReference,
    VerificationResult,
)
from agent.benchmark.scoreboard import build_scoreboard_row


def _image() -> str:
    return "example.test/python@sha256:" + "a" * 64


def _result(attempt: int, status: RunStatus) -> BenchmarkRunResult:
    verification = (
        VerificationResult(
            command=CommandSpec(argv=("python", "-V")),
            exit_code=0,
            passed=True,
        ),
    ) if status is RunStatus.PASSED else ()
    now = datetime.now(timezone.utc)
    return BenchmarkRunResult(
        run_id=f"run-{attempt}",
        task_id="fix-add-bug",
        attempt=attempt,
        status=status,
        started_at=now,
        finished_at=now,
        metrics=RunMetrics(
            duration_seconds=Decimal(attempt),
            input_tokens=10,
            output_tokens=5,
            turns=2,
            cost_usd=Decimal("0.01"),
        ),
        trajectory=TrajectoryReference(
            relative_path=f"run-{attempt}/trajectory.jsonl",
            event_count=0,
            sha256="b" * 64,
        ),
        verification=verification,
    )


def test_scoreboard_calculates_pass_rate_and_variance():
    config = BenchmarkRunConfig(
        provider="openai",
        model="gpt-5.6-terra",
        agent_revision="a" * 40,
        container_image=_image(),
    )
    row = build_scoreboard_row(
        (
            _result(1, RunStatus.PASSED),
            _result(2, RunStatus.FAILED),
            _result(3, RunStatus.PASSED),
        ),
        config,
    )

    assert row.pass_rate == Decimal(2) / Decimal(3)
    assert row.duration_standard_deviation_seconds > 0
    assert row.mean_cost_usd == Decimal("0.01")
