import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import agent.benchmark.cli as benchmark_cli
from agent.benchmark.models import (
    BenchmarkRunResult,
    BenchmarkTask,
    CommandSpec,
    RunMetrics,
    RunStatus,
    TaskCategory,
    TrajectoryReference,
    VerificationResult,
)
from agent.benchmark.report import read_diagnosis_report


def _result(attempt: int, status: RunStatus) -> BenchmarkRunResult:
    now = datetime.now(timezone.utc)
    verification = (
        VerificationResult(
            command=CommandSpec(argv=("python", "-m", "pytest", "-q")),
            exit_code=0 if status is RunStatus.PASSED else 1,
            passed=status is RunStatus.PASSED,
        ),
    )
    return BenchmarkRunResult(
        run_id=f"fix-add-bug-{attempt:03d}",
        task_id="fix-add-bug",
        attempt=attempt,
        status=status,
        started_at=now,
        finished_at=now,
        metrics=RunMetrics(duration_seconds=Decimal("1")),
        trajectory=TrajectoryReference(
            relative_path=f"fix-add-bug-{attempt:03d}/trajectory.jsonl",
            event_count=1,
            sha256=f"{attempt:x}" * 64,
        ),
        verification=verification,
    )


class _FakeCatalog:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load_task(self, task_id: str) -> BenchmarkTask:
        assert task_id == "fix-add-bug"
        return BenchmarkTask(
            task_id=task_id,
            title="Fix an addition bug",
            prompt="Fix the bug.",
            category=TaskCategory.BUG_FIX,
            fixture="fixtures/fix-add-bug",
            verification=(CommandSpec(argv=("python", "-m", "pytest", "-q")),),
        )

    def fixture_path(self, task: BenchmarkTask) -> Path:
        return self.root / task.fixture


class _FakeVerifier:
    def __init__(self, image: str) -> None:
        self.image = image

    def ensure_available(self) -> None:
        return None

    async def run(self, command: CommandSpec, workspace: Path) -> object:
        raise AssertionError("the fake runner does not invoke the verifier")


class _FakeRunner:
    def __init__(self, **_: object) -> None:
        pass

    async def run_task(
        self,
        task: BenchmarkTask,
        config: object,
        *,
        keep_workspaces: bool,
    ) -> tuple[BenchmarkRunResult, ...]:
        del task, config, keep_workspaces
        return (
            _result(1, RunStatus.PASSED),
            _result(2, RunStatus.FAILED),
            _result(3, RunStatus.PASSED),
        )


def test_run_benchmark_writes_a_diagnosis_artifact(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        benchmark_cli,
        "read_git_provenance",
        lambda _: SimpleNamespace(revision="a" * 40),
    )
    monkeypatch.setattr(benchmark_cli, "require_clean_tree", lambda _: None)
    monkeypatch.setattr(benchmark_cli, "BenchmarkCatalog", _FakeCatalog)
    monkeypatch.setattr(
        benchmark_cli,
        "DockerCommandExecutor",
        _FakeVerifier,
    )
    monkeypatch.setattr(benchmark_cli, "BenchmarkRunner", _FakeRunner)

    async def unused_agent_attempt(workspace: Path, prompt: str):
        del workspace, prompt
        raise AssertionError("the fake runner does not invoke the agent")
        yield

    execution = asyncio.run(
        benchmark_cli.run_benchmark(
            project_root=tmp_path,
            benchmark_root=tmp_path / "benchmarks",
            results_root=tmp_path / "results",
            task_id="fix-add-bug",
            provider="openai",
            model="gpt-5.6-terra",
            container_image="example.test/python@sha256:" + "b" * 64,
            attempts=3,
            parallelism=1,
            agent_settings={},
            agent_attempt=unused_agent_attempt,
        )
    )

    assert execution.diagnosis_path.exists()
    assert execution.diagnosis_path.with_suffix(".json").exists()
    assert read_diagnosis_report(
        execution.diagnosis_path.with_suffix(".json")
    ) == execution.diagnosis_report
    assert execution.diagnosis_report.summary.failed_runs == 1
