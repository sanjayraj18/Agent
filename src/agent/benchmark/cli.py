"""Application-facing orchestration for `agent bench` commands."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.benchmark.catalog import BenchmarkCatalog
from agent.benchmark.comparison import (
    BenchmarkComparison,
    compare_scoreboard_rows,
)
from agent.benchmark.diagnosis import build_diagnosis_report, diagnose_run
from agent.benchmark.gate import (
    evaluate_regression_gate,
    load_gate_policy,
    load_quality_snapshot,
)
from agent.benchmark.git_info import read_git_provenance, require_clean_tree
from agent.benchmark.images import DockerCommandExecutor
from agent.benchmark.models import (
    BenchmarkRunConfig,
    BenchmarkRunResult,
    DiagnosisReport,
    QualityReview,
    RegressionGateResult,
    ScoreboardRow,
)
from agent.benchmark.report import (
    read_diagnosis_report,
    read_run_results,
    read_scoreboard,
    write_diagnosis_report,
    write_quality_review,
    write_regression_gate,
    write_comparison,
    write_run_results,
    write_scoreboard,
)
from agent.benchmark.review import build_quality_review
from agent.benchmark.runner import AgentAttempt, BenchmarkRunner
from agent.benchmark.scoreboard import build_scoreboard_row
from agent.events import Event


@dataclass(frozen=True, slots=True)
class BenchmarkExecution:
    results: tuple[BenchmarkRunResult, ...]
    scoreboard: ScoreboardRow
    diagnosis_report: DiagnosisReport
    results_path: Path
    scoreboard_path: Path
    diagnosis_path: Path


@dataclass(frozen=True, slots=True)
class BenchmarkComparisonExecution:
    """Saved evidence produced by comparing several benchmark strategies."""

    comparison: BenchmarkComparison
    comparison_path: Path


@dataclass(frozen=True, slots=True)
class BenchmarkGateExecution:
    """Saved evidence produced by a baseline-versus-candidate gate."""

    result: RegressionGateResult
    gate_path: Path


@dataclass(frozen=True, slots=True)
class BenchmarkReviewExecution:
    """Saved evidence produced by replaying one completed benchmark attempt."""

    review: QualityReview
    review_path: Path


async def run_benchmark(
    *,
    project_root: Path,
    benchmark_root: Path,
    results_root: Path,
    task_id: str,
    provider: str,
    model: str,
    container_image: str,
    attempts: int,
    parallelism: int,
    agent_settings: dict[str, Any],
    agent_attempt: AgentAttempt,
    route_id: str = "unrouted",
    strategy_id: str | None = None,
    keep_workspaces: bool = False,
) -> BenchmarkExecution:
    """Run one catalog task repeatedly and preserve its evidence."""
    provenance = read_git_provenance(project_root)
    require_clean_tree(provenance)

    if agent_settings.get("sandbox_mode") == "disabled":
        raise ValueError(
            "benchmarks refuse sandbox_mode='disabled'; use an enforced sandbox"
        )

    resolved_strategy_id = (
        strategy_id.strip()
        if strategy_id is not None
        else f"fixed-{route_id}"
    )

    if not resolved_strategy_id:
        raise ValueError("strategy_id must not be blank")

    config = BenchmarkRunConfig(
        provider=provider,
        model=model,
        route_id=route_id,
        strategy_id=resolved_strategy_id,
        agent_revision=provenance.revision,
        container_image=container_image,
        attempts=attempts,
        parallelism=parallelism,
        agent_settings=agent_settings,
    )
    catalog = BenchmarkCatalog(benchmark_root)
    task = catalog.load_task(task_id)
    catalog.fixture_path(task)

    verifier = DockerCommandExecutor(config.container_image)
    verifier.ensure_available()
    runner = BenchmarkRunner(
        fixture_root=benchmark_root,
        workspace_root=results_root / "workspaces",
        results_root=results_root / "trajectories",
        agent_attempt=agent_attempt,
        command_runner=verifier.run,
    )
    results = await runner.run_task(
        task,
        config,
        keep_workspaces=keep_workspaces,
    )
    scoreboard = build_scoreboard_row(results, config)

    result_directory = results_root / "runs" / task.task_id / config.fingerprint
    results_path = result_directory / "results.json"
    diagnosis_path = result_directory / "diagnosis.md"
    scoreboard_path = results_root / "scoreboard.md"
    diagnosis_report = build_diagnosis_report(
        (diagnose_run(result) for result in results),
        config_fingerprint=config.fingerprint,
    )
    write_run_results(results_path, results)
    write_diagnosis_report(diagnosis_path, diagnosis_report)
    scoreboard_json = scoreboard_path.with_suffix(".json")
    existing_rows = read_scoreboard(scoreboard_json) if scoreboard_json.exists() else ()
    row_key = (
        scoreboard.task_id,
        scoreboard.provider,
        scoreboard.model,
        scoreboard.agent_revision,
        scoreboard.config_fingerprint,
    )
    merged_rows = tuple(
        row
        for row in existing_rows
        if (
            row.task_id,
            row.provider,
            row.model,
            row.agent_revision,
            row.config_fingerprint,
        ) != row_key
    ) + (scoreboard,)
    write_scoreboard(scoreboard_path, merged_rows)

    return BenchmarkExecution(
        results=results,
        scoreboard=scoreboard,
        diagnosis_report=diagnosis_report,
        results_path=results_path,
        scoreboard_path=scoreboard_path,
        diagnosis_path=diagnosis_path,
    )


def compare_benchmark_scoreboards(
    *,
    scoreboard_paths: Iterable[Path],
    baseline_strategy_id: str,
    output_path: Path,
) -> BenchmarkComparisonExecution:
    """
    Load one or more saved scoreboards, compare their rows, and save evidence.

    Multiple paths allow us to compare runs saved under separate result roots,
    such as benchmarks/results/strong-only and benchmarks/results/live-router.
    """
    paths = tuple(
        path.expanduser().resolve()
        for path in scoreboard_paths
    )

    if not paths:
        raise ValueError("at least one scoreboard path is required")

    rows = tuple(
        row
        for path in paths
        for row in read_scoreboard(path)
    )

    if not rows:
        raise ValueError("the supplied scoreboards contain no rows")

    comparison = compare_scoreboard_rows(
        rows,
        baseline_strategy_id=baseline_strategy_id,
    )

    resolved_output_path = output_path.expanduser().resolve()
    write_comparison(resolved_output_path, comparison)

    return BenchmarkComparisonExecution(
        comparison=comparison,
        comparison_path=resolved_output_path,
    )


def evaluate_saved_regression_gate(
    *,
    baseline_scoreboard_path: Path,
    baseline_diagnosis_path: Path,
    candidate_scoreboard_path: Path,
    candidate_diagnosis_path: Path,
    policy_path: Path,
    output_path: Path,
) -> BenchmarkGateExecution:
    """Load immutable artifacts, gate the candidate, and save the decision."""
    policy = load_gate_policy(policy_path.expanduser().resolve())
    baseline = load_quality_snapshot(
        scoreboard_path=baseline_scoreboard_path.expanduser().resolve(),
        diagnosis_path=baseline_diagnosis_path.expanduser().resolve(),
    )
    candidate = load_quality_snapshot(
        scoreboard_path=candidate_scoreboard_path.expanduser().resolve(),
        diagnosis_path=candidate_diagnosis_path.expanduser().resolve(),
    )
    result = evaluate_regression_gate(
        baseline=baseline,
        candidate=candidate,
        policy=policy,
    )
    resolved_output_path = output_path.expanduser().resolve()
    write_regression_gate(resolved_output_path, result)

    return BenchmarkGateExecution(
        result=result,
        gate_path=resolved_output_path,
    )


def review_saved_benchmark_run(
    *,
    results_path: Path,
    diagnosis_path: Path,
    trajectories_root: Path,
    run_id: str,
    output_path: Path,
) -> BenchmarkReviewExecution:
    """Replay one saved attempt without rerunning the agent or its tools."""
    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise ValueError("run_id must not be blank")

    results = read_run_results(results_path.expanduser().resolve())
    match = next(
        (
            result
            for result in results
            if result.run_id == normalized_run_id
        ),
        None,
    )
    if match is None:
        raise ValueError(
            f"results do not contain run_id {normalized_run_id!r}"
        )

    diagnosis_report = read_diagnosis_report(
        diagnosis_path.expanduser().resolve()
    )
    review = build_quality_review(
        result=match,
        diagnosis_report=diagnosis_report,
        trajectories_root=trajectories_root.expanduser().resolve(),
    )
    resolved_output_path = output_path.expanduser().resolve()
    write_quality_review(resolved_output_path, review)

    return BenchmarkReviewExecution(
        review=review,
        review_path=resolved_output_path,
    )
