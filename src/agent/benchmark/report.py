"""Write JSON evidence and a readable Markdown benchmark scoreboard."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from agent.benchmark.comparison import BenchmarkComparison
from agent.benchmark.models import BenchmarkRunResult, ScoreboardRow


def write_run_results(
    path: Path,
    results: tuple[BenchmarkRunResult, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {"runs": [result.model_dump(mode="json") for result in results]},
    )


def write_scoreboard(path: Path, rows: tuple[ScoreboardRow, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path.with_suffix(".json"),
        {"rows": [row.model_dump(mode="json") for row in rows]},
    )
    path.write_text(render_markdown(rows), encoding="utf-8")


def write_comparison(
    path: Path,
    comparison: BenchmarkComparison,
) -> None:
    """Write the machine-readable and human-readable comparison evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path.with_suffix(".json"),
        comparison.model_dump(mode="json"),
    )
    path.write_text(
        render_comparison_markdown(comparison),
        encoding="utf-8",
    )


def read_scoreboard(path: Path) -> tuple[ScoreboardRow, ...]:
    """Load the JSON scoreboard used to regenerate a Markdown report."""
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read scoreboard {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid scoreboard JSON ({exc})") from exc

    values = decoded.get("rows") if isinstance(decoded, dict) else None
    if not isinstance(values, list):
        raise ValueError(f"{path}: scoreboard must contain a rows list")

    try:
        return tuple(ScoreboardRow.model_validate(value) for value in values)
    except Exception as exc:
        raise ValueError(f"{path}: invalid scoreboard row ({exc})") from exc


def render_markdown(rows: tuple[ScoreboardRow, ...]) -> str:
    lines = [
        "# Agent benchmark scoreboard",
        "",
        "| Task | Strategy | Route | Configured provider / model | "
        "Actual model mix | Runs | Passed | Pass rate | Mean cost | "
        "Mean tokens | Mean turns | Mean duration |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(_scoreboard_line(row))

    lines.extend(
        [
            "",
            "Each row is tied to an agent commit SHA, pinned image digest, "
            "and full configuration fingerprint in the companion JSON file.",
            "",
        ]
    )
    return "\n".join(lines)


def render_comparison_markdown(
    comparison: BenchmarkComparison,
) -> str:
    """Render one baseline and each candidate strategy for human review."""
    baseline = comparison.baseline
    lines = [
        "# Agent benchmark strategy comparison",
        "",
        f"Task: `{comparison.task_id}`",
        "",
        "## Baseline",
        "",
        "| Strategy | Route | Configured provider / model | Pass rate | "
        "Mean cost | Mean duration | Actual model mix |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
        _comparison_baseline_line(baseline),
        "",
        "## Compared strategies",
        "",
        "| Strategy | Configured provider / model | Actual model mix | "
        "Pass-rate Δ | Cost Δ | Savings | Duration Δ | Result | "
        "Pareto-optimal |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | "
        "--- | --- |",
    ]

    for candidate in comparison.candidates:
        row = candidate.candidate
        lines.append(
            "| "
            f"{row.strategy_id} | {row.provider} / {row.model} | "
            f"{_format_model_mix(row)} | "
            f"{candidate.pass_rate_delta:+.2%} | "
            f"{_format_cost_delta(candidate.mean_cost_delta_usd)} | "
            f"{_format_savings(candidate.cost_savings_fraction)} | "
            f"{candidate.mean_duration_delta_seconds:+.2f}s | "
            f"{candidate.relation_to_baseline.value.replace('_', ' ')} | "
            f"{_format_pareto(candidate.pareto_optimal)} |"
        )

    lines.append("")
    return "\n".join(lines)


def _scoreboard_line(row: ScoreboardRow) -> str:
    return (
        "| "
        f"{row.task_id} | {row.strategy_id} | {row.route_id} | "
        f"{row.provider} / {row.model} | {_format_model_mix(row)} | "
        f"{row.attempts} | {row.passed_attempts} | "
        f"{row.pass_rate:.2%} | {_format_cost(row.mean_cost_usd)} | "
        f"{row.mean_tokens:.0f} | {row.mean_turns:.2f} | "
        f"{row.mean_duration_seconds:.2f}s |"
    )


def _comparison_baseline_line(row: ScoreboardRow) -> str:
    return (
        "| "
        f"{row.strategy_id} | {row.route_id} | "
        f"{row.provider} / {row.model} | {row.pass_rate:.2%} | "
        f"{_format_cost(row.mean_cost_usd)} | "
        f"{row.mean_duration_seconds:.2f}s | "
        f"{_format_model_mix(row)} |"
    )


def _format_model_mix(row: ScoreboardRow) -> str:
    if not row.model_mix:
        return "not recorded"

    return "<br>".join(
        (
            f"{entry.provider} / {entry.model} × {entry.turns} "
            f"({entry.route_id or 'unrouted'})"
        )
        for entry in row.model_mix
    )


def _format_cost(value: Decimal | None) -> str:
    return "unknown" if value is None else f"${value:.5f}"


def _format_cost_delta(value: Decimal | None) -> str:
    if value is None:
        return "unknown"

    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.5f}"


def _format_savings(value: Decimal | None) -> str:
    return "unknown" if value is None else f"{value:+.2%}"


def _format_pareto(value: bool | None) -> str:
    if value is None:
        return "unknown"

    return "yes" if value else "no"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
