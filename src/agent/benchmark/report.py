"""Write JSON evidence and a readable Markdown benchmark scoreboard."""

from __future__ import annotations

import json
from pathlib import Path

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
        "| Task | Route | Provider / model | Runs | Passed | Pass rate | "
        "Mean cost | Mean tokens | Mean turns | Mean duration |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        cost = "unknown" if row.mean_cost_usd is None else f"${row.mean_cost_usd:.5f}"
        lines.append(
            "| "
            f"{row.task_id} | {row.provider} / {row.model} | "
            f"{row.provider} / {row.model} | "
            f"{row.attempts} | {row.passed_attempts} | "
            f"{row.pass_rate:.2%} | {cost} | "
            f"{row.mean_tokens:.0f} | {row.mean_turns:.2f} | "
            f"{row.mean_duration_seconds:.2f}s |"
        )

    lines.extend(
        [
            "",
            "Each row is tied to an agent commit SHA, pinned image digest, "
            "and full configuration fingerprint in the companion JSON file.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
