from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from agent.benchmark.comparison import compare_scoreboard_rows
from agent.benchmark.models import ModelMixEntry, ScoreboardRow
from agent.benchmark.report import (
    read_scoreboard,
    render_comparison_markdown,
    render_markdown,
    write_comparison,
    write_scoreboard,
)


def _row(
    *,
    strategy_id: str = "fixed-strong",
    route_id: str = "strong-terra-high",
    model: str = "gpt-5.6-terra",
    mean_cost_usd: Decimal | None = Decimal("0.01"),
    mean_duration_seconds: Decimal = Decimal("1.5"),
    model_mix: tuple[ModelMixEntry, ...] | None = None,
) -> ScoreboardRow:
    if model_mix is None:
        model_mix = (
            ModelMixEntry(
                provider="openai",
                model=model,
                route_id=route_id,
                turns=6,
                input_tokens=600,
                output_tokens=120,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                cost_usd=mean_cost_usd,
            ),
        )

    return ScoreboardRow(
        task_id="fix-add-bug",
        provider="openai",
        model=model,
        route_id=route_id,
        strategy_id=strategy_id,
        model_mix=model_mix,
        agent_revision="a" * 40,
        container_image="example.test/python@sha256:" + "b" * 64,
        config_fingerprint="c" * 64,
        attempts=3,
        passed_attempts=2,
        pass_rate=Decimal(2) / Decimal(3),
        mean_duration_seconds=mean_duration_seconds,
        duration_standard_deviation_seconds=Decimal("0.5"),
        mean_tokens=Decimal("100"),
        token_standard_deviation=Decimal("2"),
        mean_turns=Decimal("3"),
        mean_cost_usd=mean_cost_usd,
        cost_standard_deviation_usd=(
            None if mean_cost_usd is None else Decimal("0.001")
        ),
    )


def test_scoreboard_writes_json_and_readable_markdown(tmp_path: Path):
    markdown_path = tmp_path / "scoreboard.md"
    write_scoreboard(markdown_path, (_row(),))

    assert "fix-add-bug" in markdown_path.read_text(encoding="utf-8")
    assert read_scoreboard(markdown_path.with_suffix(".json")) == (_row(),)
    assert "Pass rate" in render_markdown((_row(),))


def test_scoreboard_markdown_distinguishes_strategy_route_and_actual_mix():
    markdown = render_markdown((_row(),))

    assert "| Task | Strategy | Route |" in markdown
    assert (
        "| fix-add-bug | fixed-strong | strong-terra-high | "
        "openai / gpt-5.6-terra |"
    ) in markdown
    assert "openai / gpt-5.6-terra × 6 (strong-terra-high)" in markdown


def test_comparison_markdown_shows_deltas_and_writes_json_evidence(
    tmp_path: Path,
):
    baseline = _row()
    economy = _row(
        strategy_id="fixed-economy",
        route_id="economy-luna-medium",
        model="gpt-5.6-luna",
        mean_cost_usd=Decimal("0.002"),
        mean_duration_seconds=Decimal("1.0"),
    )
    comparison = compare_scoreboard_rows(
        (baseline, economy),
        baseline_strategy_id="fixed-strong",
    )

    markdown = render_comparison_markdown(comparison)
    output_path = tmp_path / "comparison.md"
    write_comparison(output_path, comparison)

    assert "# Agent benchmark strategy comparison" in markdown
    assert "dominates baseline" in markdown
    assert "-$0.00800" in markdown
    assert "+80.00%" in markdown
    assert "openai / gpt-5.6-luna × 6 (economy-luna-medium)" in markdown
    assert output_path.read_text(encoding="utf-8") == markdown
    assert '"baseline"' in output_path.with_suffix(".json").read_text(
        encoding="utf-8"
    )
