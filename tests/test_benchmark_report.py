from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from agent.benchmark.models import ScoreboardRow
from agent.benchmark.report import read_scoreboard, render_markdown, write_scoreboard


def _row() -> ScoreboardRow:
    return ScoreboardRow(
        task_id="fix-add-bug",
        provider="openai",
        model="gpt-5.6-terra",
        agent_revision="a" * 40,
        container_image="example.test/python@sha256:" + "b" * 64,
        config_fingerprint="c" * 64,
        attempts=3,
        passed_attempts=2,
        pass_rate=Decimal(2) / Decimal(3),
        mean_duration_seconds=Decimal("1.5"),
        duration_standard_deviation_seconds=Decimal("0.5"),
        mean_tokens=Decimal("100"),
        token_standard_deviation=Decimal("2"),
        mean_turns=Decimal("3"),
        mean_cost_usd=Decimal("0.01"),
        cost_standard_deviation_usd=Decimal("0.001"),
    )


def test_scoreboard_writes_json_and_readable_markdown(tmp_path: Path):
    markdown_path = tmp_path / "scoreboard.md"
    write_scoreboard(markdown_path, (_row(),))

    assert "fix-add-bug" in markdown_path.read_text(encoding="utf-8")
    assert read_scoreboard(markdown_path.with_suffix(".json")) == (_row(),)
    assert "Pass rate" in render_markdown((_row(),))
