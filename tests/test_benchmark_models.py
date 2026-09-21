from decimal import Decimal

import pytest

from agent.benchmark.models import (
    BenchmarkRunConfig,
    CommandSpec,
    ScoreboardRow,
    TrajectoryEntry,
)
from agent.events import UserMessage
from agent.providers.base import EventFactory


def _image() -> str:
    return "example.test/python@sha256:" + "a" * 64


def test_config_requires_three_attempts_and_a_pinned_image():
    with pytest.raises(ValueError, match="greater than or equal to 3"):
        BenchmarkRunConfig(
            provider="openai",
            model="gpt-5.6-terra",
            agent_revision="a" * 40,
            container_image=_image(),
            attempts=2,
        )


def test_config_fingerprint_is_stable_for_equal_configuration():
    common = {
        "provider": "openai",
        "model": "gpt-5.6-terra",
        "agent_revision": "a" * 40,
        "container_image": _image(),
        "agent_settings": {"effort": "high"},
    }
    assert BenchmarkRunConfig(**common).fingerprint == BenchmarkRunConfig(
        **common
    ).fingerprint


def test_trajectory_entry_rejects_a_mismatched_sequence():
    event = EventFactory("session")(UserMessage, text="hello")

    with pytest.raises(ValueError, match="sequence"):
        TrajectoryEntry(
            sequence=2,
            event_type="user.message",
            event=event,
        )


def test_scoreboard_row_requires_the_mathematically_correct_pass_rate():
    with pytest.raises(ValueError, match="pass_rate"):
        ScoreboardRow(
            task_id="fix-add-bug",
            provider="openai",
            model="gpt-5.6-terra",
            agent_revision="a" * 40,
            container_image=_image(),
            config_fingerprint="b" * 64,
            attempts=3,
            passed_attempts=2,
            pass_rate=Decimal("1"),
            mean_duration_seconds=Decimal("1"),
            duration_standard_deviation_seconds=Decimal("0"),
            mean_tokens=Decimal("1"),
            token_standard_deviation=Decimal("0"),
            mean_turns=Decimal("1"),
        )


def test_command_spec_uses_argv_not_an_empty_shell_command():
    with pytest.raises(ValueError, match="at least 1 item"):
        CommandSpec(argv=())
