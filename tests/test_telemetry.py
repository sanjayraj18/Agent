from decimal import Decimal

import pytest

from agent.core.telemetry import SessionTelemetry
from agent.events import Usage


def test_records_turns_and_accumulates_session_totals():
    telemetry = SessionTelemetry()

    first_turn = telemetry.record_turn(
        model="claude-sonnet-5",
        usage=Usage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=1_000,
        ),
        stable_prefix_fingerprint="same-prefix",
    )
    second_turn = telemetry.record_turn(
        model="claude-sonnet-5",
        usage=Usage(
            input_tokens=200,
            output_tokens=100,
            cache_read_input_tokens=1_000,
        ),
        stable_prefix_fingerprint="same-prefix",
    )

    assert first_turn.turn_number == 1
    assert second_turn.turn_number == 2
    assert len(telemetry.turns) == 2

    assert telemetry.input_tokens == 300
    assert telemetry.output_tokens == 150
    assert telemetry.cache_creation_tokens == 1_000
    assert telemetry.cache_read_tokens == 1_000
    assert telemetry.total_tokens == 2_450

    assert telemetry.known_total_cost_usd == Decimal("0.0072")
    assert telemetry.total_cost_usd == Decimal("0.0072")


def test_calculates_cache_hit_rate_from_prompt_tokens_only():
    telemetry = SessionTelemetry()

    telemetry.record_turn(
        model="claude-sonnet-5",
        usage=Usage(
            input_tokens=100,
            output_tokens=500,
            cache_creation_input_tokens=900,
        ),
    )
    telemetry.record_turn(
        model="claude-sonnet-5",
        usage=Usage(
            input_tokens=200,
            output_tokens=1_000,
            cache_read_input_tokens=800,
        ),
    )

    # Output tokens do not belong in this calculation.
    assert telemetry.cache_hit_rate == Decimal("0.4")


def test_returns_zero_cache_hit_rate_when_no_prompt_tokens_exist():
    telemetry = SessionTelemetry()

    assert telemetry.cache_hit_rate == Decimal("0")


def test_marks_the_total_as_unknown_when_any_turn_has_unknown_pricing():
    telemetry = SessionTelemetry()

    telemetry.record_turn(
        model="claude-sonnet-5",
        usage=Usage(input_tokens=100),
    )
    telemetry.record_turn(
        model="future-unknown-model",
        usage=Usage(input_tokens=100),
    )

    assert telemetry.unpriced_turn_count == 1
    assert telemetry.known_total_cost_usd == Decimal("0.0003")
    assert telemetry.total_cost_usd is None


def test_detects_when_the_stable_prefix_changes_during_a_session():
    telemetry = SessionTelemetry()

    telemetry.record_turn(
        model="claude-sonnet-5",
        usage=Usage(),
        stable_prefix_fingerprint="prefix-a",
    )
    telemetry.record_turn(
        model="claude-sonnet-5",
        usage=Usage(),
        stable_prefix_fingerprint="prefix-b",
    )

    assert telemetry.cache_prefix_changed is True


def test_allows_one_stable_prefix_for_the_whole_session():
    telemetry = SessionTelemetry()

    telemetry.record_turn(
        model="claude-sonnet-5",
        usage=Usage(),
        stable_prefix_fingerprint="prefix-a",
    )
    telemetry.record_turn(
        model="claude-sonnet-5",
        usage=Usage(),
        stable_prefix_fingerprint="prefix-a",
    )

    assert telemetry.cache_prefix_changed is False


def test_rejects_an_empty_model_name():
    telemetry = SessionTelemetry()

    with pytest.raises(ValueError, match="model must not be empty"):
        telemetry.record_turn(
            model="",
            usage=Usage(),
        )