from decimal import Decimal

import pytest

from agent.core.telemetry import SessionTelemetry, ShadowRouteRecommendation, TurnTiming
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

def test_records_timing_for_a_completed_turn():
    telemetry = SessionTelemetry()

    turn = telemetry.record_turn(
        model="claude-sonnet-5",
        usage=Usage(input_tokens=100, output_tokens=50),
        timing=TurnTiming(
            provider_duration_seconds=Decimal("2.50"),
            time_to_first_output_seconds=Decimal("0.75"),
        ),
    )

    assert turn.timing is not None
    assert turn.timing.provider_duration_seconds == Decimal("2.50")
    assert turn.timing.time_to_first_output_seconds == Decimal("0.75")
    assert telemetry.timed_turn_count == 1
    assert telemetry.unmeasured_turn_count == 0
    assert telemetry.known_total_provider_duration_seconds == Decimal(
        "2.50"
    )
    assert telemetry.total_provider_duration_seconds == Decimal("2.50")


def test_marks_session_timing_unknown_when_a_turn_was_not_measured():
    telemetry = SessionTelemetry()

    telemetry.record_turn(
        model="claude-sonnet-5",
        usage=Usage(),
        timing=TurnTiming(
            provider_duration_seconds=Decimal("1.25"),
        ),
    )
    telemetry.record_turn(
        model="claude-sonnet-5",
        usage=Usage(),
    )

    assert telemetry.timed_turn_count == 1
    assert telemetry.unmeasured_turn_count == 1
    assert telemetry.known_total_provider_duration_seconds == Decimal(
        "1.25"
    )
    assert telemetry.total_provider_duration_seconds is None


def test_allows_a_turn_with_no_streamed_output():
    timing = TurnTiming(
        provider_duration_seconds=Decimal("0.50"),
    )

    assert timing.time_to_first_output_seconds is None


def test_rejects_negative_provider_duration():
    with pytest.raises(
        ValueError,
        match="provider_duration_seconds must not be negative",
    ):
        TurnTiming(
            provider_duration_seconds=Decimal("-0.01"),
        )


def test_rejects_first_output_after_turn_completion():
    with pytest.raises(
        ValueError,
        match="cannot exceed provider_duration_seconds",
    ):
        TurnTiming(
            provider_duration_seconds=Decimal("1.00"),
            time_to_first_output_seconds=Decimal("1.01"),
        )

def test_records_shadow_route_recommendations():
    telemetry = SessionTelemetry()

    telemetry.record_turn(
        model="gpt-5.6-terra",
        usage=Usage(),
        shadow_route=ShadowRouteRecommendation(
            route_id="economy-luna-medium",
            provider="openai",
            model="gpt-5.6-luna",
            reasons=("simple task with no failure evidence",),
        ),
    )
    telemetry.record_turn(
        model="gpt-5.6-terra",
        usage=Usage(),
        shadow_route=ShadowRouteRecommendation(
            route_id="strong-terra-high",
            provider="openai",
            model="gpt-5.6-terra",
            reasons=("complex tasks start on the strong route",),
        ),
    )

    assert telemetry.shadow_routed_turn_count == 2

    # First recommendation differs from Terra; second recommendation matches.
    assert telemetry.shadow_model_difference_count == 1


def test_rejects_a_shadow_route_with_no_reasons():
    with pytest.raises(ValueError, match="reasons must not be empty"):
        ShadowRouteRecommendation(
            route_id="economy-luna-medium",
            provider="openai",
            model="gpt-5.6-luna",
            reasons=(),
        )


def test_rejects_a_shadow_route_with_a_blank_route_id():
    with pytest.raises(ValueError, match="route_id must not be blank"):
        ShadowRouteRecommendation(
            route_id="   ",
            provider="openai",
            model="gpt-5.6-luna",
            reasons=("simple task",),
        )