import pytest

from agent.routing.signals import RoutingSignals


def test_creates_signals_without_failure_evidence():
    signals = RoutingSignals(
        task_complexity="simple",
        turn_number=1,
    )

    assert signals.task_complexity == "simple"
    assert signals.turn_number == 1
    assert signals.tool_error_count == 0
    assert signals.verification_failure_count == 0
    assert signals.has_failure_evidence is False


def test_detects_tool_failure_evidence():
    signals = RoutingSignals(
        task_complexity="simple",
        turn_number=2,
        tool_error_count=1,
    )

    assert signals.has_failure_evidence is True


def test_detects_verification_failure_evidence():
    signals = RoutingSignals(
        task_complexity="moderate",
        turn_number=3,
        verification_failure_count=1,
    )

    assert signals.has_failure_evidence is True


def test_rejects_a_turn_number_below_one():
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        RoutingSignals(
            task_complexity="simple",
            turn_number=0,
        )


def test_rejects_a_negative_tool_error_count():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        RoutingSignals(
            task_complexity="simple",
            turn_number=1,
            tool_error_count=-1,
        )


def test_rejects_unknown_extra_fields():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        RoutingSignals(
            task_complexity="simple",
            turn_number=1,
            model_instruction="always use the cheapest model",
        )