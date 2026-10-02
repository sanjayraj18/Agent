import json

from pydantic import ValidationError
import pytest

from agent.events import (
    AssistantEnd,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    Usage,
    dumps,
    loads,
)

def test_roundtrip_preserves_concrete_type():
    e = ToolCallDelta(seq=1, session_id="s1", index=1, call_id="c1", partial_json='{"pa')
    back = loads(dumps(e))
    assert isinstance(back, ToolCallDelta)
    assert back == e


def test_discriminator_distinguishes_identical_shapes():
    """Text and thinking deltas have the same fields — only `type` separates them."""
    t = ThinkingDelta(seq=1, session_id="s1", index=0, text="hmm")
    assert isinstance(loads(dumps(t)), ThinkingDelta)


def test_usage_survives_nesting():
    e = AssistantEnd(
        seq=9,
        session_id="s1",
        stop_reason="tool_use",
        usage=Usage(input_tokens=120, cache_read_input_tokens=9000),
    )
    assert loads(dumps(e)).usage.cache_read_input_tokens == 9000


def test_events_are_frozen():
    e = TextDelta(seq=1, session_id="s1", index=0, text="hi")
    with pytest.raises(ValidationError):
        e.text = "tampered"


def test_typos_are_rejected():
    with pytest.raises(ValidationError):
        TextDelta(seq=1, session_id="s1", index=0, txt="hi")

def test_assistant_end_roundtrip_preserves_execution_metadata():
    event = AssistantEnd(
        seq=9,
        session_id="session-1",
        stop_reason="end_turn",
        usage=Usage(
            input_tokens=120,
            output_tokens=30,
        ),
        executed_provider="openai",
        executed_model="gpt-5.6-luna",
        executed_route_id="economy-luna-medium",
    )

    restored = loads(dumps(event))

    assert isinstance(restored, AssistantEnd)
    assert restored.executed_provider == "openai"
    assert restored.executed_model == "gpt-5.6-luna"
    assert restored.executed_route_id == "economy-luna-medium"


def test_assistant_end_allows_a_fixed_model_without_a_route_id():
    event = AssistantEnd(
        seq=9,
        session_id="session-1",
        stop_reason="end_turn",
        usage=Usage(),
        executed_provider="openai",
        executed_model="gpt-5.6-terra",
    )

    assert event.executed_route_id is None


def test_unannotated_assistant_end_keeps_the_legacy_wire_shape():
    event = AssistantEnd(
        seq=9,
        session_id="session-1",
        stop_reason="end_turn",
        usage=Usage(),
    )

    encoded = json.loads(dumps(event))

    assert "executed_provider" not in encoded
    assert "executed_model" not in encoded
    assert "executed_route_id" not in encoded


def test_assistant_end_rejects_partial_execution_metadata():
    with pytest.raises(
        ValidationError,
        match="executed_provider and executed_model",
    ):
        AssistantEnd(
            seq=9,
            session_id="session-1",
            stop_reason="end_turn",
            usage=Usage(),
            executed_model="gpt-5.6-luna",
        )
