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