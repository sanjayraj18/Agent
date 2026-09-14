import json

from agent.events import (
    AssistantEnd, AssistantStart, ErrorEvent, TextDelta, ThinkingDelta,
    ThinkingSignature, ToolCallDelta, ToolCallEnd, ToolCallStart,
)
from agent.providers.anthropic_wire import AnthropicTranslator
from agent.providers.base import EventFactory
from agent.providers.sse import SSEFrame


def run(*payloads: dict) -> list:
    t = AnthropicTranslator(EventFactory("s1"))
    out = []
    for p in payloads:
        out.extend(t.push(SSEFrame(event=p.get("type"), data=json.dumps(p))))
    out.extend(t.close())
    return out


def start(**usage):
    return {"type": "message_start", "message": {"usage": usage}}


STOP = {"type": "message_stop"}


def delta(stop_reason="end_turn", output_tokens=0):
    return {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason},
        "usage": {"output_tokens": output_tokens},
    }


def test_text_turn():
    events = run(
        start(input_tokens=10),
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        delta(output_tokens=5),
        STOP,
    )
    assert [type(e) for e in events] == [AssistantStart, TextDelta, AssistantEnd]
    assert events[-1].usage.input_tokens == 10
    assert events[-1].usage.output_tokens == 5


def test_usage_is_assembled_from_both_ends_of_the_stream():
    events = run(
        start(input_tokens=100, cache_read_input_tokens=9000,
              cache_creation_input_tokens=50),
        delta(output_tokens=42),
        STOP,
    )
    u = events[-1].usage
    assert (u.input_tokens, u.output_tokens) == (100, 42)
    assert (u.cache_read_input_tokens, u.cache_creation_input_tokens) == (9000, 50)


def test_tool_arguments_assemble_from_fragments():
    events = run(
        start(),
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "c1", "name": "read"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": '{"pa'}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": 'th":"a.py"}'}},
        {"type": "content_block_stop", "index": 0},
        delta(stop_reason="tool_use"),
        STOP,
    )
    end = next(e for e in events if isinstance(e, ToolCallEnd))
    assert end.arguments == {"path": "a.py"}
    assert end.call_id == "c1"


def test_interleaved_blocks_do_not_corrupt_each_other():
    """The whole reason `index` exists."""
    events = run(
        start(),
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking"}},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "c1", "name": "read"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta", "thinking": "checking "}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '{"path'}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta", "thinking": "the file"}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '":"a.py"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_stop", "index": 0},
        delta(stop_reason="tool_use"),
        STOP,
    )
    thinking = "".join(e.text for e in events if isinstance(e, ThinkingDelta))
    assert thinking == "checking the file"
    end = next(e for e in events if isinstance(e, ToolCallEnd))
    assert end.arguments == {"path": "a.py"}


def test_thinking_signature_is_emitted_once_even_if_fragmented():
    events = run(
        start(),
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "signature_delta", "signature": "sig_"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "signature_delta", "signature": "abc=="}},
        {"type": "content_block_stop", "index": 0},
        delta(), STOP,
    )
    sigs = [e for e in events if isinstance(e, ThinkingSignature)]
    assert len(sigs) == 1
    assert sigs[0].signature == "sig_abc=="


def test_zero_argument_tool_yields_empty_dict():
    events = run(
        start(),
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "c1", "name": "now"}},
        {"type": "content_block_stop", "index": 0},
        delta(stop_reason="tool_use"), STOP,
    )
    assert next(e for e in events if isinstance(e, ToolCallEnd)).arguments == {}


def test_pause_turn_is_preserved():
    events = run(start(), delta(stop_reason="pause_turn"), STOP)
    assert events[-1].stop_reason == "pause_turn"


def test_unknown_stop_reason_degrades_to_end_turn():
    events = run(start(), delta(stop_reason="something_new_in_2027"), STOP)
    assert events[-1].stop_reason == "end_turn"


def test_truncated_stream_is_an_error_not_a_completion():
    """HTTP 200, frames parsed fine, no message_stop. The silent killer."""
    events = run(
        start(),
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "half a th"}},
    )
    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].kind == "stream_truncated"
    assert events[-1].retryable is True


def test_mid_stream_error_is_classified_retryable():
    events = run(
        start(),
        {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
    )
    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].retryable is True


def test_ping_is_ignored():
    assert run(start(), {"type": "ping"}, delta(), STOP) == run(start(), delta(), STOP)


def test_exactly_one_terminal_event():
    events = run(start(), delta(), STOP, STOP, {"type": "ping"})
    terminal = [e for e in events if isinstance(e, (AssistantEnd, ErrorEvent))]
    assert len(terminal) == 1


def test_ping_is_ignored():
    with_ping = run(start(), {"type": "ping"}, delta(), STOP)
    without = run(start(), delta(), STOP)
    # ping emits nothing and consumes no sequence number
    assert [e.type for e in with_ping] == [e.type for e in without]
    assert [e.seq for e in with_ping] == [e.seq for e in without]