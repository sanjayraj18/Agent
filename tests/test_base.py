
import pytest
from pydantic import TypeAdapter

from agent.events import Event, TextDelta, ToolCallStart
from agent.providers.base import (
    ContentPart,
    EventFactory,
    Message,
    Provider,
    ProviderRequest,
    ThinkingPart,
    ToolUsePart,
)

PartAdapter: TypeAdapter[ContentPart] = TypeAdapter(ContentPart)


def test_content_parts_roundtrip_by_discriminator():
    part = ToolUsePart(call_id="c1", name="read", arguments={"path": "a.py"})
    back = PartAdapter.validate_json(PartAdapter.dump_json(part))
    assert isinstance(back, ToolUsePart)
    assert back == part


def test_thinking_signature_survives_roundtrip():
    """Signatures must come back byte-identical or the next turn is rejected."""
    part = ThinkingPart(text="considering...", signature="sig_abc123==")
    back = PartAdapter.validate_json(PartAdapter.dump_json(part))
    assert back.signature == "sig_abc123=="


def test_event_factory_is_monotonic_and_gapless():
    emit = EventFactory(session_id="s1")
    events = [emit(TextDelta, index=0, text=c) for c in "abc"]
    assert [e.seq for e in events] == [1, 2, 3]
    assert all(e.session_id == "s1" for e in events)



def test_event_factory_preserves_concrete_type():
    emit = EventFactory(session_id="s1")
    e = emit(ToolCallStart, index=0, call_id="c1", name="read")
    assert isinstance(e, ToolCallStart)
    assert e.name == "read"


def test_event_factory_can_resume_from_a_sequence():
    """Resuming a session continues the numbering rather than restarting it."""
    emit = EventFactory(session_id="s1", start_seq=47)
    assert emit(TextDelta, index=0, text="x").seq == 48


def test_a_minimal_fake_satisfies_the_protocol():
    class Fake:
        name = "fake"

        async def stream(self, request, emit):
            yield emit(TextDelta, index=0, text="hi")

    assert isinstance(Fake(), Provider)


@pytest.mark.anyio
async def test_fake_provider_yields_stamped_events():
    class Fake:
        name = "fake"

        async def stream(self, request, emit):
            yield emit(TextDelta, index=0, text="hi")

    req = ProviderRequest(
        model="claude-opus-5",
        max_tokens=1024,
        messages=[Message(role="user", content=[])],
    )
    out = [e async for e in Fake().stream(req, EventFactory("s1"))]
    assert out[0].seq == 1
