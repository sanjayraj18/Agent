from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent.events import Event, dumps
from agent.providers.anthropic_wire import AnthropicTranslator
from agent.providers.base import EventFactory
from agent.providers.sse import SSEParser

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURES = sorted(FIXTURE_DIR.glob("*.sse"))


def replay(raw: bytes, chunk_size: int | None = None) -> list[Event]:
    """Run raw bytes through the full pipeline: SSE parser -> translator."""
    parser = SSEParser()
    translator = AnthropicTranslator(EventFactory("fixture"))
    events: list[Event] = []

    chunks = (
        [raw]
        if chunk_size is None
        else [raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)]
    )

    for chunk in chunks:
        for frame in parser.feed(chunk):
            events.extend(translator.push(frame))
    # Parser first: its trailing frames still need translating.
    for frame in parser.close():
        events.extend(translator.push(frame))
    events.extend(translator.close())
    return events


def normalize(event: Event) -> dict:
    """Strip fields that differ per run so snapshots are stable."""
    d = json.loads(dumps(event))
    d.pop("ts", None)          # wall clock
    d.pop("session_id", None)  # environment
    return d                   # seq is kept — ordering is part of the contract


def snapshot_path(fixture: Path) -> Path:
    return fixture.with_name(fixture.stem + ".events.json")


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_transcript_matches_snapshot(fixture: Path):
    actual = [normalize(e) for e in replay(fixture.read_bytes())]
    snap = snapshot_path(fixture)

    if os.environ.get("REGEN_SNAPSHOTS"):
        snap.write_text(json.dumps(actual, indent=2) + "\n")
        pytest.skip(f"regenerated {snap.name}")

    assert snap.exists(), (
        f"no snapshot for {fixture.name}. "
        f"Generate with: REGEN_SNAPSHOTS=1 uv run pytest -q"
    )
    assert actual == json.loads(snap.read_text())


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
@pytest.mark.parametrize("chunk_size", [1, 3, 17, 512, None])
def test_chunking_never_changes_the_event_stream(fixture: Path, chunk_size):
    """Network chunk boundaries are arbitrary. Output must not depend on them."""
    raw = fixture.read_bytes()
    assert [normalize(e) for e in replay(raw, chunk_size)] == [
        normalize(e) for e in replay(raw)
    ]

#test

@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_every_transcript_ends_with_exactly_one_terminal_event(fixture: Path):
    events = replay(fixture.read_bytes())
    terminal = [e for e in events if e.type in ("assistant.end", "error")]
    assert len(terminal) == 1
    assert events[-1].type == "assistant.end"


@pytest.mark.parametrize("cut", [0.25, 0.5, 0.75, 0.9])
def test_truncation_at_any_point_is_an_error(cut: float):
    """Derived from a real transcript — no separate fixture needed."""
    raw = (FIXTURE_DIR / "tool_call.sse").read_bytes()
    events = replay(raw[: int(len(raw) * cut)])
    assert events[-1].type == "error"
    assert events[-1].kind == "stream_truncated"


def test_cache_tokens_survive_the_pipeline():
    """The four usage fields are the basis of every cost claim in Phase 5."""
    events = replay((FIXTURE_DIR / "text_turn.sse").read_bytes())
    usage = events[-1].usage
    assert usage.input_tokens == 2145
    assert usage.cache_read_input_tokens == 9018
    assert usage.cache_creation_input_tokens == 0
    assert usage.output_tokens == 14