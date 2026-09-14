from __future__ import annotations

import json
from pathlib import Path

import httpx          
import httpx2        
import pytest
from anthropic import AsyncAnthropic

from agent.events import Event, dumps
from agent.providers.anthropic_raw import AnthropicRawProvider
from agent.providers.anthropic_sdk import AnthropicSDKProvider
from agent.providers.base import EventFactory, Message, ProviderRequest, TextPart

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURES = sorted(FIXTURE_DIR.glob("*.sse"))

REQ = ProviderRequest(
    model="claude-opus-5",
    max_tokens=1024,
    messages=[Message(role="user", content=[TextPart(text="hi")])],
)

SSE_HEADERS = {"content-type": "text/event-stream"}


def norm(event: Event) -> dict:
    d = json.loads(dumps(event))
    d.pop("ts", None)
    d.pop("session_id", None)
    return d


def raw(body: bytes = b"", status: int = 200, headers: dict | None = None):
    def handler(_):
        return httpx.Response(status, content=body, headers=headers or SSE_HEADERS)

    return AnthropicRawProvider(
        "k", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def sdk(body: bytes = b"", status: int = 200, headers: dict | None = None):
    def handler(_):
        return httpx2.Response(status, content=body, headers=headers or SSE_HEADERS)

    return AnthropicSDKProvider(
        client=AsyncAnthropic(
            api_key="k",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
    )


async def collect(provider) -> list[dict]:
    return [norm(e) async for e in provider.stream(REQ, EventFactory("s1"))]


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
async def test_both_providers_produce_identical_events(fixture: Path):
    """The Phase 1 acceptance criterion."""
    body = fixture.read_bytes()
    assert await collect(raw(body)) == await collect(sdk(body))


async def test_both_providers_agree_on_truncation():
    body = (FIXTURE_DIR / "tool_call.sse").read_bytes()[:400]
    assert await collect(raw(body)) == await collect(sdk(body))


async def test_both_providers_agree_on_429():
    err = json.dumps(
        {"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}
    ).encode()
    headers = {"content-type": "application/json", "retry-after": "12"}

    a = (await collect(raw(err, 429, headers)))[-1]
    b = (await collect(sdk(err, 429, headers)))[-1]
    assert a["kind"] == b["kind"] == "rate_limit_error"
    assert a["retryable"] == b["retryable"] is True
    assert a["retry_after"] == b["retry_after"] == 12.0


async def test_both_providers_agree_on_401():
    err = json.dumps(
        {"type": "error", "error": {"type": "authentication_error", "message": "bad"}}
    ).encode()
    headers = {"content-type": "application/json"}

    a = (await collect(raw(err, 401, headers)))[-1]
    b = (await collect(sdk(err, 401, headers)))[-1]
    assert a["kind"] == b["kind"] == "authentication_error"
    assert a["retryable"] == b["retryable"] is False


async def test_sdk_rejects_an_httpx_client():
    """Documents the httpx / httpx2 split so it can't silently regress."""
    with pytest.raises(TypeError, match="httpx2"):
        AsyncAnthropic(api_key="k", http_client=httpx.AsyncClient())