from __future__ import annotations

import pytest
from typing import cast

from agent.events import UserMessage
from agent.providers.base import EventFactory
from agent.server.http import _sse_event, create_http_app
from agent.server.session_service import SessionService


def test_sse_encoder_emits_one_safe_event_frame():
    event = EventFactory("session-1")(UserMessage, text="hello\nworld")

    frame = _sse_event(event)

    assert frame.startswith("event: agent.event\nid: 1\ndata: {")
    assert "hello\\nworld" in frame
    assert frame.endswith("\n\n")


def test_http_app_exposes_session_and_sse_routes_when_dependency_is_installed():
    pytest.importorskip("fastapi")

    class Service:
        pass

    app = create_http_app(cast(SessionService, Service()))
    paths = {route.path for route in app.routes}

    assert {
        "/health",
        "/sessions",
        "/sessions/{session_id}/events",
    } <= paths
