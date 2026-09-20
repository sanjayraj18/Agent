from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from agent.events import Event, dumps
from agent.server.session_service import SessionService


def create_http_app(session_service: SessionService) -> Any:
    """Create the optional local HTTP/SSE view of durable sessions.

    The HTTP process does not implement an agent loop itself. Every request
    reaches the same ``SessionService`` as the terminal JSON-RPC connection,
    so reconnecting through a browser cannot lose or reorder events.
    """

    try:
        fastapi = importlib.import_module("fastapi")
        responses = importlib.import_module("fastapi.responses")
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "HTTP serving requires the optional fastapi dependency"
        ) from exc

    app = fastapi.FastAPI(
        title="Local Agent Sessions",
        version="0.1.0",
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/sessions")
    async def list_sessions(
        include_archived: bool = False,
        limit: int = fastapi.Query(default=100, ge=1, le=1_000),
    ) -> dict[str, list[dict[str, Any]]]:
        sessions = await session_service.list_sessions(
            include_archived=include_archived,
            limit=limit,
        )
        return {
            "sessions": [
                session.model_dump(mode="json")
                for session in sessions
            ]
        }

    @app.post("/sessions")
    async def create_session(
        payload: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        workspace = payload.get("workspace")
        model = payload.get("model")
        title = payload.get("title")

        if not isinstance(workspace, str) or not workspace.strip():
            raise fastapi.HTTPException(
                status_code=422,
                detail="workspace must be a non-empty string",
            )
        if not isinstance(model, str) or not model.strip():
            raise fastapi.HTTPException(
                status_code=422,
                detail="model must be a non-empty string",
            )
        if title is not None and not isinstance(title, str):
            raise fastapi.HTTPException(
                status_code=422,
                detail="title must be a string when provided",
            )

        try:
            session = await session_service.create_session(
                workspace=Path(workspace),
                model=model,
                title=title,
            )
        except ValueError as exc:
            raise fastapi.HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        return {"session": session.model_dump(mode="json")}

    @app.get("/sessions/{session_id}/events")
    async def stream_events(
        session_id: str,
        after_sequence: int = fastapi.Query(default=0, ge=0),
    ) -> Any:
        try:
            await session_service.get_session(session_id)
        except Exception as exc:
            raise fastapi.HTTPException(
                status_code=404,
                detail="session was not found",
            ) from exc

        async def sse_events() -> AsyncIterator[str]:
            try:
                async for event in session_service.subscribe(
                    session_id,
                    after_sequence=after_sequence,
                ):
                    yield _sse_event(event)
            except asyncio.CancelledError:
                raise

        return responses.StreamingResponse(
            sse_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return app


def _sse_event(event: Event) -> str:
    """Encode one event as an SSE frame without trusting content as markup."""

    payload = json.loads(dumps(event))
    return (
        "event: agent.event\n"
        f"id: {event.seq}\n"
        f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    )
