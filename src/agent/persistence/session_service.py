from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from agent.events import Event
from agent.persistence.event_log import EventLog
from agent.persistence.models import SessionRecord
from agent.persistence.sessions import SessionStore
from agent.server.session_runtime import (
    AgentRunner,
    SessionRuntime,
)
from agent.tools.dispatcher import ApprovalHandler


AgentRunnerFactory = Callable[
    [SessionRecord, ApprovalHandler | None],
    AgentRunner,
]


class SessionService:
    """
    Coordinate durable sessions and their shared live runtimes.

    One SessionService belongs to one long-running local server process.
    Every attached TUI client reaches the same SessionRuntime instance.
    """

    def __init__(
        self,
        *,
        sessions: SessionStore,
        event_log: EventLog,
        runner_factory: AgentRunnerFactory,
    ) -> None:
        self._sessions = sessions
        self._event_log = event_log
        self._runner_factory = runner_factory

        self._runtime_lock = asyncio.Lock()
        self._runtimes: dict[str, SessionRuntime] = {}

    async def create_session(
        self,
        *,
        workspace: Path | str,
        model: str,
        title: str | None = None,
    ) -> SessionRecord:
        """Create a durable session record."""

        return await asyncio.to_thread(
            self._sessions.create,
            workspace=workspace,
            model=model,
            title=title,
        )

    async def get_session(
        self,
        session_id: str,
    ) -> SessionRecord:
        """Load metadata for one session."""

        return await asyncio.to_thread(
            self._sessions.get,
            session_id,
        )

    async def list_sessions(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
    ) -> tuple[SessionRecord, ...]:
        """Return recent sessions for a TUI session picker."""

        return await asyncio.to_thread(
            self._sessions.list,
            include_archived=include_archived,
            limit=limit,
        )

    async def archive_session(
        self,
        session_id: str,
    ) -> SessionRecord:
        """Archive without deleting the durable event history."""

        return await asyncio.to_thread(
            self._sessions.archive,
            session_id,
        )

    async def fork_session(
        self,
        session_id: str,
        *,
        at_sequence: int | None = None,
        title: str | None = None,
    ) -> SessionRecord:
        """Create an independent branch from a prior event sequence."""

        return await asyncio.to_thread(
            self._sessions.fork,
            session_id,
            at_sequence=at_sequence,
            title=title,
        )

    async def run_prompt(
        self,
        session_id: str,
        prompt: str,
        *,
        approval_handler: ApprovalHandler | None = None,
    ) -> AsyncIterator[Event]:
        """
        Run one prompt in a durable session.

        The runtime persists every event before it reaches the caller.
        """

        runtime = await self._runtime_for(session_id)
        session = await self.get_session(session_id)
        runner = self._runner_factory(session, approval_handler)

        async for event in runtime.run_prompt(prompt, runner):
            yield event

    async def subscribe(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
    ) -> AsyncIterator[Event]:
        """
        Replay history, then stream new events from the shared runtime.
        """

        runtime = await self._runtime_for(session_id)

        async for event in runtime.subscribe(
            after_sequence=after_sequence,
        ):
            yield event

    async def replay(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
    ) -> tuple[Event, ...]:
        """Read a finite, durable event history for a resuming client."""

        # EventLog validates both the session ID and sequence boundary before
        # decoding rows back into typed agent events.
        return await asyncio.to_thread(
            self._event_log.replay,
            session_id,
            after_sequence=after_sequence,
        )

    async def _runtime_for(
        self,
        session_id: str,
    ) -> SessionRuntime:
        """
        Return the one in-memory runtime for this session.

        The lock prevents two simultaneous attach/run requests from creating
        separate runtimes for the same durable session.
        """

        async with self._runtime_lock:
            existing = self._runtimes.get(session_id)

            if existing is not None:
                return existing

            # Verifies the session exists before exposing a runtime.
            await self.get_session(session_id)

            runtime = SessionRuntime(
                session_id=session_id,
                sessions=self._sessions,
                event_log=self._event_log,
            )
            self._runtimes[session_id] = runtime

            return runtime
