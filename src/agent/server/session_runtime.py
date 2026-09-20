from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from agent.events import (
    AssistantEnd,
    ErrorEvent,
    Event,
    SessionStarted,
    UserMessage,
)
from agent.persistence.event_log import EventLog
from agent.persistence.models import SessionStatus
from agent.persistence.sessions import SessionStore
from agent.providers.base import EventFactory


AgentRunner = Callable[
    [tuple[Event, ...], EventFactory],
    AsyncIterator[Event],
]


class SessionRuntimeError(RuntimeError):
    """Base error for one live persistent session."""


class SessionBusyError(SessionRuntimeError):
    """Raised when a second prompt tries to run in the same session."""


class SessionArchivedError(SessionRuntimeError):
    """Raised when code tries to run an archived session."""


class SubscriberLaggedError(SessionRuntimeError):
    """
    Raised when a client cannot keep up with the live event stream.

    The client can reconnect and replay missing events from SQLite.
    """


@dataclass(slots=True, eq=False)
class _Subscriber:
    queue: asyncio.Queue[Event]
    lagged: bool = False


class SessionRuntime:
    """
    Run and broadcast one durable agent session.

    Durability comes first:
    1. append event to SQLite;
    2. broadcast it to connected clients;
    3. yield it to the caller.

    If the process dies after step 1, the client can replay the event later.
    """

    def __init__(
        self,
        *,
        session_id: str,
        sessions: SessionStore,
        event_log: EventLog,
        max_subscriber_backlog: int = 1_000,
    ) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be blank")

        if max_subscriber_backlog < 1:
            raise ValueError(
                "max_subscriber_backlog must be at least 1"
            )

        self._session_id = session_id
        self._sessions = sessions
        self._event_log = event_log
        self._max_subscriber_backlog = max_subscriber_backlog

        # Only one model/tool run may advance an event sequence at a time.
        self._run_lock = asyncio.Lock()

        # Protects the handoff between SQLite replay and live subscription.
        self._subscriber_lock = asyncio.Lock()
        self._subscribers: set[_Subscriber] = set()

    @property
    def session_id(self) -> str:
        return self._session_id

    async def run_prompt(
        self,
        prompt: str,
        runner: AgentRunner,
    ) -> AsyncIterator[Event]:
        """
        Persist and stream one user prompt plus every agent event it causes.

        `runner` will later be the adapter around AgentLoop. It receives the
        complete durable history and an EventFactory already positioned at the
        next valid sequence number.
        """

        if not prompt.strip():
            raise ValueError("prompt must not be blank")

        if self._run_lock.locked():
            raise SessionBusyError(
                "this session already has an active agent run"
            )

        async with self._run_lock:
            session = await asyncio.to_thread(
                self._sessions.get,
                self._session_id,
            )

            if session.status == SessionStatus.ARCHIVED:
                raise SessionArchivedError(
                    "cannot run an archived session"
                )

            await asyncio.to_thread(
                self._sessions.set_runtime_status,
                self._session_id,
                SessionStatus.RUNNING,
            )

            emit = EventFactory(
                self._session_id,
                start_seq=session.last_event_seq,
            )
            history = list(
                await asyncio.to_thread(
                    self._event_log.replay,
                    self._session_id,
                )
            )

            completed_normally = False
            failed = False

            try:
                if not history:
                    session_started = emit(
                        SessionStarted,
                        cwd=session.workspace,
                        model=session.model,
                    )
                    await self._persist_and_broadcast(
                        session_started
                    )
                    history.append(session_started)
                    yield session_started

                user_message = emit(
                    UserMessage,
                    text=prompt,
                )
                await self._persist_and_broadcast(user_message)
                history.append(user_message)
                yield user_message

                async for event in runner(tuple(history), emit):
                    await self._persist_and_broadcast(event)
                    yield event

                    if isinstance(event, ErrorEvent):
                        if not event.retryable:
                            failed = True

                    elif isinstance(event, AssistantEnd):
                        if event.stop_reason not in {
                            "tool_use",
                            "pause_turn",
                        }:
                            completed_normally = True

            except asyncio.CancelledError:
                raise

            except Exception:
                failure = emit(
                    ErrorEvent,
                    kind="session_runtime_error",
                    message=(
                        "the persistent session runner failed "
                        "unexpectedly"
                    ),
                    retryable=False,
                )
                await self._persist_and_broadcast(failure)
                yield failure
                failed = True

            finally:
                status = (
                    SessionStatus.FAILED
                    if failed or not completed_normally
                    else SessionStatus.READY
                )

                await asyncio.to_thread(
                    self._sessions.set_runtime_status,
                    self._session_id,
                    status,
                )

    async def subscribe(
        self,
        *,
        after_sequence: int = 0,
    ) -> AsyncIterator[Event]:
        """
        Replay durable events, then continue with the live event stream.

        The subscriber is registered while holding `_subscriber_lock`, so
        there is no gap where a newly appended event could be missed.
        """

        if after_sequence < 0:
            raise ValueError(
                "after_sequence must not be negative"
            )

        subscriber = _Subscriber(
            queue=asyncio.Queue(
                maxsize=self._max_subscriber_backlog
            )
        )

        async with self._subscriber_lock:
            replayed_events = await asyncio.to_thread(
                self._event_log.replay,
                self._session_id,
                after_sequence=after_sequence,
            )
            self._subscribers.add(subscriber)

        try:
            for event in replayed_events:
                yield event

            while True:
                if subscriber.lagged:
                    raise SubscriberLaggedError(
                        "client fell behind the live event stream; "
                        "reconnect and replay from SQLite"
                    )

                event = await subscriber.queue.get()
                yield event

        finally:
            async with self._subscriber_lock:
                self._subscribers.discard(subscriber)

    async def _persist_and_broadcast(
        self,
        event: Event,
    ) -> None:
        """Write one event durably before any client sees it."""

        if event.session_id != self._session_id:
            raise SessionRuntimeError(
                "runner emitted an event for a different session"
            )

        async with self._subscriber_lock:
            await asyncio.to_thread(
                self._event_log.append,
                event,
            )

            for subscriber in tuple(self._subscribers):
                try:
                    subscriber.queue.put_nowait(event)
                except asyncio.QueueFull:
                    # Do not let one slow screen stall the agent. SQLite has
                    # the full history, so the client can reconnect safely.
                    subscriber.lagged = True
                    self._subscribers.discard(subscriber)