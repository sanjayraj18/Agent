from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import cast

import pytest

from agent.events import AssistantEnd, AssistantStart, Usage
from agent.persistence.database import SqliteDatabase
from agent.persistence.event_log import EventLog
from agent.persistence.migrations import migrate
from agent.persistence.sessions import SessionStore
from agent.providers.base import EventFactory
from agent.server.session_runtime import (
    SessionArchivedError,
    SessionBusyError,
    SessionRuntime,
)


def _runtime(tmp_path) -> tuple[SessionRuntime, EventLog, SessionStore, str]:
    database = SqliteDatabase(tmp_path / "sessions.sqlite3")
    migrate(database)
    sessions = SessionStore(database)
    record = sessions.create(workspace=tmp_path, model="claude-sonnet-5")
    event_log = EventLog(database)
    runtime = SessionRuntime(
        session_id=record.session_id,
        sessions=sessions,
        event_log=event_log,
    )
    return runtime, event_log, sessions, record.session_id


async def test_runtime_persists_events_before_replay_and_subscription(tmp_path):
    runtime, event_log, _, session_id = _runtime(tmp_path)

    async def runner(history, emit):
        assert [event.type for event in history] == [
            "session.started",
            "user.message",
        ]
        yield emit(AssistantStart)
        yield emit(
            AssistantEnd,
            stop_reason="end_turn",
            usage=Usage(),
        )

    subscription = cast(
        AsyncGenerator,
        runtime.subscribe(),
    )

    async def next_subscription_event():
        return await anext(subscription)

    waiting_for_first_event = asyncio.create_task(
        next_subscription_event()
    )
    await asyncio.sleep(0)

    emitted = [
        event async for event in runtime.run_prompt("hello", runner)
    ]
    received = await asyncio.wait_for(waiting_for_first_event, timeout=1)
    await subscription.aclose()

    replayed = event_log.replay(session_id)
    assert received == emitted[0]
    assert replayed == tuple(emitted)
    assert [event.seq for event in replayed] == [1, 2, 3, 4]


async def test_runtime_rejects_a_second_concurrent_run(tmp_path):
    runtime, _, _, _ = _runtime(tmp_path)
    entered_runner = asyncio.Event()
    release_runner = asyncio.Event()

    async def blocked_runner(history, emit):
        yield emit(AssistantStart)
        entered_runner.set()
        await release_runner.wait()
        yield emit(
            AssistantEnd,
            stop_reason="end_turn",
            usage=Usage(),
        )

    async def collect_first_run():
        return [
            event
            async for event in runtime.run_prompt("first", blocked_runner)
        ]

    first_run = asyncio.create_task(collect_first_run())
    await asyncio.wait_for(entered_runner.wait(), timeout=1)

    with pytest.raises(SessionBusyError, match="already has an active"):
        async for _ in runtime.run_prompt("second", blocked_runner):
            pass

    release_runner.set()
    await asyncio.wait_for(first_run, timeout=1)


async def test_runtime_rejects_a_prompt_for_an_archived_session(tmp_path):
    runtime, _, sessions, session_id = _runtime(tmp_path)
    sessions.archive(session_id)

    async def unused_runner(history, emit):
        if False:
            yield emit(AssistantStart)

    with pytest.raises(SessionArchivedError, match="archived"):
        async for _ in runtime.run_prompt("resume", unused_runner):
            pass
