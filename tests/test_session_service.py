from __future__ import annotations

from agent.events import AssistantEnd, AssistantStart, Usage
from agent.persistence.database import SqliteDatabase
from agent.persistence.event_log import EventLog
from agent.persistence.migrations import migrate
from agent.persistence.sessions import SessionStore
from agent.server.session_service import SessionService


def _service(tmp_path, observed_histories):
    database = SqliteDatabase(tmp_path / "sessions.sqlite3")
    migrate(database)

    def runner_factory(session, approval_handler):
        async def runner(history, emit):
            observed_histories.append(tuple(event.type for event in history))
            yield emit(AssistantStart)
            yield emit(
                AssistantEnd,
                stop_reason="end_turn",
                usage=Usage(),
            )

        return runner

    return SessionService(
        sessions=SessionStore(database),
        event_log=EventLog(database),
        runner_factory=runner_factory,
    )


async def test_service_rebuilds_history_after_a_server_restart(tmp_path):
    first_histories: list[tuple[str, ...]] = []
    first_service = _service(tmp_path, first_histories)
    session = await first_service.create_session(
        workspace=tmp_path,
        model="claude-sonnet-5",
    )

    first_events = [
        event
        async for event in first_service.run_prompt(
            session.session_id,
            "first prompt",
        )
    ]

    second_histories: list[tuple[str, ...]] = []
    restarted_service = _service(tmp_path, second_histories)
    second_events = [
        event
        async for event in restarted_service.run_prompt(
            session.session_id,
            "second prompt",
        )
    ]

    assert first_histories == [
        ("session.started", "user.message")
    ]
    assert second_histories == [
        (
            "session.started",
            "user.message",
            "assistant.start",
            "assistant.end",
            "user.message",
        )
    ]
    assert [event.seq for event in first_events + second_events] == list(
        range(1, 8)
    )
    assert len(await restarted_service.replay(session.session_id)) == 7


async def test_service_lists_archives_and_forks_sessions(tmp_path):
    service = _service(tmp_path, [])
    session = await service.create_session(
        workspace=tmp_path,
        model="claude-sonnet-5",
    )
    _ = [
        event
        async for event in service.run_prompt(session.session_id, "hello")
    ]

    child = await service.fork_session(session.session_id)
    archived = await service.archive_session(session.session_id)

    assert archived.status.value == "archived"
    assert child.parent_session_id == session.session_id
    assert [item.session_id for item in await service.list_sessions()] == [
        child.session_id
    ]
