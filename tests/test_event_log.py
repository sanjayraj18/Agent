from __future__ import annotations

import pytest

from agent.events import SessionStarted, UserMessage
from agent.persistence.database import SqliteDatabase
from agent.persistence.event_log import EventLog, EventSequenceError
from agent.persistence.migrations import migrate
from agent.persistence.sessions import SessionStore
from agent.providers.base import EventFactory


def _event_log(tmp_path):
    database = SqliteDatabase(tmp_path / "sessions.sqlite3")
    migrate(database)
    sessions = SessionStore(database)
    session = sessions.create(workspace=tmp_path, model="claude-sonnet-5")
    return EventLog(database), sessions, session


def test_appends_and_replays_typed_events_in_exact_order(tmp_path):
    event_log, _, session = _event_log(tmp_path)
    emit = EventFactory(session.session_id)
    events = (
        emit(SessionStarted, cwd=str(tmp_path), model=session.model),
        emit(UserMessage, text="Create a note."),
    )

    stored = event_log.append_many(events)

    assert [item.sequence for item in stored] == [1, 2]
    assert event_log.replay(session.session_id) == events
    assert event_log.last_sequence(session.session_id) == 2


def test_rejects_a_gap_without_partially_writing_the_batch(tmp_path):
    event_log, _, session = _event_log(tmp_path)
    emit = EventFactory(session.session_id)
    first = emit(SessionStarted, cwd=str(tmp_path), model=session.model)
    event_log.append(first)

    skipped = EventFactory(session.session_id, start_seq=2)(
        UserMessage,
        text="This sequence skips 2.",
    )

    with pytest.raises(EventSequenceError, match="expected event sequence 2"):
        event_log.append(skipped)

    assert event_log.replay(session.session_id) == (first,)


def test_rejects_one_batch_that_mixes_multiple_sessions(tmp_path):
    event_log, sessions, session = _event_log(tmp_path)
    second_session = sessions.create(
        workspace=tmp_path,
        model="claude-sonnet-5",
    )
    first = EventFactory(session.session_id)(
        SessionStarted,
        cwd=str(tmp_path),
        model=session.model,
    )
    second = EventFactory(second_session.session_id)(
        SessionStarted,
        cwd=str(tmp_path),
        model=second_session.model,
    )

    with pytest.raises(EventSequenceError, match="one append batch"):
        event_log.append_many((first, second))
