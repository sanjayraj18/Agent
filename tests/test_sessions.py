from __future__ import annotations

import pytest

from agent.events import SessionStarted, UserMessage
from agent.persistence.database import SqliteDatabase
from agent.persistence.event_log import EventLog
from agent.persistence.migrations import migrate
from agent.persistence.models import SessionStatus
from agent.persistence.sessions import SessionForkError, SessionStore
from agent.providers.base import EventFactory


def _store(tmp_path):
    database = SqliteDatabase(tmp_path / "sessions.sqlite3")
    migrate(database)
    return SessionStore(database), EventLog(database)


def test_creates_archives_and_unarchives_a_session(tmp_path):
    store, _ = _store(tmp_path)
    session = store.create(
        workspace=tmp_path,
        model="claude-sonnet-5",
        title="Implement persistence",
    )

    archived = store.archive(session.session_id)

    assert archived.status == SessionStatus.ARCHIVED
    assert archived.archived_at is not None
    assert store.list() == ()
    assert store.list(include_archived=True) == (archived,)

    restored = store.unarchive(session.session_id)

    assert restored.status == SessionStatus.READY
    assert restored.archived_at is None


def test_fork_copies_history_but_never_mutates_the_parent(tmp_path):
    store, event_log = _store(tmp_path)
    parent = store.create(workspace=tmp_path, model="claude-sonnet-5")
    emit = EventFactory(parent.session_id)
    parent_events = (
        emit(SessionStarted, cwd=str(tmp_path), model=parent.model),
        emit(UserMessage, text="Implement persistence."),
    )
    event_log.append_many(parent_events)

    child = store.fork(parent.session_id, title="Try another design")

    child_events = event_log.replay(child.session_id)
    assert child.parent_session_id == parent.session_id
    assert child.forked_from_event_seq == 2
    assert child.last_event_seq == 2
    assert [event.session_id for event in child_events] == [child.session_id] * 2
    assert [event.seq for event in child_events] == [1, 2]
    assert event_log.replay(parent.session_id) == parent_events


def test_cannot_fork_an_empty_session(tmp_path):
    store, _ = _store(tmp_path)
    session = store.create(workspace=tmp_path, model="claude-sonnet-5")

    with pytest.raises(SessionForkError, match="without at least one event"):
        store.fork(session.session_id)
