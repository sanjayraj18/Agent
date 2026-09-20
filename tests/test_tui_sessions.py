from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

from agent.events import AssistantEnd, AssistantStart, Event, SessionStarted, TextDelta, Usage, UserMessage
from agent.persistence.models import SessionRecord
from agent.providers.base import EventFactory
from agent.tui.app import AgentTuiApp
from agent.tui.rpc_client import AgentRpcClient


class DurableClient:
    def __init__(self) -> None:
        self.session = SessionRecord(
            session_id="session-1",
            workspace="/workspace",
            model="claude-sonnet-5",
        )
        self.child = SessionRecord(
            session_id="session-2",
            workspace="/workspace",
            model="claude-sonnet-5",
            parent_session_id="session-1",
            forked_from_event_seq=4,
        )
        self.last_run_status: str | None = None
        self.run_session_ids: list[str | None] = []

    async def create_session(self) -> SessionRecord:
        return self.session

    async def run(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
    ) -> AsyncIterator[Event]:
        self.run_session_ids.append(session_id)
        emit = EventFactory(session_id or "legacy")
        yield emit(
            SessionStarted,
            cwd="/workspace",
            model="claude-sonnet-5",
        )
        yield emit(UserMessage, text=prompt)
        yield emit(AssistantStart)
        yield emit(TextDelta, index=0, text="Done")
        yield emit(
            AssistantEnd,
            stop_reason="end_turn",
            usage=Usage(),
        )
        self.last_run_status = "completed"

    async def list_sessions(self) -> tuple[SessionRecord, ...]:
        return (self.session,)

    async def replay_session(
        self,
        session_id: str,
    ) -> tuple[Event, ...]:
        emit = EventFactory(session_id)
        return (
            emit(
                SessionStarted,
                cwd="/workspace",
                model="claude-sonnet-5",
            ),
            emit(UserMessage, text="saved prompt"),
            emit(AssistantStart),
            emit(
                AssistantEnd,
                stop_reason="end_turn",
                usage=Usage(),
            ),
        )

    async def fork_session(self, session_id: str) -> SessionRecord:
        assert session_id == self.session.session_id
        return self.child


async def test_tui_creates_a_durable_session_before_its_first_prompt():
    client = DurableClient()
    app = AgentTuiApp(client=cast(AgentRpcClient, client))

    async with app.run_test(size=(120, 40)):
        await app._run_prompt("implement persistence")

        assert client.run_session_ids == ["session-1"]
        assert app.state.session_id == "session-1"
        assert app.state.transcript[-1].text == "Done"


async def test_tui_resumes_and_forks_durable_history():
    client = DurableClient()
    app = AgentTuiApp(client=cast(AgentRpcClient, client))

    async with app.run_test(size=(120, 40)):
        await app._resume_recent_session()
        assert app.state.session_id == "session-1"
        assert app.state.transcript[0].text == "saved prompt"

        await app._fork_current_session()
        assert app.state.session_id == "session-2"
