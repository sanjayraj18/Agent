from agent.events import (
    AssistantEnd,
    SessionStarted,
    ToolCallEnd,
    ToolCallStart,
    ToolResult,
    Usage,
    UserMessage,
)
from agent.providers.base import EventFactory
from agent.tui.state import RunStatus, ToolStatus, TuiState


def test_state_tracks_a_tool_and_exposes_its_latest_file_change():
    emit = EventFactory("tui-session")
    state = TuiState()

    state.apply(
        emit(
            SessionStarted,
            cwd="/workspace",
            model="claude-sonnet-5",
        )
    )
    state.apply(emit(UserMessage, text="Update the notes."))
    state.apply(
        emit(
            ToolCallStart,
            index=0,
            call_id="write-1",
            name="write_file",
        )
    )
    state.apply(
        emit(
            ToolCallEnd,
            index=0,
            call_id="write-1",
            arguments={"path": "notes.txt"},
        )
    )
    state.apply(
        emit(
            ToolResult,
            call_id="write-1",
            content="wrote notes.txt",
            file_change={
                "path": "notes.txt",
                "before": "before",
                "after": "after",
                "operation": "updated",
            },
        )
    )

    tool = state.ordered_tools[0]
    assert state.status == RunStatus.RUNNING
    assert tool.status == ToolStatus.COMPLETED
    assert tool.file_change == state.latest_file_change
    assert state.transcript[0].text == "Update the notes."


def test_state_marks_a_completed_assistant_turn_as_completed():
    emit = EventFactory("tui-session")
    state = TuiState()

    state.apply(
        emit(
            SessionStarted,
            cwd="/workspace",
            model="unknown-model",
        )
    )
    state.apply(
        emit(
            AssistantEnd,
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
        )
    )

    assert state.status == RunStatus.COMPLETED
    assert state.usage.input_tokens == 10
    assert state.usage.output_tokens == 5
    assert state.total_cost_usd is None
