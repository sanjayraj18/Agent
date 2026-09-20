from pathlib import Path
from typing import Any

import pytest

from agent.core.conversation import messages_from_events
from agent.events import ToolResult
from agent.providers.base import EventFactory, ToolResultPart, ToolUsePart
from agent.tools.base import Tool, ToolExecutionResult
from agent.tools.changes import FileChange
from agent.tools.dispatcher import ToolDispatcher
from agent.tools.edit_file import EditFileTool
from agent.tools.registry import ToolRegistry
from agent.tools.workspace import Workspace
from agent.tools.write_file import WriteFileTool


def test_file_change_rejects_paths_that_could_escape_the_workspace():
    with pytest.raises(ValueError, match="workspace-relative"):
        FileChange(
            path="../outside.txt",
            before="",
            after="text",
            operation="created",
        )


async def test_write_file_reports_a_created_file_change(tmp_path: Path):
    tool = WriteFileTool(Workspace(tmp_path))

    result = await tool.execute(
        {"path": "notes.txt", "content": "hello\n"}
    )

    assert result.is_error is False
    assert result.file_change == FileChange(
        path="notes.txt",
        before="",
        after="hello\n",
        operation="created",
    )


async def test_write_file_reports_the_previous_text_when_updating(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("before\n", encoding="utf-8")
    tool = WriteFileTool(Workspace(tmp_path))

    result = await tool.execute(
        {"path": "notes.txt", "content": "after\n"}
    )

    assert result.file_change == FileChange(
        path="notes.txt",
        before="before\n",
        after="after\n",
        operation="updated",
    )


async def test_write_file_refuses_to_blindly_overwrite_binary_content(
    tmp_path: Path,
):
    (tmp_path / "image.bin").write_bytes(b"\x00binary")
    tool = WriteFileTool(Workspace(tmp_path))

    result = await tool.execute(
        {"path": "image.bin", "content": "replacement"}
    )

    assert result.is_error is True
    assert "refusing to overwrite" in result.content
    assert (tmp_path / "image.bin").read_bytes() == b"\x00binary"


async def test_edit_file_reports_a_before_and_after_text_change(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("hello NAME", encoding="utf-8")
    tool = EditFileTool(Workspace(tmp_path))

    result = await tool.execute(
        {
            "path": "notes.txt",
            "old_text": "NAME",
            "new_text": "Sanjay",
        }
    )

    assert result.file_change == FileChange(
        path="notes.txt",
        before="hello NAME",
        after="hello Sanjay",
        operation="updated",
    )


class ChangeReportingTool(Tool):
    name = "write_file"
    description = "Test tool that returns a local UI file change."
    input_schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            content="wrote notes.txt",
            file_change=FileChange(
                path="notes.txt",
                before="before",
                after="after",
                operation="updated",
            ),
        )


async def test_dispatcher_keeps_file_change_out_of_provider_serialization():
    dispatcher = ToolDispatcher(ToolRegistry([ChangeReportingTool()]))

    result = await dispatcher.dispatch(
        ToolUsePart(
            call_id="call-1",
            name="write_file",
            arguments={},
        )
    )

    assert result.file_change == {
        "path": "notes.txt",
        "before": "before",
        "after": "after",
        "operation": "updated",
    }
    assert "file_change" not in result.model_dump()


def test_conversation_omits_file_change_metadata_from_the_next_llm_request():
    emit = EventFactory("session-1")
    event = emit(
        ToolResult,
        call_id="call-1",
        content="wrote notes.txt",
        file_change={
            "path": "notes.txt",
            "before": "before",
            "after": "after",
            "operation": "updated",
        },
    )

    messages = messages_from_events([event])
    part = messages[0].content[0]

    assert isinstance(part, ToolResultPart)
    assert part.content == "wrote notes.txt"
    assert part.file_change is None
    assert "file_change" not in part.model_dump()
