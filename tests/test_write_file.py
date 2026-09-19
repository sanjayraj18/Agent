from pathlib import Path

import pytest

from agent.tools.workspace import Workspace
from agent.tools.write_file import WriteFileTool


def _tool(tmp_path: Path, *, max_bytes: int = 1_000_000) -> WriteFileTool:
    return WriteFileTool(Workspace(tmp_path), max_bytes=max_bytes)


async def test_writes_a_utf8_file_and_creates_requested_parents(tmp_path: Path):
    result = await _tool(tmp_path).execute(
        {
            "path": "notes/today.txt",
            "content": "hello\nworld\n",
            "create_parents": True,
        }
    )

    assert result.is_error is False
    assert result.content == "wrote 12 bytes to notes/today.txt"
    assert (tmp_path / "notes" / "today.txt").read_text(encoding="utf-8") == (
        "hello\nworld\n"
    )


async def test_does_not_overwrite_a_file_when_overwrite_is_false(tmp_path: Path):
    target = tmp_path / "notes.txt"
    target.write_text("original", encoding="utf-8")

    result = await _tool(tmp_path).execute(
        {
            "path": "notes.txt",
            "content": "replacement",
            "overwrite": False,
        }
    )

    assert result.is_error is True
    assert result.content == "file already exists: notes.txt"
    assert target.read_text(encoding="utf-8") == "original"


async def test_requires_explicit_permission_to_create_parent_directories(
    tmp_path: Path,
):
    result = await _tool(tmp_path).execute(
        {
            "path": "missing/notes.txt",
            "content": "hello",
        }
    )

    assert result.is_error is True
    assert result.content == "parent directory does not exist: missing"
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"path": "../outside.txt", "content": "hello"},
            "path escapes the workspace",
        ),
        (
            {"path": "notes.txt", "content": "hello", "overwrite": "yes"},
            "overwrite must be a boolean",
        ),
        (
            {"path": "notes.txt", "content": "hello", "unknown": True},
            "unexpected arguments: unknown",
        ),
    ],
)
async def test_rejects_unsafe_or_invalid_write_arguments(
    tmp_path: Path,
    arguments: dict,
    message: str,
):
    result = await _tool(tmp_path).execute(arguments)

    assert result.is_error is True
    assert result.content == message


async def test_rejects_content_larger_than_the_write_limit(tmp_path: Path):
    result = await _tool(tmp_path, max_bytes=4).execute(
        {"path": "notes.txt", "content": "hello"}
    )

    assert result.is_error is True
    assert result.content == "content exceeds the 4-byte write limit"
    assert not (tmp_path / "notes.txt").exists()
