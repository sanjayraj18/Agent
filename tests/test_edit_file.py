from pathlib import Path

import pytest

from agent.tools.edit_file import EditFileTool
from agent.tools.workspace import Workspace


def _tool(tmp_path: Path, *, max_bytes: int = 1_000_000) -> EditFileTool:
    return EditFileTool(Workspace(tmp_path), max_bytes=max_bytes)


async def test_replaces_one_unique_text_match(tmp_path: Path):
    target = tmp_path / "greeting.txt"
    target.write_text("hello, NAME!", encoding="utf-8")

    result = await _tool(tmp_path).execute(
        {
            "path": "greeting.txt",
            "old_text": "NAME",
            "new_text": "Sanjay",
        }
    )

    assert result.is_error is False
    assert result.content == "edited greeting.txt (replaced 1 occurrence)"
    assert target.read_text(encoding="utf-8") == "hello, Sanjay!"


async def test_refuses_an_ambiguous_edit_without_changing_the_file(tmp_path: Path):
    target = tmp_path / "repeated.txt"
    target.write_text("draft\ndraft\n", encoding="utf-8")

    result = await _tool(tmp_path).execute(
        {
            "path": "repeated.txt",
            "old_text": "draft",
            "new_text": "final",
        }
    )

    assert result.is_error is True
    assert result.content == "old_text occurs 2 times; provide a unique match"
    assert target.read_text(encoding="utf-8") == "draft\ndraft\n"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {
                "path": "notes.txt",
                "old_text": "",
                "new_text": "replacement",
            },
            "old_text must not be empty",
        ),
        (
            {
                "path": "notes.txt",
                "old_text": "missing",
                "new_text": "replacement",
            },
            "old_text was not found in the file",
        ),
        (
            {
                "path": "../outside.txt",
                "old_text": "old",
                "new_text": "new",
            },
            "path escapes the workspace",
        ),
        (
            {
                "path": "notes.txt",
                "old_text": "old",
                "new_text": "new",
                "unknown": True,
            },
            "unexpected arguments: unknown",
        ),
    ],
)
async def test_rejects_invalid_edit_requests(
    tmp_path: Path,
    arguments: dict,
    message: str,
):
    (tmp_path / "notes.txt").write_text("old", encoding="utf-8")

    result = await _tool(tmp_path).execute(arguments)

    assert result.is_error is True
    assert result.content == message


async def test_refuses_binary_input(tmp_path: Path):
    (tmp_path / "binary.dat").write_bytes(b"before\x00after")

    result = await _tool(tmp_path).execute(
        {
            "path": "binary.dat",
            "old_text": "before",
            "new_text": "after",
        }
    )

    assert result.is_error is True
    assert result.content == "binary files cannot be edited as text"


async def test_refuses_an_edit_that_would_exceed_the_size_limit(tmp_path: Path):
    target = tmp_path / "notes.txt"
    target.write_text("one", encoding="utf-8")

    result = await _tool(tmp_path, max_bytes=4).execute(
        {
            "path": "notes.txt",
            "old_text": "one",
            "new_text": "longer",
        }
    )

    assert result.is_error is True
    assert result.content == "edited file exceeds the 4-byte limit"
    assert target.read_text(encoding="utf-8") == "one"
