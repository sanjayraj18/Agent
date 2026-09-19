from pathlib import Path

import pytest

from agent.tools.read_file import ReadFileTool
from agent.tools.workspace import Workspace


def _tool(
    tmp_path: Path,
    *,
    max_bytes: int = 100_000,
    max_lines: int = 2_000,
) -> ReadFileTool:
    return ReadFileTool(
        Workspace(tmp_path),
        max_bytes=max_bytes,
        max_lines=max_lines,
    )


async def test_reads_a_text_file_with_line_numbers(tmp_path: Path):
    (tmp_path / "notes.txt").write_text(
        "first line\nsecond line\nthird line\n",
        encoding="utf-8",
    )

    result = await _tool(tmp_path).execute(
        {"path": "notes.txt"}
    )

    assert result.is_error is False
    assert result.content == (
        "     1: first line\n"
        "     2: second line\n"
        "     3: third line"
    )


async def test_reads_an_inclusive_requested_line_range(tmp_path: Path):
    (tmp_path / "notes.txt").write_text(
        "one\ntwo\nthree\nfour\n",
        encoding="utf-8",
    )

    result = await _tool(tmp_path).execute(
        {
            "path": "notes.txt",
            "start_line": 2,
            "end_line": 3,
        }
    )

    assert result.is_error is False
    assert result.content == (
        "     2: two\n"
        "     3: three"
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"path": "missing.txt"}, "file does not exist"),
        ({"path": "../outside.txt"}, "path escapes the workspace"),
        ({"path": "folder"}, "path is not a regular file"),
    ],
)
async def test_rejects_missing_outside_or_non_file_paths(
    tmp_path: Path,
    arguments: dict,
    message: str,
):
    (tmp_path / "folder").mkdir()

    result = await _tool(tmp_path).execute(arguments)

    assert result.is_error is True
    assert message in result.content


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"path": "notes.txt", "start_line": 0},
            "start_line must be at least 1",
        ),
        (
            {"path": "notes.txt", "start_line": True},
            "start_line must be an integer",
        ),
        (
            {
                "path": "notes.txt",
                "start_line": 4,
                "end_line": 2,
            },
            "end_line must not be less than start_line",
        ),
        (
            {"path": "notes.txt", "unknown": "value"},
            "unexpected arguments: unknown",
        ),
    ],
)
async def test_rejects_invalid_arguments(
    tmp_path: Path,
    arguments: dict,
    message: str,
):
    (tmp_path / "notes.txt").write_text(
        "one\ntwo\nthree\n",
        encoding="utf-8",
    )

    result = await _tool(tmp_path).execute(arguments)

    assert result.is_error is True
    assert result.content == message


async def test_rejects_binary_and_non_utf8_files(tmp_path: Path):
    (tmp_path / "binary.bin").write_bytes(b"one\x00two")
    (tmp_path / "non_utf8.txt").write_bytes(b"\xff\xfe")

    tool = _tool(tmp_path)

    binary_result = await tool.execute({"path": "binary.bin"})
    non_utf8_result = await tool.execute({"path": "non_utf8.txt"})

    assert binary_result.is_error is True
    assert binary_result.content == "binary files cannot be read as text"

    assert non_utf8_result.is_error is True
    assert non_utf8_result.content == "file is not valid UTF-8 text"


async def test_truncates_line_output_at_the_configured_limit(
    tmp_path: Path,
):
    (tmp_path / "notes.txt").write_text(
        "one\ntwo\nthree\n",
        encoding="utf-8",
    )

    result = await _tool(
        tmp_path,
        max_lines=2,
    ).execute({"path": "notes.txt"})

    assert result.is_error is False
    assert result.content == (
        "     1: one\n"
        "     2: two\n"
        "[line output truncated at line 2]"
    )


async def test_marks_byte_truncated_output(tmp_path: Path):
    (tmp_path / "notes.txt").write_text(
        "first line\nsecond line\n",
        encoding="utf-8",
    )

    result = await _tool(
        tmp_path,
        max_bytes=10,
    ).execute({"path": "notes.txt"})

    assert result.is_error is False
    assert "[file output truncated after 10 bytes]" in result.content