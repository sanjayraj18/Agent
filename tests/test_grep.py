from pathlib import Path

import pytest

from agent.tools.grep import GrepTool
from agent.tools.workspace import Workspace


def _tool(
    tmp_path: Path,
    *,
    max_results: int = 200,
    max_file_bytes: int = 1_000_000,
) -> GrepTool:
    return GrepTool(
        Workspace(tmp_path),
        max_results=max_results,
        max_file_bytes=max_file_bytes,
    )


async def test_searches_sorted_workspace_files_and_reports_line_numbers(
    tmp_path: Path,
):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "# TODO: implement\nprint('ready')\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "notes.txt").write_text(
        "TODO: cover failures\n",
        encoding="utf-8",
    )
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "ignored.txt").write_text(
        "TODO: ignored\n",
        encoding="utf-8",
    )

    result = await _tool(tmp_path).execute({"pattern": "TODO"})

    assert result.is_error is False
    assert result.content == (
        "src/app.py:1: # TODO: implement\n"
        "tests/notes.txt:1: TODO: cover failures"
    )


async def test_accepts_regular_expressions_and_a_single_file_path(tmp_path: Path):
    (tmp_path / "values.py").write_text(
        "value = 41\nvalue = text\nvalue = 42\n",
        encoding="utf-8",
    )

    result = await _tool(tmp_path).execute(
        {
            "pattern": r"^value = \d+$",
            "path": "values.py",
        }
    )

    assert result.is_error is False
    assert result.content == "values.py:1: value = 41\nvalues.py:3: value = 42"


async def test_skips_binary_non_utf8_oversized_and_escaping_files(tmp_path: Path):
    (tmp_path / "binary.dat").write_bytes(b"TODO\x00hidden")
    (tmp_path / "not_text.dat").write_bytes(b"TODO\xff")
    (tmp_path / "large.txt").write_text("TODO large", encoding="utf-8")
    outside = tmp_path.parent / "grep-outside.txt"
    outside.write_text("TODO outside", encoding="utf-8")
    (tmp_path / "outside-link.txt").symlink_to(outside)

    result = await _tool(tmp_path, max_file_bytes=4).execute({"pattern": "TODO"})

    assert result.is_error is False
    assert result.content == "no matches"


async def test_limits_the_number_of_returned_matching_lines(tmp_path: Path):
    (tmp_path / "notes.txt").write_text(
        "TODO one\nTODO two\nTODO three\n",
        encoding="utf-8",
    )

    result = await _tool(tmp_path, max_results=2).execute({"pattern": "TODO"})

    assert result.is_error is False
    assert result.content == (
        "notes.txt:1: TODO one\n"
        "notes.txt:2: TODO two\n"
        "[results truncated at 2 matches]"
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"pattern": "["}, "invalid regular expression:"),
        ({"pattern": ""}, "pattern must be a non-empty string"),
        ({"pattern": "TODO", "path": "../outside"}, "path escapes the workspace"),
        ({"pattern": "TODO", "path": "missing"}, "path does not exist: missing"),
        ({"pattern": "TODO", "max_results": True}, "max_results must be an integer"),
        ({"pattern": "TODO", "max_results": 0}, "max_results must be between 1 and 200"),
        ({"pattern": "TODO", "unknown": True}, "unexpected arguments: unknown"),
    ],
)
async def test_rejects_invalid_search_requests(
    tmp_path: Path,
    arguments: dict,
    message: str,
):
    result = await _tool(tmp_path).execute(arguments)

    assert result.is_error is True
    if message.endswith(":"):
        assert result.content.startswith(message)
    else:
        assert result.content == message
