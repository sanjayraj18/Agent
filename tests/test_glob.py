from pathlib import Path

import pytest

from agent.tools.glob import GlobTool
from agent.tools.workspace import Workspace


def _tool(
    tmp_path: Path,
    *,
    max_results: int = 200,
) -> GlobTool:
    return GlobTool(
        Workspace(tmp_path),
        max_results=max_results,
    )


async def test_finds_sorted_workspace_relative_paths(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "utils.py").write_text("", encoding="utf-8")

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text(
        "",
        encoding="utf-8",
    )

    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hidden.py").write_text(
        "",
        encoding="utf-8",
    )

    result = await _tool(tmp_path).execute(
        {"pattern": "**/*.py"}
    )

    assert result.is_error is False
    assert result.content == (
        "src/main.py\n"
        "src/utils.py\n"
        "tests/test_main.py"
    )


async def test_marks_directory_results_with_a_trailing_slash(
    tmp_path: Path,
):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "nested").mkdir()
    (tmp_path / "src" / "main.py").write_text("", encoding="utf-8")

    result = await _tool(tmp_path).execute({"pattern": "src/*"})

    assert result.is_error is False
    assert result.content == (
        "src/main.py\n"
        "src/nested/"
    )


async def test_reports_when_no_paths_match(tmp_path: Path):
    result = await _tool(tmp_path).execute(
        {"pattern": "**/*.py"}
    )

    assert result.is_error is False
    assert result.content == "no matches"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"pattern": "../*.py"},
            "pattern must not contain '..'",
        ),
        (
            {"pattern": "/tmp/*.py"},
            "absolute patterns are not allowed",
        ),
        (
            {"pattern": ""},
            "pattern must be a non-empty string",
        ),
        (
            {"pattern": "*.py", "max_results": True},
            "max_results must be an integer",
        ),
        (
            {"pattern": "*.py", "max_results": 0},
            "max_results must be between 1 and 200",
        ),
        (
            {"pattern": "*.py", "unknown": "value"},
            "unexpected arguments: unknown",
        ),
    ],
)
async def test_rejects_invalid_glob_arguments(
    tmp_path: Path,
    arguments: dict,
    message: str,
):
    result = await _tool(tmp_path).execute(arguments)

    assert result.is_error is True
    assert result.content == message


async def test_truncates_large_result_sets(tmp_path: Path):
    for number in range(3):
        (tmp_path / f"file_{number}.py").write_text(
            "",
            encoding="utf-8",
        )

    result = await _tool(
        tmp_path,
        max_results=2,
    ).execute({"pattern": "*.py"})

    lines = result.content.splitlines()

    assert result.is_error is False
    assert len(lines) == 3
    assert lines[-1] == "[results truncated at 2 paths]"


async def test_hides_a_symlink_that_escapes_the_workspace(tmp_path: Path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("", encoding="utf-8")

    (tmp_path / "outside_link.py").symlink_to(outside)

    result = await _tool(tmp_path).execute({"pattern": "*.py"})

    assert result.is_error is False
    assert result.content == "no matches"