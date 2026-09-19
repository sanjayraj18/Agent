from pathlib import Path

import pytest

from agent.tools.workspace import Workspace, WorkspacePathError


def test_resolves_a_relative_path_inside_the_workspace(tmp_path: Path):
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()

    workspace = Workspace(workspace_root)

    assert workspace.resolve("src/agent/events.py") == (
        workspace_root / "src/agent/events.py"
    )


def test_normalizes_the_workspace_root_to_an_absolute_path(tmp_path: Path):
    workspace = Workspace(tmp_path)

    assert workspace.root.is_absolute()
    assert workspace.root == tmp_path.resolve()


@pytest.mark.parametrize(
    ("path", "error_message"),
    [
        ("", "path must not be empty"),
        ("/etc/passwd", "absolute paths are not allowed"),
        ("../outside.txt", "path escapes the workspace"),
        ("nested/../../outside.txt", "path escapes the workspace"),
        ("bad\x00path", "path must not contain a null byte"),
    ],
)
def test_rejects_invalid_or_escaping_paths(
    tmp_path: Path,
    path: str,
    error_message: str,
):
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspacePathError, match=error_message):
        workspace.resolve(path)


def test_rejects_a_symlink_that_escapes_the_workspace(tmp_path: Path):
    workspace_root = tmp_path / "project"
    outside_root = tmp_path / "outside"

    workspace_root.mkdir()
    outside_root.mkdir()

    link = workspace_root / "outside_link"
    link.symlink_to(outside_root, target_is_directory=True)

    workspace = Workspace(workspace_root)

    with pytest.raises(
        WorkspacePathError,
        match="path escapes the workspace",
    ):
        workspace.resolve("outside_link/secret.txt")


def test_returns_a_workspace_relative_display_path(tmp_path: Path):
    workspace = Workspace(tmp_path)
    file_path = tmp_path / "src" / "agent" / "events.py"

    assert workspace.relative(file_path) == "src/agent/events.py"


def test_rejects_an_external_path_when_making_a_display_path(tmp_path: Path):
    workspace = Workspace(tmp_path)
    external_path = tmp_path.parent / "outside.txt"

    with pytest.raises(
        WorkspacePathError,
        match="path escapes the workspace",
    ):
        workspace.relative(external_path)