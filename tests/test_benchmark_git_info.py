from pathlib import Path

import pytest

from agent.benchmark.git_info import GitInfoError, GitProvenance, read_git_provenance, require_clean_tree


class _Completed:
    def __init__(self, stdout: str) -> None:
        self.returncode = 0
        self.stdout = stdout
        self.stderr = ""


def test_git_provenance_captures_commit_and_dirty_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls: list[tuple[str, ...]] = []

    def fake_run(arguments, **_kwargs):
        calls.append(arguments)
        if arguments[1:] == ("rev-parse", "HEAD"):
            return _Completed("a" * 40 + "\n")
        return _Completed(" M src/agent/core/loop.py\n")

    monkeypatch.setattr("agent.benchmark.git_info.subprocess.run", fake_run)

    provenance = read_git_provenance(tmp_path)

    assert provenance == GitProvenance(revision="a" * 40, is_dirty=True)
    assert calls == [
        ("git", "rev-parse", "HEAD"),
        ("git", "status", "--porcelain"),
    ]


def test_clean_tree_requirement_rejects_uncommitted_changes():
    with pytest.raises(GitInfoError, match="clean Git worktree"):
        require_clean_tree(GitProvenance(revision="a" * 40, is_dirty=True))
