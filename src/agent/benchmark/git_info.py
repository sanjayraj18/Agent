"""Capture the exact agent revision that a benchmark measured."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


class GitInfoError(RuntimeError):
    """The current source tree cannot provide reproducible Git provenance."""


@dataclass(frozen=True, slots=True)
class GitProvenance:
    revision: str
    is_dirty: bool


def read_git_provenance(root: Path) -> GitProvenance:
    """Return the checked-out commit and whether uncommitted changes exist."""
    working_directory = root.expanduser().resolve()

    revision = _git(working_directory, "rev-parse", "HEAD")
    dirty = bool(_git(working_directory, "status", "--porcelain"))

    return GitProvenance(revision=revision, is_dirty=dirty)


def require_clean_tree(provenance: GitProvenance) -> None:
    """Reject a score that cannot be tied to one committed implementation."""
    if provenance.is_dirty:
        raise GitInfoError(
            "benchmark requires a clean Git worktree; commit or stash changes first"
        )


def _git(working_directory: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=working_directory,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise GitInfoError(f"could not run git: {exc}") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GitInfoError(f"git {' '.join(arguments)} failed: {detail}")

    return completed.stdout.strip()
