"""Fresh, independent workspaces for every benchmark attempt."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile


class IsolationError(RuntimeError):
    """A benchmark fixture cannot be safely isolated."""


@dataclass(frozen=True, slots=True)
class IsolatedWorkspace:
    """One private copy of a task fixture and its original file snapshot."""

    root: Path
    workspace: Path
    temporary_directory: Path
    baseline: dict[str, str]

    def changed_paths(self) -> tuple[str, ...]:
        """Return added, removed, or modified regular files since setup."""
        current = snapshot_tree(self.workspace)
        return tuple(
            sorted(
                path
                for path in set(self.baseline) | set(current)
                if self.baseline.get(path) != current.get(path)
            )
        )

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def create_isolated_workspace(
    fixture: Path,
    *,
    run_root: Path,
    run_id: str,
) -> IsolatedWorkspace:
    """Copy a self-contained fixture into a private directory for one run."""
    source = fixture.expanduser().resolve()
    destination_root = run_root.expanduser().resolve()

    if not source.is_dir():
        raise IsolationError(f"fixture must be a directory: {source}")
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in run_id):
        raise IsolationError("run_id may contain only lowercase letters, digits, '-' and '_'")

    _reject_symlinks(source)
    destination_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=destination_root))
    workspace = root / "workspace"
    temporary_directory = root / "tmp"

    try:
        shutil.copytree(source, workspace)
        temporary_directory.mkdir()
        return IsolatedWorkspace(
            root=root,
            workspace=workspace,
            temporary_directory=temporary_directory,
            baseline=snapshot_tree(workspace),
        )
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def snapshot_tree(root: Path) -> dict[str, str]:
    """Hash regular files by their relative path without following symlinks."""
    resolved_root = root.expanduser().resolve()
    snapshot: dict[str, str] = {}

    for path in sorted(resolved_root.rglob("*")):
        if path.is_symlink():
            raise IsolationError(f"symlinks are not allowed in benchmark fixtures: {path}")
        if path.is_file():
            relative = path.relative_to(resolved_root).as_posix()
            snapshot[relative] = _sha256_file(path)

    return snapshot


def _reject_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise IsolationError(
                f"benchmark fixture contains a symlink and cannot be isolated: {path}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
