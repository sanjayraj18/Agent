from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class SandboxPolicyError(ValueError):
    """Raised when sandbox rules would create an unsafe boundary."""


class SandboxMode(StrEnum):
    """How the command should be isolated."""

    ENFORCED = "enforced"
    DISABLED = "disabled"
    CONTAINER = "container"


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    """The security rules a sandbox backend must enforce."""

    workspace_root: Path
    network_allowed: bool = False
    temporary_directory: Path | None = None
    runtime_read_paths: tuple[Path, ...] = ()
    mode: SandboxMode = SandboxMode.ENFORCED
    protected_read_paths: tuple[Path, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, SandboxMode):
            raise SandboxPolicyError(
                "mode must be a SandboxMode value"
            )

        if not isinstance(self.network_allowed, bool):
            raise SandboxPolicyError(
                "network_allowed must be a boolean"
            )

        workspace_root = _resolve_directory(
            self.workspace_root,
            label="workspace root",
        )

        temporary_directory = None
        if self.temporary_directory is not None:
            temporary_directory = _resolve_directory(
                self.temporary_directory,
                label="temporary directory",
            )

        runtime_read_paths = tuple(
            _resolve_path(path, label="runtime read path")
            for path in self.runtime_read_paths
        )

        protected_read_paths = tuple(
            path.resolve()
            for path in (
                Path.home() / ".ssh",
                Path.home() / ".aws",
                Path.home() / ".gnupg",
            )
        )

        for protected_path in protected_read_paths:
            _reject_overlap(
                workspace_root,
                protected_path,
                left_name="workspace root",
                right_name="protected path",
            )

            if temporary_directory is not None:
                _reject_overlap(
                    temporary_directory,
                    protected_path,
                    left_name="temporary directory",
                    right_name="protected path",
                )

            for runtime_path in runtime_read_paths:
                _reject_overlap(
                    runtime_path,
                    protected_path,
                    left_name="runtime read path",
                    right_name="protected path",
                )

        object.__setattr__(
            self,
            "workspace_root",
            workspace_root,
        )
        object.__setattr__(
            self,
            "temporary_directory",
            temporary_directory,
        )
        object.__setattr__(
            self,
            "runtime_read_paths",
            _unique_paths(runtime_read_paths),
        )
        object.__setattr__(
            self,
            "protected_read_paths",
            protected_read_paths,
        )

    @property
    def readable_paths(self) -> tuple[Path, ...]:
        """Paths the sandbox may expose as readable."""

        return _unique_paths(
            (
                self.workspace_root,
                *self.runtime_read_paths,
            )
        )

    @property
    def writable_paths(self) -> tuple[Path, ...]:
        """Paths the sandbox may expose as writable."""

        paths = [self.workspace_root]

        if self.temporary_directory is not None:
            paths.append(self.temporary_directory)

        return _unique_paths(tuple(paths))

    @property
    def requires_enforcement(self) -> bool:
        """Whether a real OS/container boundary is required."""

        return self.mode is not SandboxMode.DISABLED


def _resolve_directory(path: Path, *, label: str) -> Path:
    resolved = _resolve_path(path, label=label)

    if not resolved.is_dir():
        raise SandboxPolicyError(
            f"{label} must be a directory: {resolved}"
        )

    return resolved


def _resolve_path(path: Path, *, label: str) -> Path:
    if not isinstance(path, Path):
        raise SandboxPolicyError(
            f"{label} must be a pathlib.Path"
        )

    resolved = path.expanduser().resolve()

    if not resolved.exists():
        raise SandboxPolicyError(
            f"{label} does not exist: {resolved}"
        )

    return resolved


def _reject_overlap(
    left: Path,
    right: Path,
    *,
    left_name: str,
    right_name: str,
) -> None:
    if _overlaps(left, right):
        raise SandboxPolicyError(
            f"{left_name} must not overlap {right_name}: "
            f"{left} and {right}"
        )


def _overlaps(first: Path, second: Path) -> bool:
    return _is_inside(first, second) or _is_inside(second, first)


def _is_inside(path: Path, possible_parent: Path) -> bool:
    try:
        path.relative_to(possible_parent)
    except ValueError:
        return False

    return True


def _unique_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    unique: list[Path] = []

    for path in paths:
        if path not in unique:
            unique.append(path)

    return tuple(unique)