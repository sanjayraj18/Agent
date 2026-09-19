from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class WorkspacePathError(ValueError):
    """Raised when a path is invalid or escapes the workspace root."""


@dataclass(frozen=True, slots=True)
class Workspace:
    root : Path

    def __post_init__(self) -> None:
        resolved_root = self.root.expanduser().resolve()

        if not resolved_root.is_dir():
            raise ValueError(f"workspace root is not a directory: {resolved_root}")

        object.__setattr__(self, "root", resolved_root)

    @classmethod
    def from_cwd(cls) -> Workspace:
        return cls(Path.cwd())

    def resolve(self, relative_path : str) -> Path:
        if not relative_path:
            raise WorkspacePathError("path must not be empty")

        if "\x00" in relative_path:
            raise WorkspacePathError("path must not contain a null byte")

        path = Path(relative_path)

        if path.is_absolute():
            raise WorkspacePathError(
                "absolute paths are not allowed"
            )

        candidate = (self.root / path).resolve()

        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspacePathError(
                "path escapes the workspace"
            ) from exc

        return candidate

    def relative(self, path:Path) -> str:
        resolved_path = path.resolve()
        try:
            return resolved_path.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise WorkspacePathError(
                "path escapes the workspace"
            ) from exc
