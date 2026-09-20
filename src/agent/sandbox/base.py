from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from agent.sandbox.policy import SandboxPolicy


class SandboxError(RuntimeError):
    """Base error for sandbox startup and enforcement failures."""


class SandboxUnavailableError(SandboxError):
    """Raised when the requested sandbox backend is unavailable."""


class SandboxViolationError(SandboxError):
    """Raised when a command violates the sandbox policy."""


@dataclass(frozen=True, slots=True)
class SandboxCommand:
    """A shell command that must run inside the sandbox."""

    command: str
    working_directory: Path

    def __post_init__(self) -> None:
        if not isinstance(self.command, str) or not self.command.strip():
            raise ValueError("command must be a non-empty string")

        if "\x00" in self.command:
            raise ValueError("command must not contain a null byte")

        if not isinstance(self.working_directory, Path):
            raise ValueError(
                "working_directory must be a pathlib.Path"
            )

        working_directory = (
            self.working_directory.expanduser().resolve()
        )

        if not working_directory.is_dir():
            raise ValueError(
                "working_directory must be a directory: "
                f"{working_directory}"
            )

        object.__setattr__(
            self,
            "working_directory",
            working_directory,
        )


@runtime_checkable
class SandboxBackend(Protocol):
    """The common interface implemented by every sandbox backend."""

    @property
    def name(self) -> str:
        """A human-readable backend name, such as 'macos-sandbox-exec'."""
        ...

    @property
    def policy(self) -> SandboxPolicy:
        """The policy enforced by this backend."""
        ...

    async def start(
        self,
        command: SandboxCommand,
    ) -> asyncio.subprocess.Process:
        """
        Start one restricted command.

        The returned process must have stdout configured as PIPE and
        stderr merged into stdout. BashTool already knows how to read,
        truncate, time out, and stop that process.
        """
        ...


def validate_command(
    command: SandboxCommand,
    policy: SandboxPolicy,
) -> None:
    """Reject a command whose working directory escapes the workspace."""

    try:
        command.working_directory.relative_to(
            policy.workspace_root
        )
    except ValueError as exc:
        raise SandboxViolationError(
            "command working directory escapes the sandbox workspace: "
            f"{command.working_directory}"
        ) from exc
