from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from agent.sandbox.base import (
    SandboxBackend,
    SandboxCommand,
    SandboxError,
    SandboxUnavailableError,
    SandboxViolationError,
    validate_command,
)
from agent.sandbox.policy import SandboxMode, SandboxPolicy


_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")

_SYSTEM_READ_PATHS = (
    Path("/System"),
    Path("/usr/bin"),
    Path("/usr/lib"),
    Path("/usr/share"),
    Path("/bin"),
    Path("/sbin"),
    Path("/Library"),
    Path("/private/etc"),
    Path("/dev/null"),
    Path("/dev/zero"),
    Path("/dev/random"),
    Path("/dev/urandom"),
)


class MacOSSandbox(SandboxBackend):
    """Run commands inside a macOS sandbox-exec boundary."""

    def __init__(
        self,
        policy: SandboxPolicy,
        executable: Path = _SANDBOX_EXEC,
    ) -> None:
        if policy.mode is not SandboxMode.ENFORCED:
            raise SandboxViolationError(
                "MacOSSandbox requires mode='enforced'"
            )

        self._policy = policy
        self._executable = executable.expanduser().resolve()

    @property
    def name(self) -> str:
        return "macos-sandbox-exec"

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    @classmethod
    def is_available(
        cls,
        executable: Path = _SANDBOX_EXEC,
    ) -> bool:
        return (
            sys.platform == "darwin"
            and executable.is_file()
            and os.access(executable, os.X_OK)
        )

    def profile(self) -> str:
        """Build the sandbox-exec profile for this policy."""

        return build_profile(self._policy)

    async def start(
        self,
        command: SandboxCommand,
    ) -> asyncio.subprocess.Process:
        validate_command(command, self._policy)
        self._ensure_available()

        environment = _safe_environment(self._policy)

        try:
            return await asyncio.create_subprocess_exec(
                str(self._executable),
                "-p",
                self.profile(),
                "/bin/sh",
                "-lc",
                command.command,
                cwd=str(command.working_directory),
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            raise SandboxError(
                f"could not start macOS sandbox: {exc}"
            ) from exc

    def _ensure_available(self) -> None:
        if sys.platform != "darwin":
            raise SandboxUnavailableError(
                "macOS sandbox-exec is only available on macOS"
            )

        if not self._executable.is_file():
            raise SandboxUnavailableError(
                "sandbox-exec was not found at "
                f"{self._executable}"
            )

        if not os.access(self._executable, os.X_OK):
            raise SandboxUnavailableError(
                f"sandbox-exec is not executable: {self._executable}"
            )


def build_profile(policy: SandboxPolicy) -> str:
    """Convert a SandboxPolicy into sandbox-exec profile text."""

    if policy.mode is not SandboxMode.ENFORCED:
        raise SandboxViolationError(
            "a macOS profile requires mode='enforced'"
        )

    readable_paths = _unique_paths(
        (
            *_existing_system_read_paths(),
            *policy.readable_paths,
        )
    )
    writable_paths = policy.writable_paths

    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-fork)",
        "(allow process-exec*)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
        "(allow ipc-posix-shm)",
    ]

    lines.extend(
        _deny_path("file-read*", path)
        for path in policy.protected_read_paths
    )
    lines.extend(
        _deny_path("file-write*", path)
        for path in policy.protected_read_paths
    )

    lines.extend(
        _allow_paths(
            "file-read* file-map-executable",
            readable_paths,
        )
    )
    lines.extend(
        _allow_paths(
            "file-read* file-write*",
            writable_paths,
        )
    )

    metadata_paths = _metadata_paths(
        (
            *readable_paths,
            *writable_paths,
        )
    )
    lines.extend(
        _allow_paths("file-read-metadata", metadata_paths)
    )

    if policy.network_allowed:
        lines.append("(allow network-outbound)")

    return "\n".join(lines)


def _existing_system_read_paths() -> tuple[Path, ...]:
    return tuple(
        path.resolve()
        for path in _SYSTEM_READ_PATHS
        if path.exists()
    )


def _allow_paths(
    operations: str,
    paths: tuple[Path, ...],
) -> tuple[str, ...]:
    if not paths:
        return ()

    filters = "\n  ".join(
        _path_filter(path)
        for path in paths
    )
    return (
        f"(allow {operations}\n"
        f"  {filters}\n"
        f")",
    )


def _deny_path(operation: str, path: Path) -> str:
    return (
        f"(deny {operation} "
        f"{_path_filter(path)})"
    )


def _path_filter(path: Path) -> str:
    predicate = "subpath" if path.is_dir() else "literal"
    return f"({predicate} {_quote(path.as_posix())})"


def _metadata_paths(
    paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    metadata: list[Path] = []

    for path in paths:
        for parent in (path, *path.parents):
            if parent not in metadata:
                metadata.append(parent)

    return tuple(metadata)


def _unique_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    unique: list[Path] = []

    for path in paths:
        if path not in unique:
            unique.append(path)

    return tuple(unique)


def _quote(value: str) -> str:
    if "\x00" in value or "\n" in value or "\r" in value:
        raise SandboxViolationError(
            "sandbox paths must not contain control characters"
        )

    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _safe_environment(
    policy: SandboxPolicy,
) -> dict[str, str]:
    """Pass only non-sensitive environment values to the command."""

    path_entries = [
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ]

    for runtime_path in policy.runtime_read_paths:
        entry = (
            runtime_path
            if runtime_path.is_dir()
            else runtime_path.parent
        )
        text = str(entry)

        if text not in path_entries:
            path_entries.append(text)

    temporary_directory = (
        policy.temporary_directory
        if policy.temporary_directory is not None
        else Path("/tmp")
    )

    return {
        "HOME": str(policy.workspace_root),
        "PATH": ":".join(path_entries),
        "TMPDIR": str(temporary_directory),
        "LANG": "C.UTF-8",
    }
