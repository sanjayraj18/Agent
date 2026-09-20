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


_BWRAP = Path("/usr/bin/bwrap")

_SYSTEM_READ_PATHS = (
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/lib"),
    Path("/lib64"),
    Path("/etc"),
)


class LinuxSandbox(SandboxBackend):
    """Run commands inside a Bubblewrap Linux sandbox."""

    def __init__(
        self,
        policy: SandboxPolicy,
        executable: Path = _BWRAP,
    ) -> None:
        if policy.mode is not SandboxMode.ENFORCED:
            raise SandboxViolationError(
                "LinuxSandbox requires mode='enforced'"
            )

        self._policy = policy
        self._executable = executable.expanduser().resolve()

    @property
    def name(self) -> str:
        return "linux-bubblewrap"

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    @classmethod
    def is_available(
        cls,
        executable: Path = _BWRAP,
    ) -> bool:
        return (
            sys.platform.startswith("linux")
            and executable.is_file()
            and os.access(executable, os.X_OK)
        )

    async def start(
        self,
        command: SandboxCommand,
    ) -> asyncio.subprocess.Process:
        validate_command(command, self._policy)
        self._ensure_available()

        arguments = build_bwrap_command(
            self._policy,
            command,
            executable=self._executable,
        )

        try:
            return await asyncio.create_subprocess_exec(
                *arguments,
                cwd=str(command.working_directory),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            raise SandboxError(
                f"could not start Bubblewrap sandbox: {exc}"
            ) from exc

    def _ensure_available(self) -> None:
        if not sys.platform.startswith("linux"):
            raise SandboxUnavailableError(
                "Bubblewrap sandboxing is only available on Linux"
            )

        if not self._executable.is_file():
            raise SandboxUnavailableError(
                f"Bubblewrap was not found at {self._executable}"
            )

        if not os.access(self._executable, os.X_OK):
            raise SandboxUnavailableError(
                f"Bubblewrap is not executable: {self._executable}"
            )


def build_bwrap_command(
    policy: SandboxPolicy,
    command: SandboxCommand,
    *,
    executable: Path = _BWRAP,
) -> tuple[str, ...]:
    """Build Bubblewrap arguments without executing them."""

    if policy.mode is not SandboxMode.ENFORCED:
        raise SandboxViolationError(
            "a Bubblewrap command requires mode='enforced'"
        )

    validate_command(command, policy)

    runtime_paths = _unique_paths(
        (
            *_existing_system_read_paths(),
            *policy.runtime_read_paths,
        )
    )
    writable_paths = policy.writable_paths
    destination_dirs = _destination_directories(
        (
            *runtime_paths,
            *writable_paths,
        )
    )

    arguments = [
        str(executable),
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--clearenv",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]

    if policy.network_allowed:
        arguments.append("--share-net")

    for directory in destination_dirs:
        arguments.extend(("--dir", str(directory)))

    for path in runtime_paths:
        arguments.extend(
            ("--ro-bind", str(path), str(path))
        )

    for path in writable_paths:
        arguments.extend(
            ("--bind", str(path), str(path))
        )

    arguments.extend(_environment_arguments(policy))
    arguments.extend(
        (
            "--chdir",
            str(command.working_directory),
            "/bin/sh",
            "-lc",
            command.command,
        )
    )

    return tuple(arguments)


def _existing_system_read_paths() -> tuple[Path, ...]:
    return tuple(
        path
        for path in _SYSTEM_READ_PATHS
        if path.exists()
    )


def _environment_arguments(
    policy: SandboxPolicy,
) -> tuple[str, ...]:
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

    return (
        "--setenv",
        "HOME",
        str(policy.workspace_root),
        "--setenv",
        "PATH",
        ":".join(path_entries),
        "--setenv",
        "TMPDIR",
        str(temporary_directory),
        "--setenv",
        "LANG",
        "C.UTF-8",
    )


def _destination_directories(
    paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    directories: list[Path] = []

    for path in paths:
        current = path if path.is_dir() else path.parent

        while current != Path("/"):
            if current not in directories:
                directories.append(current)
            current = current.parent

    return tuple(
        sorted(
            directories,
            key=lambda path: len(path.parts),
        )
    )


def _unique_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    unique: list[Path] = []

    for path in paths:
        if path not in unique:
            unique.append(path)

    return tuple(unique)