from __future__ import annotations

import asyncio
import sys

from agent.sandbox.base import (
    SandboxBackend,
    SandboxCommand,
    SandboxError,
    SandboxUnavailableError,
    SandboxViolationError,
    validate_command,
)
from agent.sandbox.linux import LinuxSandbox
from agent.sandbox.macos import MacOSSandbox
from agent.sandbox.policy import SandboxMode, SandboxPolicy


class DisabledSandbox(SandboxBackend):
    """
    Explicit development-only backend with no OS isolation.

    This must never be selected by accident. It exists only when the
    policy mode is explicitly set to "disabled".
    """

    def __init__(self, policy: SandboxPolicy) -> None:
        if policy.mode is not SandboxMode.DISABLED:
            raise SandboxViolationError(
                "DisabledSandbox requires mode='disabled'"
            )

        self._policy = policy

    @property
    def name(self) -> str:
        return "disabled-development-sandbox"

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    async def start(
        self,
        command: SandboxCommand,
    ) -> asyncio.subprocess.Process:
        validate_command(command, self._policy)

        try:
            return await asyncio.create_subprocess_shell(
                command.command,
                cwd=str(command.working_directory),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            raise SandboxError(
                f"could not start development command: {exc}"
            ) from exc


def create_sandbox(
    policy: SandboxPolicy,
    *,
    platform_name: str | None = None,
) -> SandboxBackend:
    """
    Create the appropriate sandbox for the current operating system.

    Enforced policies fail closed: if no real backend is available,
    this function raises instead of running an unrestricted command.
    """

    if policy.mode is SandboxMode.DISABLED:
        return DisabledSandbox(policy)

    if policy.mode is SandboxMode.CONTAINER:
        raise SandboxUnavailableError(
            "container sandbox mode is not configured yet"
        )

    current_platform = (
        sys.platform
        if platform_name is None
        else platform_name
    )

    if current_platform == "darwin":
        if not MacOSSandbox.is_available():
            raise SandboxUnavailableError(
                "macOS sandbox-exec is unavailable"
            )

        return MacOSSandbox(policy)

    if current_platform.startswith("linux"):
        if not LinuxSandbox.is_available():
            raise SandboxUnavailableError(
                "Bubblewrap is unavailable; refusing to run "
                "without sandbox enforcement"
            )

        return LinuxSandbox(policy)

    raise SandboxUnavailableError(
        f"no sandbox backend is available for {current_platform!r}"
    )