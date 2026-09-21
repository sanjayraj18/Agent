"""Pinned-container verification for benchmark tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import shutil

from agent.benchmark.models import CommandSpec


class ContainerUnavailableError(RuntimeError):
    """The requested reproducible container boundary is unavailable."""


class PinnedImageError(ValueError):
    """The image is not immutable enough for a reproducible score."""


@dataclass(frozen=True, slots=True)
class CommandOutput:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class LocalCommandExecutor:
    """Test-only executor; production benchmark commands use Docker."""

    async def run(
        self,
        command: CommandSpec,
        workspace: Path,
    ) -> CommandOutput:
        return await _run_process(command.argv, workspace, command.timeout_seconds)


class DockerCommandExecutor:
    """Run verifier commands in a pinned, network-isolated container."""

    def __init__(self, image: str, *, executable: str = "docker") -> None:
        self._image = validate_pinned_image(image)
        self._executable = executable

    @property
    def image(self) -> str:
        return self._image

    def ensure_available(self) -> None:
        if shutil.which(self._executable) is None:
            raise ContainerUnavailableError(
                "Docker is required for benchmark verification but was not found"
            )

    async def run(
        self,
        command: CommandSpec,
        workspace: Path,
    ) -> CommandOutput:
        self.ensure_available()
        resolved_workspace = workspace.expanduser().resolve()

        arguments = (
            self._executable,
            "run",
            "--rm",
            "--init",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=128m",
            "--mount",
            f"type=bind,src={resolved_workspace},dst=/workspace",
            "--workdir",
            "/workspace",
            self._image,
            *command.argv,
        )
        return await _run_process(arguments, resolved_workspace, command.timeout_seconds)


def validate_pinned_image(image: str) -> str:
    normalized = image.strip()
    marker = "@sha256:"

    if marker not in normalized:
        raise PinnedImageError("container image must include @sha256:<digest>")

    digest = normalized.split(marker, maxsplit=1)[1]
    if (
        len(digest) != 64
        or not all(character in "0123456789abcdef" for character in digest)
    ):
        raise PinnedImageError("container image must include a SHA-256 digest")

    return normalized


async def _run_process(
    arguments: tuple[str, ...],
    workspace: Path,
    timeout_seconds: int,
) -> CommandOutput:
    try:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            cwd=str(workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise ContainerUnavailableError(
            f"could not start verification command: {exc}"
        ) from exc

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        process.kill()
        stdout, stderr = await process.communicate()
        return CommandOutput(
            exit_code=-1,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            timed_out=True,
        )

    return CommandOutput(
        exit_code=process.returncode or 0,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )
