from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def read_limited_stream(
    stream: asyncio.StreamReader,
    max_bytes: int,
) -> tuple[bytes, bool]:
    """Drain a subprocess stream without retaining unbounded output."""
    output = bytearray()
    truncated = False

    while chunk := await stream.read(8_192):
        remaining = max_bytes - len(output)
        if remaining > 0:
            output.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True

    return bytes(output), truncated


@dataclass(frozen=True, slots=True)
class ProcessSnapshot:
    job_id: str
    pid: int
    command: str
    status: str
    return_code: int | None
    output: str
    output_truncated: bool
    started_at: datetime


@dataclass(slots=True)
class _Job:
    job_id: str
    command: str
    process: asyncio.subprocess.Process
    started_at: datetime = field(default_factory=_now)
    output: bytearray = field(default_factory=bytearray)
    output_truncated: bool = False
    reader_task: asyncio.Task[None] | None = None


class ProcessRegistry:
    """Tracks background subprocesses started by the bash tool."""

    def __init__(self, max_output_bytes: int = 100_000) -> None:
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be at least 1")
        self._max_output_bytes = max_output_bytes
        self._jobs: dict[str, _Job] = {}

    async def register(
        self,
        command: str,
        process: asyncio.subprocess.Process,
    ) -> ProcessSnapshot:
        if process.stdout is None:
            raise ValueError("background process must have stdout configured")

        job = _Job(
            job_id=uuid4().hex[:12],
            command=command,
            process=process,
        )
        self._jobs[job.job_id] = job
        job.reader_task = asyncio.create_task(self._capture_output(job))
        snapshot = self.snapshot(job.job_id)
        assert snapshot is not None
        return snapshot

    def snapshot(self, job_id: str) -> ProcessSnapshot | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None

        return_code = job.process.returncode
        status = "running"
        if return_code is not None:
            status = "completed" if return_code == 0 else "failed"

        return ProcessSnapshot(
            job_id=job.job_id,
            pid=job.process.pid,
            command=job.command,
            status=status,
            return_code=return_code,
            output=job.output.decode("utf-8", errors="replace"),
            output_truncated=job.output_truncated,
            started_at=job.started_at,
        )

    async def wait(
        self,
        job_id: str,
        timeout: float | None = None,
    ) -> ProcessSnapshot | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None

        try:
            await asyncio.wait_for(job.process.wait(), timeout=timeout)
        except TimeoutError:
            return self.snapshot(job_id)

        if job.reader_task is not None:
            await job.reader_task
        return self.snapshot(job_id)

    async def stop(self, job_id: str) -> ProcessSnapshot | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None

        if job.process.returncode is None:
            job.process.terminate()
            try:
                await asyncio.wait_for(job.process.wait(), timeout=2)
            except TimeoutError:
                job.process.kill()
                await job.process.wait()

        if job.reader_task is not None:
            await job.reader_task
        return self.snapshot(job_id)

    async def close(self) -> None:
        await asyncio.gather(
            *(self.stop(job_id) for job_id in self._jobs),
            return_exceptions=True,
        )

    async def _capture_output(self, job: _Job) -> None:
        assert job.process.stdout is not None
        output, truncated = await read_limited_stream(
            job.process.stdout,
            self._max_output_bytes,
        )
        job.output.extend(output)
        job.output_truncated = truncated
