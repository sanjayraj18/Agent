"""Append-only JSONL evidence for one benchmark attempt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent.benchmark.models import TrajectoryEntry, TrajectoryReference
from agent.events import Event


class TrajectoryError(RuntimeError):
    """A trajectory cannot be safely written or read."""


class TrajectoryWriter:
    """Persist every normalized event before the benchmark evaluates it."""

    def __init__(self, results_root: Path, run_id: str) -> None:
        self._results_root = results_root.expanduser().resolve()
        if not run_id or "/" in run_id or "\\" in run_id:
            raise ValueError("run_id must be a non-empty path-safe identifier")

        self._relative_path = Path(run_id) / "trajectory.jsonl"
        self._path = self._results_root / self._relative_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("wb")
        self._digest = hashlib.sha256()
        self._event_count = 0
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: Event) -> None:
        if self._closed:
            raise TrajectoryError("cannot append to a closed trajectory")

        entry = TrajectoryEntry(
            sequence=event.seq,
            event_type=event.type,
            event=event,
        )
        encoded = entry.model_dump_json().encode("utf-8") + b"\n"
        self._handle.write(encoded)
        self._handle.flush()
        self._digest.update(encoded)
        self._event_count += 1

    def close(self) -> TrajectoryReference:
        if not self._closed:
            self._handle.flush()
            self._handle.close()
            self._closed = True

        return TrajectoryReference(
            relative_path=self._relative_path.as_posix(),
            event_count=self._event_count,
            sha256=self._digest.hexdigest(),
        )


def read_trajectory(
    results_root: Path,
    reference: TrajectoryReference,
) -> tuple[TrajectoryEntry, ...]:
    """Read and integrity-check a previously written trajectory."""
    root = results_root.expanduser().resolve()
    path = (root / reference.relative_path).resolve()

    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TrajectoryError("trajectory path escapes results root") from exc

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TrajectoryError(f"could not read trajectory {path}: {exc}") from exc

    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise TrajectoryError("trajectory SHA-256 does not match its reference")

    entries: list[TrajectoryEntry] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            decoded = json.loads(line)
            entries.append(TrajectoryEntry.model_validate(decoded))
        except Exception as exc:
            raise TrajectoryError(
                f"invalid trajectory record at line {line_number}: {exc}"
            ) from exc

    if len(entries) != reference.event_count:
        raise TrajectoryError("trajectory event count does not match its reference")

    return tuple(entries)
