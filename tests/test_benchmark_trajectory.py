from pathlib import Path

import pytest

from agent.benchmark.trajectory import TrajectoryError, TrajectoryWriter, read_trajectory
from agent.events import UserMessage
from agent.providers.base import EventFactory


def test_trajectory_round_trips_and_detects_tampering(tmp_path: Path):
    writer = TrajectoryWriter(tmp_path, "run-001")
    event = EventFactory("session")(UserMessage, text="fix it")
    writer.append(event)
    reference = writer.close()

    entries = read_trajectory(tmp_path, reference)
    assert entries[0].event == event

    writer.path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(TrajectoryError, match="SHA-256"):
        read_trajectory(tmp_path, reference)
