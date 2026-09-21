from pathlib import Path

from agent.benchmark.isolation import create_isolated_workspace


def test_every_attempt_gets_a_private_copy_and_a_change_snapshot(tmp_path: Path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    original = fixture / "answer.txt"
    original.write_text("wrong", encoding="utf-8")

    isolated = create_isolated_workspace(
        fixture,
        run_root=tmp_path / "runs",
        run_id="fix-add-001",
    )
    copied = isolated.workspace / "answer.txt"
    copied.write_text("right", encoding="utf-8")

    assert original.read_text(encoding="utf-8") == "wrong"
    assert isolated.changed_paths() == ("answer.txt",)

    isolated.cleanup()
    assert not isolated.root.exists()
