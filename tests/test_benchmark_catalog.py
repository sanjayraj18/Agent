import json
from pathlib import Path

import pytest

from agent.benchmark.catalog import BenchmarkCatalog, BenchmarkCatalogError


def _task() -> dict[str, object]:
    return {
        "task_id": "fix-add-bug",
        "title": "Fix add",
        "prompt": "Fix the bug.",
        "category": "bug_fix",
        "fixture": "fixtures/fix-add-bug",
        "verification": [{"argv": ["python", "-m", "pytest"]}],
    }


def test_catalog_loads_a_task_and_resolves_its_fixture(tmp_path: Path):
    (tmp_path / "tasks").mkdir()
    fixture = tmp_path / "fixtures" / "fix-add-bug"
    fixture.mkdir(parents=True)
    (tmp_path / "tasks" / "fix-add-bug.json").write_text(
        json.dumps(_task()),
        encoding="utf-8",
    )

    catalog = BenchmarkCatalog(tmp_path)
    task = catalog.load_task("fix-add-bug")

    assert task.task_id == "fix-add-bug"
    assert catalog.fixture_path(task) == fixture.resolve()


def test_catalog_rejects_a_filename_that_disagrees_with_task_id(tmp_path: Path):
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "different.json").write_text(
        json.dumps(_task()),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkCatalogError, match="does not match"):
        BenchmarkCatalog(tmp_path).load_task("different")


def test_catalog_rejects_an_invalid_task_id_before_creating_a_path(
    tmp_path: Path,
):
    with pytest.raises(BenchmarkCatalogError, match="lowercase letters"):
        BenchmarkCatalog(tmp_path).load_task("../outside")


def test_catalog_rejects_a_task_with_a_missing_fixture(tmp_path: Path):
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "fix-add-bug.json").write_text(
        json.dumps(_task()),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkCatalogError, match="fixture is not a directory"):
        BenchmarkCatalog(tmp_path).load_task("fix-add-bug")


def test_catalog_rejects_a_task_that_does_not_match_the_contract(
    tmp_path: Path,
):
    (tmp_path / "tasks").mkdir()
    invalid_task = _task()
    invalid_task["category"] = "unknown_category"
    (tmp_path / "tasks" / "fix-add-bug.json").write_text(
        json.dumps(invalid_task),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkCatalogError, match="invalid task definition"):
        BenchmarkCatalog(tmp_path).load_task("fix-add-bug")


def test_list_tasks_validates_every_task_fixture(tmp_path: Path):
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "fix-add-bug.json").write_text(
        json.dumps(_task()),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkCatalogError, match="fixture is not a directory"):
        BenchmarkCatalog(tmp_path).list_tasks()
