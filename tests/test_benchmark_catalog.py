import json
from pathlib import Path
import subprocess
import sys

import pytest

from agent.benchmark.catalog import BenchmarkCatalog, BenchmarkCatalogError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
INITIAL_TASK_IDS = {
    "fix-add-bug",
    "fix-clamp-boundary",
    "investigate-cache-setting",
    "rename-formatter-api",
    "repair-label-parser",
    "respect-protected-test",
    "update-auth-header",
    "verify-tax-rounding",
}


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


def test_repository_catalog_loads_the_complete_initial_task_suite():
    catalog = BenchmarkCatalog(BENCHMARK_ROOT)
    tasks = catalog.list_tasks()

    assert {task.task_id for task in tasks} == INITIAL_TASK_IDS
    assert all(task.allowed_changed_paths for task in tasks)
    assert all(task.milestones for task in tasks)
    assert all(task.verification for task in tasks)


@pytest.mark.parametrize("task_id", sorted(INITIAL_TASK_IDS))
def test_each_initial_fixture_starts_unsolved(task_id: str):
    task = BenchmarkCatalog(BENCHMARK_ROOT).load_task(task_id)
    fixture = BENCHMARK_ROOT / task.fixture

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=fixture,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, result.stdout + result.stderr
