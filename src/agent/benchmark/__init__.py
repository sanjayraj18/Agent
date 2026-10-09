"""Reproducible behavioral evaluation for the terminal coding agent."""

from agent.benchmark.catalog import BenchmarkCatalog, BenchmarkCatalogError
from agent.benchmark.comparison import (
    BaselineRelation,
    BenchmarkComparison,
    CandidateComparison,
    ComparisonError,
    compare_scoreboard_rows,
)
from agent.benchmark.models import (
    BenchmarkRunConfig,
    BenchmarkRunResult,
    BenchmarkTask,
    CommandSpec,
    MilestoneKind,
    MilestoneSpec,
    ScoreboardRow,
    TaskCategory,
)
from agent.benchmark.runner import BenchmarkRunner

__all__ = [
    "BaselineRelation",
    "BenchmarkCatalog",
    "BenchmarkCatalogError",
    "BenchmarkComparison",
    "BenchmarkRunConfig",
    "BenchmarkRunResult",
    "BenchmarkRunner",
    "BenchmarkTask",
    "CandidateComparison",
    "CommandSpec",
    "ComparisonError",
    "MilestoneKind",
    "MilestoneSpec",
    "ScoreboardRow",
    "TaskCategory",
    "compare_scoreboard_rows",
]