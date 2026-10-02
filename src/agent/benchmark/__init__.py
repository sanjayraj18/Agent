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
    ScoreboardRow,
)
from agent.benchmark.runner import BenchmarkRunner

__all__ = [
    "BenchmarkCatalog",
    "BenchmarkCatalogError",
    "BaselineRelation",
    "BenchmarkComparison",
    "CandidateComparison",
    "ComparisonError",
    "BenchmarkRunConfig",
    "BenchmarkRunResult",
    "BenchmarkRunner",
    "BenchmarkTask",
    "ScoreboardRow",
    "compare_scoreboard_rows",
]
