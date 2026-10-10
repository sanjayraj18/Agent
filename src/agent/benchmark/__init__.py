"""Reproducible behavioral evaluation for the terminal coding agent."""

from agent.benchmark.catalog import BenchmarkCatalog, BenchmarkCatalogError
from agent.benchmark.comparison import (
    BaselineRelation,
    BenchmarkComparison,
    CandidateComparison,
    ComparisonError,
    compare_scoreboard_rows,
)
from agent.benchmark.diagnosis import (
    DiagnosisError,
    diagnose_run,
    summarize_diagnoses,
)
from agent.benchmark.models import (
    BenchmarkRunConfig,
    BenchmarkRunResult,
    BenchmarkTask,
    CommandSpec,
    DiagnosisCategory,
    DiagnosisEvidence,
    DiagnosisEvidenceSource,
    DiagnosisSummary,
    EvaluationResult,
    EvaluationViolation,
    EvaluationViolationKind,
    FailurePattern,
    MilestoneKind,
    MilestoneResult,
    MilestoneSpec,
    RunDiagnosis,
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
    "DiagnosisCategory",
    "DiagnosisError",
    "DiagnosisEvidence",
    "DiagnosisEvidenceSource",
    "DiagnosisSummary",
    "EvaluationResult",
    "EvaluationViolation",
    "EvaluationViolationKind",
    "FailurePattern",
    "MilestoneKind",
    "MilestoneResult",
    "MilestoneSpec",
    "RunDiagnosis",
    "ScoreboardRow",
    "TaskCategory",
    "compare_scoreboard_rows",
    "diagnose_run",
    "summarize_diagnoses",
]
