from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from agent.events import Event, Usage


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_relative_path(value: str, *, field_name: str) -> str:
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)

    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or normalized == "."
    ):
        raise ValueError(
            f"{field_name} must be a non-empty relative path inside benchmarks"
        )

    return normalized


class TaskCategory(StrEnum):
    """The kind of capability an evaluation task measures."""

    BUG_FIX = "bug_fix"
    MULTI_FILE_CHANGE = "multi_file_change"
    INVESTIGATION = "investigation"
    TOOL_RECOVERY = "tool_recovery"
    SAFETY_REFUSAL = "safety_refusal"
    VERIFICATION = "verification"


class MilestoneKind(StrEnum):
    """
    Observable evidence required during a run.

    We intentionally inspect public runtime evidence such as tool calls,
    never private model reasoning.
    """

    TOOL_CALLED = "tool_called"


class MilestoneSpec(BaseModel):
    """
    One observable event that a task requires.

    Example: before editing a bug fix, the agent should inspect the workspace.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    milestone_id: str = Field(
        min_length=3,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    description: str = Field(min_length=1)
    kind: MilestoneKind
    tool_names: tuple[str, ...] = Field(min_length=1)

    @field_validator("description")
    @classmethod
    def reject_blank_description(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("milestone description must not be blank")

        return normalized

    @field_validator("tool_names")
    @classmethod
    def validate_tool_names(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized_names: list[str] = []

        for value in values:
            normalized = value.strip()

            if not normalized:
                raise ValueError("milestone tool names must not be blank")

            if not normalized.replace("_", "").isalnum():
                raise ValueError(
                    "milestone tool names may contain only letters, numbers, "
                    "and underscores"
                )

            normalized_names.append(normalized)

        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError("milestone tool names must be unique")

        return tuple(normalized_names)


class CommandSpec(BaseModel):
    """
    One verification command.

    We use argv instead of a shell string. That means the harness can execute
    the command without shell interpolation changing its meaning.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    argv: tuple[str, ...] = Field(min_length=1)
    timeout_seconds: int = Field(default=300, ge=1, le=3_600)

    @field_validator("argv")
    @classmethod
    def validate_argv(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        for part in value:
            if not part.strip():
                raise ValueError("command arguments must not be blank")

            if "\0" in part:
                raise ValueError("command arguments must not contain NUL bytes")

        return value


class BenchmarkTask(BaseModel):
    """
    One reproducible coding task.

    The task definition will later live in benchmarks/tasks/<task-id>.json.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(
        min_length=3,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    title: str = Field(min_length=1)
    prompt: str = Field(min_length=1)

    # Lets reports answer: “Which capability is weak?”
    category: TaskCategory

    # A repository fixture copied into a fresh workspace for every attempt.
    fixture: str

    # Commands that decide whether the task succeeded.
    verification: tuple[CommandSpec, ...] = Field(min_length=1)

    timeout_seconds: int = Field(default=900, ge=1, le=7_200)

    # Empty means the agent may modify any file inside the fixture workspace.
    allowed_changed_paths: tuple[str, ...] = ()

    # Public evidence required during the trajectory.
    milestones: tuple[MilestoneSpec, ...] = ()

    forbidden_tool_names: tuple[str, ...] = ()

    @field_validator("title", "prompt")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")

        return value

    @field_validator("fixture")
    @classmethod
    def validate_fixture(cls, value: str) -> str:
        return _require_relative_path(value, field_name="fixture")


    @field_validator("allowed_changed_paths")
    @classmethod
    def validate_allowed_changed_paths(cls,values: tuple[str, ...]) -> tuple[str, ...]:
        normalized_paths = tuple(
            _require_relative_path(
                value,
                field_name="allowed_changed_paths item",
            )
            for value in values
        )

        if len(normalized_paths) != len(set(normalized_paths)):
            raise ValueError("allowed_changed_paths must be unique")

        return normalized_paths

    @field_validator("milestones")
    @classmethod
    def validate_milestones(cls,values: tuple[MilestoneSpec, ...]) -> tuple[MilestoneSpec, ...]:
        milestone_ids = [
            milestone.milestone_id
            for milestone in values
        ]

        if len(milestone_ids) != len(set(milestone_ids)):
            raise ValueError("milestone_id values must be unique")

        return values

    @field_validator("forbidden_tool_names")
    @classmethod
    def validate_forbidden_tool_names(cls,values: tuple[str, ...]) -> tuple[str, ...]:
        normalized_names: list[str] = []

        for value in values:
            normalized = value.strip()

            if not normalized:
                raise ValueError("forbidden tool names must not be blank")

            if not normalized.replace("_", "").isalnum():
                raise ValueError(
                    "forbidden tool names may contain only letters, numbers, "
                    "and underscores"
                )

            normalized_names.append(normalized)

        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError("forbidden_tool_names must be unique")

        return tuple(normalized_names)

    @model_validator(mode="after")
    def reject_impossible_milestones(self) -> BenchmarkTask:
        forbidden_tools = set(self.forbidden_tool_names)

        for milestone in self.milestones:
            if forbidden_tools.intersection(milestone.tool_names):
                raise ValueError(
                    "a milestone cannot require a forbidden tool"
                )

        return self


class BenchmarkRunConfig(BaseModel):
    """
    Configuration shared by every repeated attempt of one benchmark task.

    Three attempts is the minimum because one successful run could be luck.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(pattern=r"^(anthropic|openai)$")
    model: str = Field(min_length=1)

    route_id: str = Field(
        default="unrouted",
        min_length=1,
        pattern=r"^[a-z][a-z0-9-]*$",
    )

    # Describes the routing policy being measured: fixed, shadow, or live.
    # It is separate from route_id because live routing can use multiple routes.
    strategy_id: str = Field(
        default="fixed",
        min_length=1,
        pattern=r"^[a-z][a-z0-9-]*$",
    )

    # Git commit of this agent implementation being measured.
    agent_revision: str = Field(min_length=7, max_length=64)

    # Example:
    # ghcr.io/example/agent-bench-python@sha256:abc123...
    container_image: str = Field(min_length=1)

    attempts: int = Field(default=3, ge=3, le=100)
    parallelism: int = Field(default=1, ge=1, le=32)

    # Exact configuration affecting behavior: effort, max_tokens,
    # permission mode, sandbox settings, and so on.
    agent_settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("model must not be blank")

        return normalized

    @field_validator("agent_revision")
    @classmethod
    def validate_agent_revision(cls, value: str) -> str:
        normalized = value.strip().lower()

        if not all(character in "0123456789abcdef" for character in normalized):
            raise ValueError("agent_revision must be a Git commit SHA")

        return normalized

    @field_validator("container_image")
    @classmethod
    def require_pinned_container_image(cls, value: str) -> str:
        normalized = value.strip()
        marker = "@sha256:"

        if marker not in normalized:
            raise ValueError(
                "container_image must be pinned with @sha256:<digest>"
            )

        digest = normalized.split(marker, maxsplit=1)[1]

        if (
            len(digest) != 64
            or not all(character in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "container_image must contain a 64-character sha256 digest"
            )

        return normalized

    @model_validator(mode="after")
    def validate_parallelism(self) -> BenchmarkRunConfig:
        if self.parallelism > self.attempts:
            raise ValueError("parallelism cannot exceed attempts")

        return self

    @property
    def fingerprint(self) -> str:
        """A stable identifier for every behavior-affecting run setting."""
        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class RunStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMED_OUT = "timed_out"


class ExecutedTurn(BaseModel):
    """
    The real model decision and usage for one completed LLM turn.

    A live router may choose a different model from the benchmark's starting
    model, so benchmark cost must come from this record—not an assumption.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(pattern=r"^(anthropic|openai)$")
    model: str = Field(min_length=1)
    route_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9-]*$",
    )
    usage: Usage
    cost_usd: Decimal | None = Field(
        default=None,
        ge=Decimal("0"),
    )

    @field_validator(
        "provider",
        "model",
        "route_id",
        mode="before",
    )
    @classmethod
    def normalize_identifiers(cls, value: object) -> object:
        if value is None or not isinstance(value, str):
            return value

        normalized = value.strip()

        if not normalized:
            raise ValueError("execution identifiers must not be blank")

        return normalized


class ModelMixEntry(BaseModel):
    """
    Aggregate usage for one actual provider/model/route combination across all
    attempts that formed a scoreboard row.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(pattern=r"^(anthropic|openai)$")
    model: str = Field(min_length=1)
    route_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9-]*$",
    )

    turns: int = Field(ge=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    cache_creation_tokens: int = Field(ge=0)

    # None means one or more turns used a model with unknown pricing.
    cost_usd: Decimal | None = Field(
        default=None,
        ge=Decimal("0"),
    )


class RunMetrics(BaseModel):
    """Measured values from one agent attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    duration_seconds: Decimal = Field(ge=Decimal("0"))
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_creation_tokens: int = Field(default=0, ge=0)
    turns: int = Field(default=0, ge=0)
    executed_turns: tuple[ExecutedTurn, ...] = ()

    # None means we honestly do not know the model's token price.
    cost_usd: Decimal | None = Field(default=None, ge=Decimal("0"))

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_creation_tokens
        )

    @model_validator(mode="after")
    def validate_executed_turn_totals(self) -> RunMetrics:
        """
        When detailed records exist, ensure the summary cannot disagree with
        the underlying per-turn evidence.
        """
        if not self.executed_turns:
            return self

        if self.turns != len(self.executed_turns):
            raise ValueError(
                "turns must equal the number of executed_turns"
            )

        if self.input_tokens != sum(
            turn.usage.input_tokens for turn in self.executed_turns
        ):
            raise ValueError(
                "input_tokens must equal executed_turn input tokens"
            )

        if self.output_tokens != sum(
            turn.usage.output_tokens for turn in self.executed_turns
        ):
            raise ValueError(
                "output_tokens must equal executed_turn output tokens"
            )

        if self.cache_read_tokens != sum(
            turn.usage.cache_read_input_tokens
            for turn in self.executed_turns
        ):
            raise ValueError(
                "cache_read_tokens must equal executed_turn cache reads"
            )

        if self.cache_creation_tokens != sum(
            turn.usage.cache_creation_input_tokens
            for turn in self.executed_turns
        ):
            raise ValueError(
                "cache_creation_tokens must equal executed_turn cache writes"
            )

        costs = [turn.cost_usd for turn in self.executed_turns]
        expected_cost = (
            None
            if any(cost is None for cost in costs)
            else sum(
                (cost for cost in costs if cost is not None),
                start=Decimal("0"),
            )
        )

        if self.cost_usd != expected_cost:
            raise ValueError(
                "cost_usd must equal the total executed_turn cost"
            )

        return self


class TrajectoryEntry(BaseModel):
    """
    One normalized agent event saved during a benchmark attempt.

    Keeping the original Event lets us later answer questions such as:
    “Why did run 2 fail while run 1 passed?”
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int = Field(ge=1)
    event_type: str = Field(min_length=1)
    event: Event

    @model_validator(mode="after")
    def validate_event_identity(self) -> TrajectoryEntry:
        if self.event.seq != self.sequence:
            raise ValueError(
                "trajectory sequence must match the embedded event sequence"
            )

        if self.event.type != self.event_type:
            raise ValueError(
                "trajectory event_type must match the embedded event type"
            )

        return self


class TrajectoryReference(BaseModel):
    """Metadata for a JSONL trajectory stored on disk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    relative_path: str
    event_count: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _require_relative_path(value, field_name="relative_path")

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.strip().lower()

        if not all(character in "0123456789abcdef" for character in normalized):
            raise ValueError("sha256 must be a hexadecimal digest")

        return normalized


class VerificationResult(BaseModel):
    """Outcome of one task verification command."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: CommandSpec
    exit_code: int
    passed: bool

    # The evaluator will truncate these before storing them.
    stdout_tail: str = ""
    stderr_tail: str = ""


class EvaluationViolationKind(StrEnum):
    """A deterministic reason that an evaluation contract failed."""

    FILE_BOUNDARY = "file_boundary"
    FORBIDDEN_TOOL = "forbidden_tool"
    MISSING_MILESTONE = "missing_milestone"
    VERIFICATION = "verification"


class MilestoneResult(BaseModel):
    """
    Evidence for one required task milestone.

    This records public runtime evidence only. It never stores private model
    reasoning or chain-of-thought.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    milestone_id: str = Field(
        min_length=3,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    kind: MilestoneKind
    passed: bool

    # For the current tool_called milestone kind, these identify the event
    # that proved the milestone occurred.
    observed_tool_name: str | None = None
    observed_sequence: int | None = Field(default=None, ge=1)

    @field_validator("observed_tool_name")
    @classmethod
    def normalize_observed_tool_name(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip()

        if not normalized:
            raise ValueError("observed_tool_name must not be blank")

        return normalized

    @model_validator(mode="after")
    def validate_evidence(self) -> MilestoneResult:
        if self.passed:
            if self.observed_tool_name is None:
                raise ValueError(
                    "a passed milestone requires observed_tool_name"
                )

            if self.observed_sequence is None:
                raise ValueError(
                    "a passed milestone requires observed_sequence"
                )
        elif (
            self.observed_tool_name is not None
            or self.observed_sequence is not None
        ):
            raise ValueError(
                "a failed milestone cannot contain passing evidence"
            )

        return self


class EvaluationViolation(BaseModel):
    """One policy or contract violation found by deterministic evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EvaluationViolationKind
    message: str = Field(min_length=1)

    # A violation caused by a tool event points to that immutable trajectory
    # sequence. File-boundary and verification failures have no event sequence.
    event_sequence: int | None = Field(default=None, ge=1)

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("evaluation violation message must not be blank")

        return normalized


class EvaluationResult(BaseModel):
    """
    Complete deterministic verdict for one completed benchmark attempt.

    The result is an evidence receipt: it retains changed files, milestone
    evidence, policy violations, and verification results.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    changed_paths: tuple[str, ...] = ()
    milestones: tuple[MilestoneResult, ...] = ()
    violations: tuple[EvaluationViolation, ...] = ()
    verification: tuple[VerificationResult, ...] = ()

    @field_validator("changed_paths")
    @classmethod
    def validate_changed_paths(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized_paths = tuple(
            _require_relative_path(
                value,
                field_name="changed_paths item",
            )
            for value in values
        )

        if len(normalized_paths) != len(set(normalized_paths)):
            raise ValueError("changed_paths must be unique")

        return normalized_paths

    @model_validator(mode="after")
    def validate_verdict(self) -> EvaluationResult:
        milestone_ids = [
            milestone.milestone_id
            for milestone in self.milestones
        ]

        if len(milestone_ids) != len(set(milestone_ids)):
            raise ValueError("evaluation milestone IDs must be unique")

        if not self.passed:
            return self

        if self.violations:
            raise ValueError(
                "a passed evaluation cannot contain violations"
            )

        if not self.verification:
            raise ValueError(
                "a passed evaluation requires verification results"
            )

        if not all(result.passed for result in self.verification):
            raise ValueError(
                "a passed evaluation cannot contain failed verification"
            )

        if not all(milestone.passed for milestone in self.milestones):
            raise ValueError(
                "a passed evaluation cannot contain failed milestones"
            )

        return self


class BenchmarkRunResult(BaseModel):
    """One isolated attempt of one task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=3)
    attempt: int = Field(ge=1)

    status: RunStatus
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)

    metrics: RunMetrics
    trajectory: TrajectoryReference
    verification: tuple[VerificationResult, ...] = ()

    error_message: str | None = None

    # Full deterministic evidence receipt. It remains None only when the agent
    # crashed or timed out before evaluation could happen.

    evaluation: EvaluationResult | None = None

    @model_validator(mode="after")
    def validate_success(self) -> BenchmarkRunResult:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot be earlier than started_at")

        if self.status is RunStatus.PASSED:
            if not self.verification:
                raise ValueError(
                    "a passed run must include verification results"
                )

            if not all(result.passed for result in self.verification):
                raise ValueError(
                    "a passed run cannot contain a failed verification"
                )

            if self.evaluation is not None:
                if self.evaluation.verification != self.verification:
                    raise ValueError(
                        "evaluation verification must match run verification"
                    )

                if self.status is RunStatus.PASSED and not self.evaluation.passed:
                    raise ValueError(
                        "a passed run requires a passed evaluation"
                    )

        return self

    


class ScoreboardRow(BaseModel):
    """
    Reproducible summary of all attempts for one task/configuration pair.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=3)
    provider: str = Field(pattern=r"^(anthropic|openai)$")
    model: str = Field(min_length=1)

    route_id: str = Field(
        default="unrouted",
        min_length=1,
        pattern=r"^[a-z][a-z0-9-]*$",
    )

    strategy_id: str = Field(
        default="fixed",
        min_length=1,
        pattern=r"^[a-z][a-z0-9-]*$",
    )

    # Totals across every benchmark attempt, grouped by actual execution.
    model_mix: tuple[ModelMixEntry, ...] = ()

    agent_revision: str = Field(min_length=7)
    container_image: str = Field(min_length=1)
    config_fingerprint: str = Field(min_length=64, max_length=64)

    attempts: int = Field(ge=1)
    passed_attempts: int = Field(ge=0)
    pass_rate: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))

    mean_duration_seconds: Decimal = Field(ge=Decimal("0"))
    duration_standard_deviation_seconds: Decimal = Field(
        ge=Decimal("0")
    )
    mean_tokens: Decimal = Field(ge=Decimal("0"))
    token_standard_deviation: Decimal = Field(ge=Decimal("0"))
    mean_turns: Decimal = Field(ge=Decimal("0"))

    # None means at least one relevant cost was unknown.
    mean_cost_usd: Decimal | None = Field(
        default=None,
        ge=Decimal("0"),
    )
    cost_standard_deviation_usd: Decimal | None = Field(
        default=None,
        ge=Decimal("0"),
    )

    @model_validator(mode="after")
    def validate_pass_rate(self) -> ScoreboardRow:
        if self.passed_attempts > self.attempts:
            raise ValueError(
                "passed_attempts cannot exceed attempts"
            )

        expected = Decimal(self.passed_attempts) / Decimal(self.attempts)

        if self.pass_rate != expected:
            raise ValueError(
                "pass_rate must equal passed_attempts / attempts"
            )

        if not all(
            character in "0123456789abcdef"
            for character in self.config_fingerprint.lower()
        ):
            raise ValueError("config_fingerprint must be a SHA-256 digest")

        return self
