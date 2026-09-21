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

from agent.events import Event


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

    # A repository fixture copied into a fresh workspace for every attempt.
    fixture: str

    # Commands that decide whether the task succeeded.
    verification: tuple[CommandSpec, ...] = Field(min_length=1)

    timeout_seconds: int = Field(default=900, ge=1, le=7_200)

    # Empty means the agent may modify any file inside the fixture workspace.
    allowed_changed_paths: tuple[str, ...] = ()

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
    def validate_allowed_changed_paths(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            _require_relative_path(
                value,
                field_name="allowed_changed_paths item",
            )
            for value in values
        )


class BenchmarkRunConfig(BaseModel):
    """
    Configuration shared by every repeated attempt of one benchmark task.

    Three attempts is the minimum because one successful run could be luck.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(pattern=r"^(anthropic|openai)$")
    model: str = Field(min_length=1)

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


class RunMetrics(BaseModel):
    """Measured values from one agent attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    duration_seconds: Decimal = Field(ge=Decimal("0"))
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_creation_tokens: int = Field(default=0, ge=0)
    turns: int = Field(default=0, ge=0)

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

        return self


class ScoreboardRow(BaseModel):
    """
    Reproducible summary of all attempts for one task/configuration pair.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=3)
    provider: str = Field(pattern=r"^(anthropic|openai)$")
    model: str = Field(min_length=1)
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
