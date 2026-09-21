"""Repeated, isolated execution of one benchmark task."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import time
from uuid import uuid4

from agent.benchmark.evaluator import EvaluationOutcome, evaluate_workspace
from agent.benchmark.images import CommandOutput
from agent.benchmark.isolation import create_isolated_workspace
from agent.benchmark.models import (
    BenchmarkRunConfig,
    BenchmarkRunResult,
    BenchmarkTask,
    CommandSpec,
    RunMetrics,
    RunStatus,
    utc_now,
)
from agent.benchmark.trajectory import TrajectoryWriter
from agent.core.costs import calculate_known_model_cost
from agent.events import AssistantEnd, ErrorEvent, Event, Usage


AgentAttempt = Callable[[Path, str], AsyncIterator[Event]]
CommandRunner = Callable[[CommandSpec, Path], Awaitable[CommandOutput]]


class BenchmarkRunner:
    """Run the same task repeatedly without letting attempts share state."""

    def __init__(
        self,
        *,
        fixture_root: Path,
        workspace_root: Path,
        results_root: Path,
        agent_attempt: AgentAttempt,
        command_runner: CommandRunner,
    ) -> None:
        self._fixture_root = fixture_root.expanduser().resolve()
        self._workspace_root = workspace_root.expanduser().resolve()
        self._results_root = results_root.expanduser().resolve()
        self._agent_attempt = agent_attempt
        self._command_runner = command_runner

    async def run_task(
        self,
        task: BenchmarkTask,
        config: BenchmarkRunConfig,
        *,
        keep_workspaces: bool = False,
    ) -> tuple[BenchmarkRunResult, ...]:
        fixture = (self._fixture_root / task.fixture).resolve()
        try:
            fixture.relative_to(self._fixture_root)
        except ValueError as exc:
            raise ValueError("task fixture escapes fixture_root") from exc

        limiter = asyncio.Semaphore(config.parallelism)

        async def one_attempt(attempt: int) -> BenchmarkRunResult:
            async with limiter:
                return await self._run_attempt(
                    task,
                    config,
                    fixture,
                    attempt,
                    keep_workspaces=keep_workspaces,
                )

        results = await asyncio.gather(
            *(one_attempt(attempt) for attempt in range(1, config.attempts + 1))
        )
        return tuple(sorted(results, key=lambda result: result.attempt))

    async def _run_attempt(
        self,
        task: BenchmarkTask,
        config: BenchmarkRunConfig,
        fixture: Path,
        attempt: int,
        *,
        keep_workspaces: bool,
    ) -> BenchmarkRunResult:
        run_id = f"{task.task_id}-{attempt:03d}-{uuid4().hex[:12]}"
        started_at = utc_now()
        started_monotonic = time.perf_counter()
        trajectory = TrajectoryWriter(self._results_root, run_id)
        isolated_workspace = None
        assistant_ends: list[AssistantEnd] = []
        error_message: str | None = None
        status = RunStatus.ERROR
        evaluation: EvaluationOutcome | None = None

        try:
            isolated_workspace = await asyncio.to_thread(
                create_isolated_workspace,
                fixture,
                run_root=self._workspace_root,
                run_id=run_id,
            )

            try:
                async with asyncio.timeout(task.timeout_seconds):
                    async for event in self._agent_attempt(
                        isolated_workspace.workspace,
                        task.prompt,
                    ):
                        trajectory.append(event)
                        if isinstance(event, AssistantEnd):
                            assistant_ends.append(event)
                        elif isinstance(event, ErrorEvent):
                            error_message = f"{event.kind}: {event.message}"
            except TimeoutError:
                status = RunStatus.TIMED_OUT
                error_message = (
                    f"agent attempt exceeded {task.timeout_seconds} seconds"
                )
            else:
                if error_message is None:
                    evaluation = await evaluate_workspace(
                        task,
                        isolated_workspace,
                        self._command_runner,
                    )
                    if evaluation.passed:
                        status = RunStatus.PASSED
                    else:
                        status = RunStatus.FAILED
                        error_message = evaluation.violation_message
                else:
                    status = RunStatus.ERROR
        except Exception as exc:
            status = RunStatus.ERROR
            error_message = f"{type(exc).__name__}: {exc}"
        finally:
            reference = trajectory.close()
            if isolated_workspace is not None and not keep_workspaces:
                await asyncio.to_thread(isolated_workspace.cleanup)

        finished_at = utc_now()
        metrics = _metrics(
            assistant_ends,
            duration_seconds=Decimal(str(time.perf_counter() - started_monotonic)),
            model=config.model,
        )

        return BenchmarkRunResult(
            run_id=run_id,
            task_id=task.task_id,
            attempt=attempt,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            metrics=metrics,
            trajectory=reference,
            verification=(
                evaluation.verification
                if evaluation is not None
                else ()
            ),
            error_message=error_message,
        )


def _metrics(
    assistant_ends: list[AssistantEnd],
    *,
    duration_seconds: Decimal,
    model: str,
) -> RunMetrics:
    usage = Usage(
        input_tokens=sum(event.usage.input_tokens for event in assistant_ends),
        output_tokens=sum(event.usage.output_tokens for event in assistant_ends),
        cache_read_input_tokens=sum(
            event.usage.cache_read_input_tokens for event in assistant_ends
        ),
        cache_creation_input_tokens=sum(
            event.usage.cache_creation_input_tokens for event in assistant_ends
        ),
    )
    costs = [calculate_known_model_cost(model, event.usage) for event in assistant_ends]
    total_cost = (
        None
        if any(cost is None for cost in costs)
        else sum(
            (cost.total_usd for cost in costs if cost is not None),
            start=Decimal("0"),
        )
    )

    return RunMetrics(
        duration_seconds=duration_seconds,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens,
        cache_creation_tokens=usage.cache_creation_input_tokens,
        turns=len(assistant_ends),
        cost_usd=total_cost,
    )
