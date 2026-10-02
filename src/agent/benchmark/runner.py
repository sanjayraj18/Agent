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
    ExecutedTurn,
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
            duration_seconds=Decimal(
                str(time.perf_counter() - started_monotonic)
            ),
            provider=config.provider,
            configured_model=config.model,
            configured_route_id=config.route_id,
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
    provider: str,
    configured_model: str,
    configured_route_id: str,
) -> RunMetrics:
    """
    Build auditable metrics from the actual model used on every LLM turn.

    Older providers and unit-test fakes may omit execution metadata. In that
    case, use the benchmark configuration as the truthful fixed-model fallback.
    """
    executed_turns = tuple(
        _executed_turn(
            event,
            fallback_provider=provider,
            fallback_model=configured_model,
            fallback_route_id=configured_route_id,
        )
        for event in assistant_ends
    )

    usage = Usage(
        input_tokens=sum(
            turn.usage.input_tokens
            for turn in executed_turns
        ),
        output_tokens=sum(
            turn.usage.output_tokens
            for turn in executed_turns
        ),
        cache_read_input_tokens=sum(
            turn.usage.cache_read_input_tokens
            for turn in executed_turns
        ),
        cache_creation_input_tokens=sum(
            turn.usage.cache_creation_input_tokens
            for turn in executed_turns
        ),
    )

    costs = [turn.cost_usd for turn in executed_turns]
    total_cost = (
        None
        if any(cost is None for cost in costs)
        else sum(
            (cost for cost in costs if cost is not None),
            start=Decimal("0"),
        )
    )

    return RunMetrics(
        duration_seconds=duration_seconds,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens,
        cache_creation_tokens=usage.cache_creation_input_tokens,
        turns=len(executed_turns),
        executed_turns=executed_turns,
        cost_usd=total_cost,
    )


def _executed_turn(
    event: AssistantEnd,
    *,
    fallback_provider: str,
    fallback_model: str,
    fallback_route_id: str,
) -> ExecutedTurn:
    """Convert one assistant-end event into an auditable cost receipt."""
    actual_provider = event.executed_provider or fallback_provider
    actual_model = event.executed_model or fallback_model
    actual_route_id = event.executed_route_id or fallback_route_id

    calculated_cost = calculate_known_model_cost(
        actual_model,
        event.usage,
    )

    return ExecutedTurn(
        provider=actual_provider,
        model=actual_model,
        route_id=actual_route_id,
        usage=event.usage,
        cost_usd=(
            calculated_cost.total_usd
            if calculated_cost is not None
            else None
        ),
    )