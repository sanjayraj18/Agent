from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from agent.core.costs import (
    MODEL_PRICING,
    ModelPricing,
    TurnCost,
    calculate_known_model_cost,
)
from agent.events import Usage


ZERO_DOLLARS = Decimal("0")
ZERO_SECONDS = Decimal("0")


@dataclass(frozen=True, slots=True)
class TurnTiming:

    """
    Latency measured around one completed provider stream.

    ``provider_duration_seconds`` starts immediately before the provider call
    and ends when ``assistant.end`` arrives.

    ``time_to_first_output_seconds`` is optional because a provider might end
    without streaming text, thinking, or a tool call.
    """

    provider_duration_seconds: Decimal
    time_to_first_output_seconds: Decimal | None = None

    def __post_init__(self) -> None:
        if self.provider_duration_seconds < ZERO_SECONDS:
            raise ValueError(
                "provider_duration_seconds must not be negative"
            )

        if (
            self.time_to_first_output_seconds is not None
            and self.time_to_first_output_seconds < ZERO_SECONDS
        ):
            raise ValueError(
                "time_to_first_output_seconds must not be negative"
            )

        if (
            self.time_to_first_output_seconds is not None
            and self.time_to_first_output_seconds
            > self.provider_duration_seconds
        ):
            raise ValueError(
                "time_to_first_output_seconds cannot exceed "
                "provider_duration_seconds"
            )


@dataclass(frozen=True, slots=True)
class ShadowRouteRecommendation:
    """
    A route the router recommended but did not execute.

    Phase 5 records this beside the actual model call so we can compare
    proposed routing behavior against the unchanged strong-route behavior.
    """

    route_id: str
    provider: str
    model: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.route_id.strip():
            raise ValueError("route_id must not be blank")

        if not self.provider.strip():
            raise ValueError("provider must not be blank")

        if not self.model.strip():
            raise ValueError("model must not be blank")

        if not self.reasons:
            raise ValueError("reasons must not be empty")

        if any(not reason.strip() for reason in self.reasons):
            raise ValueError("reasons must not contain blank text")


@dataclass(frozen=True, slots=True)
class TurnTelemetry:
    """Usage and cost information from one completed LLM turn."""

    turn_number: int
    model: str
    usage: Usage
    cost: TurnCost | None
    stable_prefix_fingerprint: str | None = None
    shadow_route: ShadowRouteRecommendation | None = None
    timing: TurnTiming | None = None

    @property
    def total_tokens(self) -> int:
        return (
            self.usage.input_tokens
            + self.usage.output_tokens
            + self.usage.cache_read_input_tokens
            + self.usage.cache_creation_input_tokens
        )

    @property
    def prompt_tokens(self) -> int:
        return (
            self.usage.input_tokens
            + self.usage.cache_read_input_tokens
            + self.usage.cache_creation_input_tokens
        )


class SessionTelemetry:
    """
    Collects token and cost totals for one agent session.

    A turn can have no cost when the selected model is not in our local
    pricing catalog. We track that honestly instead of pretending it costs $0.
    """

    def __init__(
        self,
        pricing_catalog: Mapping[str, ModelPricing] = MODEL_PRICING,
    ) -> None:
        self._pricing_catalog = pricing_catalog
        self._turns: list[TurnTelemetry] = []

    @property
    def turns(self) -> tuple[TurnTelemetry, ...]:
        return tuple(self._turns)

    def record_turn(
        self,
        model: str,
        usage: Usage,
        stable_prefix_fingerprint: str | None = None,
        shadow_route: ShadowRouteRecommendation | None = None,
        timing: TurnTiming | None = None,
    ) -> TurnTelemetry:
        if not model:
            raise ValueError("model must not be empty")

        cost = calculate_known_model_cost(
            model,
            usage,
            pricing_catalog=self._pricing_catalog,
        )

        turn = TurnTelemetry(
            turn_number=len(self._turns) + 1,
            model=model,
            usage=usage,
            cost=cost,
            stable_prefix_fingerprint=stable_prefix_fingerprint,
            shadow_route=shadow_route,
            timing=timing
        )
        self._turns.append(turn)
        return turn

    @property
    def input_tokens(self) -> int:
        return sum(turn.usage.input_tokens for turn in self._turns)

    @property
    def output_tokens(self) -> int:
        return sum(turn.usage.output_tokens for turn in self._turns)

    @property
    def cache_read_tokens(self) -> int:
        return sum(
            turn.usage.cache_read_input_tokens
            for turn in self._turns
        )

    @property
    def cache_creation_tokens(self) -> int:
        return sum(
            turn.usage.cache_creation_input_tokens
            for turn in self._turns
        )

    @property
    def total_tokens(self) -> int:
        return sum(turn.total_tokens for turn in self._turns)

    @property
    def known_total_cost_usd(self) -> Decimal:
        return sum(
            (
                turn.cost.total_usd
                for turn in self._turns
                if turn.cost is not None
            ),
            start=ZERO_DOLLARS,
        )

    @property
    def unpriced_turn_count(self) -> int:
        return sum(
            1
            for turn in self._turns
            if turn.cost is None
        )

    @property
    def total_cost_usd(self) -> Decimal | None:
        if self.unpriced_turn_count:
            return None

        return self.known_total_cost_usd

    @property
    def timed_turn_count(self) -> int:
        """Number of completed turns for which latency was measured."""

        return sum(
            1
            for turn in self._turns
            if turn.timing is not None
        )

    @property
    def unmeasured_turn_count(self) -> int:
        """Number of completed turns missing latency data."""

        return len(self._turns) - self.timed_turn_count


    @property
    def known_total_provider_duration_seconds(self) -> Decimal:
        """Sum timing only for turns that were actually measured."""

        return sum(
            (
                turn.timing.provider_duration_seconds
                for turn in self._turns
                if turn.timing is not None
            ),
            start=ZERO_SECONDS,
        )

    @property
    def total_provider_duration_seconds(self) -> Decimal | None:
        """
        Total LLM-provider time, or ``None`` if any completed turn is missing
        a timing measurement.
        """

        if self.unmeasured_turn_count:
            return None

        return self.known_total_provider_duration_seconds

    @property
    def cache_hit_rate(self) -> Decimal:
        prompt_tokens = (
            self.input_tokens
            + self.cache_read_tokens
            + self.cache_creation_tokens
        )

        if prompt_tokens == 0:
            return Decimal("0")

        return (
            Decimal(self.cache_read_tokens)
            / Decimal(prompt_tokens)
        )


    @property
    def shadow_routed_turn_count(self) -> int:
        """Number of completed turns with a shadow recommendation."""

        return sum(
            1
            for turn in self._turns
            if turn.shadow_route is not None
        )

    @property
    def shadow_model_difference_count(self) -> int:
        """
        Number of turns where the recommended model differed from the model
        actually executed.
        """

        return sum(
            1
            for turn in self._turns
            if (
                turn.shadow_route is not None
                and turn.shadow_route.model != turn.model
            )
        )

    @property
    def cache_prefix_changed(self) -> bool:
        fingerprints = {
            turn.stable_prefix_fingerprint
            for turn in self._turns
            if turn.stable_prefix_fingerprint is not None
        }
        return len(fingerprints) > 1