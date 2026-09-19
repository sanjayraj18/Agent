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


@dataclass(frozen=True, slots=True)
class TurnTelemetry:
    """Usage and cost information from one completed LLM turn."""

    turn_number: int
    model: str
    usage: Usage
    cost: TurnCost | None
    stable_prefix_fingerprint: str | None = None

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
    def cache_prefix_changed(self) -> bool:
        fingerprints = {
            turn.stable_prefix_fingerprint
            for turn in self._turns
            if turn.stable_prefix_fingerprint is not None
        }
        return len(fingerprints) > 1