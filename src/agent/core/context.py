from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Protocol

from agent.core.capabilities import ModelCapabilities
from agent.providers.base import ProviderRequest


DEFAULT_SAFETY_MARGIN_TOKENS = 1_024
DEFAULT_CHARACTERS_PER_TOKEN = 3


class ContextBudgetError(ValueError):
    """Raised when a request cannot be assigned a safe context budget."""


class TokenCounter(Protocol):
    """Anything that can estimate the number of tokens in text."""

    def count_text(self, text: str) -> int:
        ...


@dataclass(frozen=True, slots=True)
class ApproximateTokenCounter:
    """
    A conservative fallback when a provider tokenizer is unavailable.

    This is an estimate, not an exact provider token count. We deliberately
    assume only three characters per token, which leaves useful safety room
    for code, punctuation, and non-English text.
    """

    characters_per_token: int = DEFAULT_CHARACTERS_PER_TOKEN

    def __post_init__(self) -> None:
        if self.characters_per_token < 1:
            raise ValueError("characters_per_token must be at least 1")

    def count_text(self, text: str) -> int:
        if not text:
            return 0

        return math.ceil(len(text) / self.characters_per_token)


@dataclass(frozen=True, slots=True)
class RequestTokenEstimate:
    """Estimated input-token cost of one provider request."""

    system_tokens: int
    tool_tokens: int
    message_tokens: int

    def __post_init__(self) -> None:
        counts = (
            self.system_tokens,
            self.tool_tokens,
            self.message_tokens,
        )

        if any(count < 0 for count in counts):
            raise ValueError("token estimates must not be negative")

    @property
    def total_input_tokens(self) -> int:
        return (
            self.system_tokens
            + self.tool_tokens
            + self.message_tokens
        )


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """
    The portion of a model context window available for prompt input.

    The model needs room to produce an answer, and we keep a small additional
    margin because our fallback token counter is only an estimate.
    """

    context_window_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS

    def __post_init__(self) -> None:
        if self.context_window_tokens < 1:
            raise ContextBudgetError(
                "context_window_tokens must be at least 1"
            )

        if self.reserved_output_tokens < 1:
            raise ContextBudgetError(
                "reserved_output_tokens must be at least 1"
            )

        if self.safety_margin_tokens < 0:
            raise ContextBudgetError(
                "safety_margin_tokens must not be negative"
            )

        if (
            self.reserved_output_tokens + self.safety_margin_tokens
            >= self.context_window_tokens
        ):
            raise ContextBudgetError(
                "output reservation and safety margin must leave room "
                "for prompt input"
            )

    @property
    def max_input_tokens(self) -> int:
        return (
            self.context_window_tokens
            - self.reserved_output_tokens
            - self.safety_margin_tokens
        )


@dataclass(frozen=True, slots=True)
class ContextAssessment:
    """The result of checking one request against its model's context limit."""

    estimate: RequestTokenEstimate
    budget: ContextBudget

    @property
    def remaining_input_tokens(self) -> int:
        return max(
            0,
            self.budget.max_input_tokens
            - self.estimate.total_input_tokens,
        )

    @property
    def overflow_tokens(self) -> int:
        return max(
            0,
            self.estimate.total_input_tokens
            - self.budget.max_input_tokens,
        )

    @property
    def needs_compaction(self) -> bool:
        return self.overflow_tokens > 0


def estimate_request_tokens(
    request: ProviderRequest,
    token_counter: TokenCounter | None = None,
) -> RequestTokenEstimate:
    """
    Estimate the provider input size.

    Prompt caching may reduce cost, but cached tokens still occupy context
    space, so cached system instructions and tools are counted here too.
    """
    counter = token_counter or ApproximateTokenCounter()

    system_tokens = counter.count_text(request.system_prompt or "")

    tools_payload = [
        tool.model_dump(mode="json")
        for tool in request.tools
    ]
    tool_tokens = counter.count_text(_canonical_json(tools_payload))

    messages_payload = [
        message.model_dump(mode="json")
        for message in request.messages
    ]
    message_tokens = counter.count_text(_canonical_json(messages_payload))

    return RequestTokenEstimate(
        system_tokens=system_tokens,
        tool_tokens=tool_tokens,
        message_tokens=message_tokens,
    )


def assess_request_context(
    request: ProviderRequest,
    capabilities: ModelCapabilities,
    token_counter: TokenCounter | None = None,
    *,
    safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS,
) -> ContextAssessment:
    """
    Decide whether the request fits before sending it to a provider.

    A request that does not fit is not a provider failure yet. It is a signal
    for the agent loop to compact its conversation and try again.
    """
    if request.model != capabilities.model:
        raise ContextBudgetError(
            "request model does not match supplied capabilities: "
            f"{request.model!r} != {capabilities.model!r}"
        )

    if request.max_tokens > capabilities.max_output_tokens:
        raise ContextBudgetError(
            f"request max_tokens ({request.max_tokens}) exceeds "
            f"{request.model!r}'s output limit "
            f"({capabilities.max_output_tokens})"
        )

    budget = ContextBudget(
        context_window_tokens=capabilities.context_window_tokens,
        reserved_output_tokens=request.max_tokens,
        safety_margin_tokens=safety_margin_tokens,
    )

    return ContextAssessment(
        estimate=estimate_request_tokens(request, token_counter),
        budget=budget,
    )


def _canonical_json(value: object) -> str:
    """Produce stable text so repeated estimates are deterministic."""
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )