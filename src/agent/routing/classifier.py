from __future__ import annotations

from agent.routing.signals import RoutingSignals, TaskComplexity


_SIMPLE_TASK_MARKERS = (
    "typo",
    "spelling",
    "rename",
    "readme",
    "comment",
    "format",
    "formatting",
    "whitespace",
)

_COMPLEX_TASK_MARKERS = (
    "async",
    "concurrency",
    "race condition",
    "deadlock",
    "distributed",
    "security",
    "authentication",
    "authorization",
    "migration",
    "architecture",
    "refactor",
    "performance",
    "benchmark",
    "flaky",
    "stack trace",
    "traceback",
    "multiple files",
)

_SIMPLE_MAX_WORDS = 40
_COMPLEX_MIN_WORDS = 120


def classify_initial_prompt(
    prompt: str,
    *,
    turn_number: int = 1,
) -> RoutingSignals:
    """
    Convert an initial user prompt into conservative routing signals.

    This is intentionally deterministic: it does not call an LLM, spend
    tokens, or follow instructions contained inside the prompt.

    The classifier only makes an initial guess. Tool errors and verification
    failures later provide stronger evidence and can escalate the route.
    """

    normalized = " ".join(prompt.split()).lower()

    if not normalized:
        raise ValueError("prompt must not be blank")

    return RoutingSignals(
        task_complexity =_classify_complexity(normalized),
        turn_number =turn_number,
    )


def _classify_complexity(prompt: str) -> TaskComplexity:
    word_count = len(prompt.split())

    if (
        word_count >= _COMPLEX_MIN_WORDS
        or _contains_marker(prompt, _COMPLEX_TASK_MARKERS)
    ):
        return "complex"

    if (
        word_count <= _SIMPLE_MAX_WORDS
        and _contains_marker(prompt, _SIMPLE_TASK_MARKERS)
    ):
        return "simple"

    return "moderate"


def _contains_marker(
    prompt: str,
    markers: tuple[str, ...],
) -> bool:
    return any(marker in prompt for marker in markers)