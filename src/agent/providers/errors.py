from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping

from agent.events import ErrorEvent
from agent.providers.base import EventFactory


@dataclass(frozen=True)
class Failure:
    """A classified failure. The input to a retry decision."""

    kind: str
    message: str
    retryable: bool
    retry_after: float | None = None


# Status code -> (kind, retryable). The HTTP status is authoritative for
# retryability: a 400 must never become retryable because of its body text.
_STATUS_KINDS: dict[int, tuple[str, bool]] = {
    400: ("invalid_request_error", False),
    401: ("authentication_error", False),
    403: ("permission_error", False),
    404: ("not_found_error", False),
    408: ("timeout_error", True),
    409: ("conflict_error", True),
    413: ("request_too_large", False),
    422: ("invalid_request_error", False),
    429: ("rate_limit_error", True),
    500: ("api_error", True),
    502: ("gateway_error", True),
    503: ("gateway_error", True),
    504: ("gateway_error", True),
    529: ("overloaded_error", True),
}


_RETRYABLE_STREAM_TYPES = {
    "api_error",
    "overloaded_error",
    "rate_limit_error",
    "timeout_error",
}


_CONTEXT_OVERFLOW_TYPES = {
    "context_length_exceeded",
    "context_window_exceeded",
    "input_length_exceeded",
    "request_too_large",
}


_CONTEXT_OVERFLOW_MARKERS = (
    "context length",
    "context window",
    "maximum context",
    "prompt is too long",
    "prompt too long",
    "input is too long",
    "input too long",
    "too many tokens",
    "token limit",
)


def parse_retry_after(
    value: str | None,
    *,
    now: datetime | None = None,
) -> float | None:
    """Retry-After is either delta-seconds or an HTTP-date. Both are legal."""
    if not value:
        return None

    value = value.strip()

    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None

    if when is None:
        return None

    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)

    return max(
        0.0,
        (when - (now or datetime.now(timezone.utc))).total_seconds(),
    )


def is_context_overflow(
    *,
    error_type: str | None = None,
    message: str | None = None,
    status_code: int | None = None,
) -> bool:
    """
    Recognize errors where compaction may let the agent continue.

    A context overflow is terminal for the current request, so it must never
    enter the normal retry-with-delay path.
    """
    if status_code == 413:
        return True

    normalized_type = (error_type or "").strip().lower()

    if normalized_type in _CONTEXT_OVERFLOW_TYPES:
        return True

    normalized_message = (message or "").lower()

    return any(
        marker in normalized_message
        for marker in _CONTEXT_OVERFLOW_MARKERS
    )


def classify_http(
    status_code: int,
    *,
    body: Any = None,
    headers: Mapping[str, str] | None = None,
) -> Failure:
    """Convert one non-200 HTTP response into a classified failure."""
    headers = headers or {}

    kind, retryable = _STATUS_KINDS.get(
        status_code,
        (f"http_{status_code}", 500 <= status_code < 600),
    )

    provider_error_type = ""
    detail = ""

    if isinstance(body, dict):
        error = body.get("error")

        if isinstance(error, dict):
            raw_type = error.get("type")
            raw_message = error.get("message")

            if isinstance(raw_type, str):
                provider_error_type = raw_type
                kind = raw_type

            if isinstance(raw_message, str):
                detail = raw_message

    elif isinstance(body, str):
        detail = body[:500]

    if is_context_overflow(
        error_type=provider_error_type or kind,
        message=detail,
        status_code=status_code,
    ):
        kind = "context_overflow"
        retryable = False

    message = (
        f"HTTP {status_code}: {detail}"
        if detail
        else f"HTTP {status_code}"
    )

    return Failure(
        kind=kind,
        message=message,
        retryable=retryable,
        retry_after=(
            parse_retry_after(headers.get("retry-after"))
            if retryable
            else None
        ),
    )


def to_event(emit: EventFactory, failure: Failure) -> ErrorEvent:
    return emit(
        ErrorEvent,
        kind=failure.kind,
        message=failure.message,
        retryable=failure.retryable,
        retry_after=failure.retry_after,
    )


def classify_stream(
    error_type: str,
    message: str = "",
) -> Failure:
    """Convert a mid-stream provider error into a classified failure."""
    kind = error_type or "api_error"

    if is_context_overflow(
        error_type=kind,
        message=message,
    ):
        return Failure(
            kind="context_overflow",
            message=message,
            retryable=False,
        )

    return Failure(
        kind=kind,
        message=message,
        retryable=kind in _RETRYABLE_STREAM_TYPES,
    )


def classify_transport(exc: BaseException) -> Failure:
    """A connection-level failure means the request never got an answer."""
    return Failure(
        kind="connection_error",
        message=f"{type(exc).__name__}: {exc}",
        retryable=True,
    )


@dataclass(frozen=True)
class RetryPolicy:
    """Pure policy. Owns no state — the caller counts attempts."""

    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.5

    def delay_for(
        self,
        attempt: int,
        failure: Failure,
        *,
        rand: Callable[[], float] = random.random,
    ) -> float | None:
        """
        Return seconds to wait before another attempt, or None to give up.

        `attempt` is one-based: 1 means the first request just failed.
        """
        if not failure.retryable or attempt >= self.max_attempts:
            return None

        if failure.retry_after is not None:
            if failure.retry_after > self.max_delay:
                return None

            return failure.retry_after * (
                1.0 + self.jitter * rand()
            )

        backoff = min(
            self.base_delay * (2 ** (attempt - 1)),
            self.max_delay,
        )

        return (
            backoff * (1.0 - self.jitter)
            + backoff * self.jitter * rand()
        )