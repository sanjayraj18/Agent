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


# status -> (kind, retryable). Status is authoritative for `retryable`.
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

# Mid-stream errors arrive with no status code — classify by type alone.
_RETRYABLE_STREAM_TYPES = {
    "api_error",
    "overloaded_error",
    "rate_limit_error",
    "timeout_error",
}


def parse_retry_after(
    value: str | None, *, now: datetime | None = None
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

    return max(0.0, (when - (now or datetime.now(timezone.utc))).total_seconds())


def classify_http(
    status_code: int,
    *,
    body: Any = None,
    headers: Mapping[str, str] | None = None,
) -> Failure:
    """A non-200 response -> Failure."""
    headers = headers or {}
    kind, retryable = _STATUS_KINDS.get(
        status_code, (f"http_{status_code}", 500 <= status_code < 600)
    )

    detail = ""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            # The API's own type is more specific than our status map, so it
            # refines `kind` — but it never overrides `retryable`. A 400 that
            # happens to say "api_error" must not become retryable, or you
            # loop forever on a malformed request.
            kind = err.get("type") or kind
            detail = err.get("message") or ""
    elif isinstance(body, str):
        detail = body[:500]

    return Failure(
        kind=kind,
        message=f"HTTP {status_code}: {detail}" if detail else f"HTTP {status_code}",
        retryable=retryable,
        retry_after=parse_retry_after(headers.get("retry-after")),
    )


def to_event(emit: EventFactory, failure: Failure) -> ErrorEvent:
    return emit(
        ErrorEvent,
        kind=failure.kind,
        message=failure.message,
        retryable=failure.retryable,
        retry_after=failure.retry_after,
    )

def classify_stream(error_type: str, message: str = "") -> Failure:
    """A mid-stream `error` event -> Failure. HTTP 200 already succeeded."""
    kind = error_type or "api_error"
    return Failure(
        kind=kind, message=message, retryable=kind in _RETRYABLE_STREAM_TYPES
    )


def classify_transport(exc: BaseException) -> Failure:
    """A connection-level failure -> Failure. The request never got an answer."""
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
    jitter: float = 0.5  # fraction of the delay that is randomized

    def delay_for(
        self,
        attempt: int,
        failure: Failure,
        *,
        rand: Callable[[], float] = random.random,
    ) -> float | None:
        """Seconds to wait before attempt+1, or None to give up.

        `attempt` is 1-based: 1 means the first try just failed.
        """
        if not failure.retryable or attempt >= self.max_attempts:
            return None

        if failure.retry_after is not None:
            if failure.retry_after > self.max_delay:
                return None  # longer than we are willing to stall
            # Honor the server, then add jitter — never less than it asked.
            return failure.retry_after * (1.0 + self.jitter * rand())

        backoff = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        return backoff * (1.0 - self.jitter) + backoff * self.jitter * rand()
