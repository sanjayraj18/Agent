from datetime import datetime, timedelta, timezone

from agent.providers.errors import (
    Failure, RetryPolicy, classify_http, classify_stream, classify_transport,
    parse_retry_after,
)


def test_retry_after_seconds():
    assert parse_retry_after("30") == 30.0


def test_retry_after_http_date():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    later = (now + timedelta(seconds=45)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(later, now=now) == 45.0


def test_retry_after_past_date_clamps_to_zero():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    past = (now - timedelta(seconds=60)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(past, now=now) == 0.0


def test_retry_after_absent_or_garbage():
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None


def test_429_is_retryable_with_hint():
    f = classify_http(429, body={"error": {"type": "rate_limit_error", "message": "slow"}},
                      headers={"retry-after": "12"})
    assert (f.kind, f.retryable, f.retry_after) == ("rate_limit_error", True, 12.0)


def test_529_and_429_are_distinct_kinds():
    assert classify_http(529).kind == "overloaded_error"
    assert classify_http(429).kind == "rate_limit_error"
    assert classify_http(529).retryable and classify_http(429).retryable


def test_401_is_terminal():
    assert classify_http(401).retryable is False


def test_body_error_type_cannot_make_a_400_retryable():
    """The most important rule in this file."""
    f = classify_http(400, body={"error": {"type": "api_error", "message": "bad"}})
    assert f.kind == "api_error"      # refined
    assert f.retryable is False       # not upgraded


def test_unknown_5xx_is_retryable_by_default():
    assert classify_http(507).retryable is True
    assert classify_http(418).retryable is False


def test_non_json_body_still_produces_a_message():
    f = classify_http(502, body="<html>bad gateway</html>")
    assert "bad gateway" in f.message


def test_stream_error_classification():
    assert classify_stream("overloaded_error").retryable is True
    assert classify_stream("invalid_request_error").retryable is False


def test_transport_errors_are_always_retryable():
    assert classify_transport(ConnectionResetError("reset")).retryable is True


# --------------------------------------------------------------------- policy

RETRYABLE = Failure(kind="overloaded_error", message="", retryable=True)
TERMINAL = Failure(kind="authentication_error", message="", retryable=False)


def test_terminal_failures_never_retry():
    assert RetryPolicy().delay_for(1, TERMINAL) is None


def test_attempts_are_capped():
    p = RetryPolicy(max_attempts=3)
    assert p.delay_for(2, RETRYABLE) is not None
    assert p.delay_for(3, RETRYABLE) is None


def test_backoff_is_exponential_and_bounded():
    p = RetryPolicy(base_delay=1.0, max_delay=60.0, jitter=0.5, max_attempts=99)
    lo = [p.delay_for(n, RETRYABLE, rand=lambda: 0.0) for n in (1, 2, 3, 4)]
    hi = [p.delay_for(n, RETRYABLE, rand=lambda: 1.0) for n in (1, 2, 3, 4)]
    assert lo == [0.5, 1.0, 2.0, 4.0]
    assert hi == [1.0, 2.0, 4.0, 8.0]
    assert p.delay_for(20, RETRYABLE, rand=lambda: 1.0) == 60.0


def test_jitter_never_waits_less_than_the_server_asked():
    f = Failure(kind="rate_limit_error", message="", retryable=True, retry_after=10.0)
    p = RetryPolicy()
    assert p.delay_for(1, f, rand=lambda: 0.0) == 10.0
    assert p.delay_for(1, f, rand=lambda: 1.0) > 10.0


def test_retry_after_beyond_max_delay_gives_up():
    f = Failure(kind="rate_limit_error", message="", retryable=True, retry_after=900.0)
    assert RetryPolicy(max_delay=60.0).delay_for(1, f) is None