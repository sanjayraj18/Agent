import pytest

from agent.core.retry import RetryPolicy
from agent.events import ErrorEvent


def _error(
    *,
    retryable: bool,
    retry_after: float | None = None,
) -> ErrorEvent:
    return ErrorEvent(
        seq=1,
        session_id="session-1",
        kind="rate_limit_error",
        message="Too many requests",
        retryable=retryable,
        retry_after=retry_after,
    )


def test_retries_only_retryable_errors_within_the_retry_budget():
    policy = RetryPolicy(max_retries=2)

    retryable_error = _error(retryable=True)
    permanent_error = _error(retryable=False)

    assert policy.should_retry(retryable_error, retries_completed=0) is True
    assert policy.should_retry(retryable_error, retries_completed=1) is True
    assert policy.should_retry(retryable_error, retries_completed=2) is False

    assert policy.should_retry(permanent_error, retries_completed=0) is False


def test_uses_exponential_backoff_with_deterministic_jitter():
    policy = RetryPolicy(
        base_delay=0.5,
        max_delay=30.0,
        jitter_fraction=0.20,
        random_source=lambda: 0.5,
    )
    error = _error(retryable=True)

    assert policy.delay_for(error, retry_number=1) == pytest.approx(0.55)
    assert policy.delay_for(error, retry_number=2) == pytest.approx(1.10)
    assert policy.delay_for(error, retry_number=3) == pytest.approx(2.20)


def test_retry_after_is_a_minimum_wait_with_upward_jitter():
    policy = RetryPolicy(
        jitter_fraction=0.20,
        random_source=lambda: 0.5,
    )
    error = _error(retryable=True, retry_after=12.0)

    assert policy.delay_for(error, retry_number=1) == pytest.approx(13.2)


def test_rejects_invalid_retry_policy_configuration():
    with pytest.raises(ValueError, match="max_retries must not be negative"):
        RetryPolicy(max_retries=-1)

    with pytest.raises(ValueError, match="base_delay must be greater than zero"):
        RetryPolicy(base_delay=0)

    with pytest.raises(
        ValueError,
        match="max_delay must be greater than or equal to base_delay",
    ):
        RetryPolicy(base_delay=5, max_delay=1)

    with pytest.raises(
        ValueError,
        match="jitter_fraction must be between 0 and 1",
    ):
        RetryPolicy(jitter_fraction=1.5)