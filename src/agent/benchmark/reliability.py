"""Pure, auditable reliability metrics for repeated agent attempts."""

from __future__ import annotations

from decimal import Decimal
from math import comb


class ReliabilityError(ValueError):
    """Observed benchmark outcomes cannot form a valid reliability estimate."""


def pass_one(*,attempts: int,passed_attempts: int) -> Decimal:
    """
    Estimate pass¹: probability that one isolated attempt succeeds.

    Example:
        7 passes from 8 attempts -> 7 / 8 -> 0.875
    """
    _validate_outcomes(
        attempts=attempts,
        passed_attempts=passed_attempts,
    )
    return Decimal(passed_attempts) / Decimal(attempts)


def pass_all_k(*,attempts: int,passed_attempts: int,k: int,) -> Decimal:
    """
    Estimate pass^k: probability that all k selected attempts succeed.

    This is the strict reliability metric. It answers:

        “If this task is attempted k times, how likely is every attempt
        to succeed?”

    Formula:
        C(passed_attempts, k) / C(attempts, k)
    """
    _validate_k(
        attempts=attempts,
        passed_attempts=passed_attempts,
        k=k,
    )

    denominator = comb(attempts, k)
    numerator = (
        comb(passed_attempts, k)
        if passed_attempts >= k
        else 0
    )

    return Decimal(numerator) / Decimal(denominator)


def pass_at_k(*,attempts: int,passed_attempts: int,k: int,) -> Decimal:
    """
    Estimate pass@k: probability that at least one of k attempts succeeds.

    This is the retry-usefulness metric. It answers:

        “If we allow at most k independent attempts, how likely are we
        to get one successful result?”

    Formula:
        1 - C(failed_attempts, k) / C(attempts, k)
    """
    _validate_k(
        attempts=attempts,
        passed_attempts=passed_attempts,
        k=k,
    )

    failed_attempts = attempts - passed_attempts
    denominator = comb(attempts, k)
    all_failures = (
        comb(failed_attempts, k)
        if failed_attempts >= k
        else 0
    )

    return Decimal("1") - (
        Decimal(all_failures) / Decimal(denominator)
    )


def _validate_outcomes(
    *,
    attempts: int,
    passed_attempts: int,
) -> None:
    if attempts < 1:
        raise ReliabilityError("attempts must be at least 1")

    if not 0 <= passed_attempts <= attempts:
        raise ReliabilityError(
            "passed_attempts must be between 0 and attempts"
        )


def _validate_k(
    *,
    attempts: int,
    passed_attempts: int,
    k: int,
) -> None:
    _validate_outcomes(
        attempts=attempts,
        passed_attempts=passed_attempts,
    )

    if not 1 <= k <= attempts:
        raise ReliabilityError(
            "k must be between 1 and attempts"
        )