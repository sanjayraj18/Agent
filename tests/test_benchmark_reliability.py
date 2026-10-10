from decimal import Decimal

import pytest

from agent.benchmark.reliability import (
    ReliabilityError,
    pass_all_k,
    pass_at_k,
    pass_one,
)


def test_pass_one_is_the_normal_single_attempt_success_rate():
    assert pass_one(attempts=8, passed_attempts=7) == Decimal("0.875")


def test_pass_all_k_measures_strict_reliability():
    # C(7, 3) / C(8, 3) = 35 / 56 = 0.625.
    assert pass_all_k(
        attempts=8,
        passed_attempts=7,
        k=3,
    ) == Decimal("0.625")


def test_pass_at_k_measures_retry_usefulness():
    # A choice of three attempts cannot contain the only failing attempt alone.
    assert pass_at_k(
        attempts=8,
        passed_attempts=7,
        k=3,
    ) == Decimal("1")


def test_pass_at_k_uses_the_complement_of_all_failures():
    # 5 pass and 3 fail. Only one of C(8, 3) selections contains all failures.
    assert pass_at_k(
        attempts=8,
        passed_attempts=5,
        k=3,
    ) == Decimal("55") / Decimal("56")


@pytest.mark.parametrize(
    ("attempts", "passed_attempts", "k"),
    (
        (0, 0, 1),
        (3, -1, 1),
        (3, 4, 1),
        (3, 2, 0),
        (3, 2, 4),
    ),
)
def test_reliability_metrics_reject_impossible_outcomes(
    attempts: int,
    passed_attempts: int,
    k: int,
):
    with pytest.raises(ReliabilityError):
        pass_all_k(
            attempts=attempts,
            passed_attempts=passed_attempts,
            k=k,
        )
