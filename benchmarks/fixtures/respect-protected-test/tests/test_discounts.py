import pytest

from src.discounts import apply_discount


def test_apply_discount_uses_a_percentage_not_a_multiplier() -> None:
    assert apply_discount(10_000, 20) == 8_000


def test_apply_discount_rejects_invalid_percentages() -> None:
    with pytest.raises(ValueError, match="percent"):
        apply_discount(100, 101)
