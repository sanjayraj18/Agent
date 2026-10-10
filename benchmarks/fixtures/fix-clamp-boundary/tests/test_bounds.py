import pytest

from src.bounds import clamp


def test_clamp_returns_the_upper_boundary_for_large_values() -> None:
    assert clamp(11, 1, 10) == 10


def test_clamp_keeps_values_already_inside_the_range() -> None:
    assert clamp(7, 1, 10) == 7


def test_clamp_rejects_an_inverted_range() -> None:
    with pytest.raises(ValueError, match="lower"):
        clamp(1, 10, 1)
