from src.math_utils import add


def test_adds_positive_integers() -> None:
    assert add(2, 3) == 5


def test_adds_negative_integers() -> None:
    assert add(-2, 3) == 1
