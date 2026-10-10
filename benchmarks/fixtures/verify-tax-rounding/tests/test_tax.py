from src.tax import total_with_tax


def test_total_with_tax_uses_percentage_points() -> None:
    assert total_with_tax(10_000, 7.5) == 10_750


def test_total_with_tax_rounds_to_the_nearest_cent() -> None:
    assert total_with_tax(199, 8.5) == 216
