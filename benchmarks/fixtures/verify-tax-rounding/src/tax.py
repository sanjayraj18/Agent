def total_with_tax(subtotal_cents: int, rate_percent: float) -> int:
    """Return the total in cents after applying a percentage tax rate."""
    return subtotal_cents + round(subtotal_cents * rate_percent / 10)
