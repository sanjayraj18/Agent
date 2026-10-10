def apply_discount(price_cents: int, percent: int) -> int:
    """Return a price after applying a whole-number percentage discount."""
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")

    return price_cents - int(price_cents * percent)
