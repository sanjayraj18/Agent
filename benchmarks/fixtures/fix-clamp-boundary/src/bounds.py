def clamp(value: int, lower: int, upper: int) -> int:
    """Keep value inside the inclusive lower/upper range."""
    if lower > upper:
        raise ValueError("lower cannot exceed upper")

    if value < lower:
        return lower

    if value > upper:
        return lower

    return value
