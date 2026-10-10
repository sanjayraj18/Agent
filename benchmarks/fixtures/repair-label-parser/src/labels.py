def parse_labels(raw: str) -> list[str]:
    """Parse a comma-separated list of labels."""
    return [label for label in raw.split(",") if label]
