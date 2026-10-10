from src.labels import parse_labels


def test_parse_labels_strips_whitespace_and_omits_empty_values() -> None:
    assert parse_labels(" bug, feature, , urgent ") == [
        "bug",
        "feature",
        "urgent",
    ]


def test_parse_labels_handles_an_empty_input() -> None:
    assert parse_labels("") == []
