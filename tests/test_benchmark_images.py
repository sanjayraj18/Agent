import pytest

from agent.benchmark.images import PinnedImageError, validate_pinned_image


def test_pinned_image_requires_a_full_sha256_digest():
    image = "registry.example/python@sha256:" + "a" * 64
    assert validate_pinned_image(image) == image

    with pytest.raises(PinnedImageError, match="@sha256"):
        validate_pinned_image("python:3.12")
