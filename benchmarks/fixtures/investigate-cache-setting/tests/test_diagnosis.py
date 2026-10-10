import json
from pathlib import Path


def test_diagnosis_records_the_configured_cache_timeout() -> None:
    diagnosis = json.loads(Path("diagnosis.json").read_text(encoding="utf-8"))

    assert diagnosis == {
        "setting": "CACHE_TTL_SECONDS",
        "unit": "seconds",
        "value": 60,
    }
