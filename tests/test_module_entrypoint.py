from __future__ import annotations

import subprocess
import sys


def test_python_module_invocation_runs_the_agent_cli():
    result = subprocess.run(
        (sys.executable, "-m", "agent", "config"),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "SETTING" in result.stdout
    assert "provider" in result.stdout
