import asyncio
import sys

from agent.tools.processes import ProcessRegistry


async def _process(script: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )


async def test_registry_captures_a_background_process_result():
    registry = ProcessRegistry()
    process = await _process("print('finished')")

    started = await registry.register("print('finished')", process)
    completed = await registry.wait(started.job_id, timeout=2)

    assert completed is not None
    assert completed.job_id == started.job_id
    assert completed.pid == process.pid
    assert completed.status == "completed"
    assert completed.return_code == 0
    assert completed.output == "finished\n"
    await registry.close()


async def test_registry_marks_and_bounds_large_process_output():
    registry = ProcessRegistry(max_output_bytes=8)
    process = await _process("print('x' * 20)")

    started = await registry.register("large output", process)
    completed = await registry.wait(started.job_id, timeout=2)

    assert completed is not None
    assert completed.output == "xxxxxxxx"
    assert completed.output_truncated is True
    await registry.close()


async def test_registry_stops_a_running_background_process():
    registry = ProcessRegistry()
    process = await _process("import time; time.sleep(30)")

    started = await registry.register("sleep", process)
    stopped = await registry.stop(started.job_id)

    assert stopped is not None
    assert stopped.status == "failed"
    assert stopped.return_code is not None
    assert process.returncode is not None
    assert registry.snapshot("missing") is None
    await registry.close()
