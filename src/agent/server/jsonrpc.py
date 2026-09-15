
import asyncio
import json
import logging
from typing import AsyncIterator, Callable, TextIO, Any

from agent.events import ErrorEvent, Event, dumps


logger = logging.getLogger(__name__)

AgentRun = Callable[[str], AsyncIterator[Event]]

class JsonRpcServer:
    def __init__(self, run_agent : AgentRun) -> None:
        self._run_agent = run_agent

    async def serve(self, reader : TextIO, writer : TextIO) -> None:
        while True:
            line = await asyncio.to_thread(reader.readline)
            if line == "":
                return

            async for message in self.handle_line(line):
                writer.write(
                    json.dumps(
                        message,
                        separators=(",", ":")
                    )
                )
                writer.write("\n")
                writer.flush()


    async def handle_line(self, line : str) -> AsyncIterator[dict[str,Any]]:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            yield self._error(
                request_id = None,
                code=-32700,
                message="parse error",
            )
            return

        if not isinstance(request, dict):
            yield self._error(
                request_id=None,
                code=-32600,
                message="invalid request",
            )
            return

        request_id = request.get("id")

        if request.get("jsonrpc") != "2.0":
            yield self._error(
                request_id=request_id,
                code=-32600,
                message="invalid request",
            )
            return

        method = request.get("method")
        if not isinstance(method, str):
            yield self._error(
                request_id=request_id,
                code=-32600,
                message="invalid request",
            )
            return

        if method != "agent.run":
            yield self._error(
                request_id=request_id,
                code=-32601,
                message=f"method not found: {method}",
            )
            return

        params = request.get("params", {})
        if not isinstance(params, dict):
            yield self._error(
                request_id=request_id,
                code=-32602,
                message="invalid params",
            )
            return

        prompt = params.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            yield self._error(
                request_id=request_id,
                code=-32602,
                message="params.prompt must be a non-empty string",
            )
            return

        failed = False

        try:
            async for event in self._run_agent(prompt):
                if isinstance(event, ErrorEvent):
                    failed = True

                yield {
                    "jsonrpc": "2.0",
                    "method": "agent.event",
                    "params": {
                        "request_id": request_id,
                        "event": json.loads(dumps(event)),
                    },
                }

        except Exception:
            logger.exception("agent execution failed")

            yield self._error(
                request_id=request_id,
                code=-32603,
                message="internal error",
            )
            return

        yield {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "status": "failed" if failed else "completed",
            },
        }

    @staticmethod
    def _error(request_id: str | int | None,code: int, message: str,) -> dict[str, Any]:
        return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": code,
                    "message": message,
                },
            }
