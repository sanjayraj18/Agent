from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

from agent.server.jsonrpc import JsonRpcServer


class UnixSocketServer:
    """Expose the newline-delimited JSON-RPC server on a private Unix socket."""

    def __init__(
        self,
        *,
        rpc: JsonRpcServer,
        path: Path,
    ) -> None:
        self._rpc = rpc
        self._path = path.expanduser().resolve()
        self._server: asyncio.AbstractServer | None = None
        self._owns_socket = False

    @property
    def path(self) -> Path:
        return self._path

    async def start(self) -> None:
        """Start listening, replacing only a stale socket at this path."""

        if self._server is not None:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)

        if self._path.exists() or self._path.is_symlink():
            mode = self._path.lstat().st_mode

            if not stat.S_ISSOCK(mode):
                raise RuntimeError(
                    "refusing to replace non-socket path: "
                    f"{self._path}"
                )

            # A socket left behind by a crashed previous server has no useful
            # contents. We only unlink after confirming its file type.
            self._path.unlink()

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self._path),
        )
        self._owns_socket = True
        os.chmod(self._path, 0o600)

    async def serve_forever(self) -> None:
        await self.start()
        assert self._server is not None

        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        """Stop accepting clients and remove only the socket we created."""

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self._owns_socket:
            try:
                mode = self._path.lstat().st_mode
            except FileNotFoundError:
                pass
            else:
                if stat.S_ISSOCK(mode):
                    self._path.unlink()

            self._owns_socket = False

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            await self._rpc.serve_stream(reader, writer)
        finally:
            writer.close()
            await writer.wait_closed()
