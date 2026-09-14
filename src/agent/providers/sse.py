import codecs
from dataclasses import dataclass
from typing import AsyncIterator, Iterator


@dataclass(frozen=True)
class SSEFrame:
    event : str | None
    data : str | None
    id : str | None = None
    retry : int | None = None


class SSEParser:

    __slots__ = ("_decoder", "_buf", "_event", "_data", "_id", "_retry")

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buf: str = ""
        self._event: str | None = None
        self._data: list[str] = []
        self._id: str | None = None
        self._retry: int | None = None


    def feed(self, chunk: bytes) -> Iterator[SSEFrame]:
        self._buf += self._decoder.decode(chunk)
        while (line := self._next_line()) is not None:
            frame = self._handle_line(line)
            if frame is not None:
                yield frame


    def close(self) -> Iterator[SSEFrame]:
            self._buf += self._decoder.decode(b"", final=True)
            while (line := self._next_line(final=True)) is not None:
                frame = self._handle_line(line)
                if frame is not None:
                    yield frame

    
    def _next_line(self, final: bool = False) -> str | None:
        buf = self._buf
        idx_n = buf.find("\n")
        idx_r = buf.find("\r")

        if idx_n == -1 and idx_r == -1:
            return None

        if idx_r != -1 and (idx_n == -1 or idx_r < idx_n):
            if idx_r == len(buf) - 1 and not final:
                return None
            end = idx_r
            skip = 2 if (idx_r + 1 < len(buf) and buf[idx_r + 1] == "\n") else 1
        else:
            end = idx_n
            skip = 1

        self._buf = buf[end + skip :]
        return buf[:end]


    def _handle_line(self, line: str) -> SSEFrame | None:
        if line == "":
            return self._dispatch()
        if line.startswith(":"):
            return None  # comment / keepalive

        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]  

        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        elif field == "id" and "\0" not in value:
            self._id = value
        elif field == "retry" and value.isdigit():
            self._retry = int(value)
        return None


    def _dispatch(self) -> SSEFrame | None:
        if not self._data:
            self._event = None
            return None

        frame = SSEFrame(
            event=self._event,
            data="\n".join(self._data),
            id=self._id,
            retry=self._retry,
        )
        self._event = None
        self._data = []
        return frame


async def iter_sse(chunks: AsyncIterator[bytes]) -> AsyncIterator[SSEFrame]:
    parser = SSEParser()
    async for chunk in chunks:
        for frame in parser.feed(chunk):
            yield frame
    for frame in parser.close(): 
        yield frame
