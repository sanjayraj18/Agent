from __future__ import annotations
from typing import Any, AsyncIterator
import httpx

from agent.auth.credentials import Credential
from agent.events import ErrorEvent, Event
from agent.providers.anthropic_wire import AnthropicTranslator
from agent.providers.base import ContentPart, EventFactory, Message, ProviderRequest
from agent.providers.errors import Failure, classify_http, classify_transport
from agent.providers.sse import iter_sse

ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)



#converting to anthropic json
def _part(part : ContentPart) -> dict[str, Any]:
    if part.type == "text":
        return {"type": "text", "text": part.text}

    if part.type == "thinking":
        block: dict[str, Any] = {"type": "thinking", "thinking": part.text}
        if part.signature:
            block["signature"] = part.signature

        return block

    if part.type == "tool_use":
        return {
            "type": "tool_use",
            "id": part.call_id,
            "name": part.name,
            "input": part.arguments,
        }

    if part.type == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": part.call_id,
            "content": part.content,
            "is_error": part.is_error,
        }

    raise ValueError(f"unhandled content part: {part.type!r}")


#converting message to anthropic json
def _message(message: Message) -> dict[str, Any]:
    return {"role": message.role, "content": [_part(p) for p in message.content]}


#ProviderRequest -> Anthropic Messages API JSON.
def build_body(request: ProviderRequest) -> dict[str, Any]:
    
    if not request.thinking and request.effort in ("xhigh", "max"):
        raise ValueError(
            "disabled thinking is only accepted at effort 'high' or below; "
            f"got effort={request.effort!r}"
        )

    system_prompt = request.system_prompt

    body: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "stream": True,
        "messages": [_message(m) for m in request.messages],
    }

    body["thinking"] = (
        {"type": "adaptive", "display": "summarized"}
        if request.thinking
        else {"type": "disabled"}
    )

    if request.effort is not None:
        body["output_config"] = {"effort": request.effort}

    if request.tools:
        body["tools"] = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in request.tools
        ]

    if system_prompt:
        system_block: dict[str, Any] = {
            "type": "text",
            "text": system_prompt,
        }
        body["system"] = [
            system_block
        ]

    if request.cache_stable_prefix:
        cache_plan = request.cache_plan
        assert cache_plan is not None

        if system_prompt:
            body["system"][-1]["cache_control"] = {
                "type": "ephemeral",
            }
        elif request.tools:
            body["tools"][-1]["cache_control"] = {
                "type": "ephemeral",
            }

    return body


class AnthropicRawProvider:
    name = "anthropic-raw"

    def __init__(self,credential: Credential,base_url :str = "https://api.anthropic.com", client : httpx.AsyncClient | None = None,  timeout: httpx.Timeout | None = None)-> None:
        self._credential = credential
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout or DEFAULT_TIMEOUT)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str,str]:
        return {
             **self._credential.headers(),
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            "accept": "text/event-stream",
        }

    async def stream(self, request: ProviderRequest, emit: EventFactory) -> AsyncIterator[Event]:
        try:
            body = build_body(request=request)
        except ValueError as exc:
            yield emit(
                ErrorEvent, kind="invalid_request", message=str(exc), retryable=False
            )
            return

        url = f"{self._base_url}/v1/messages"

        try:
            async with self._client.stream(
                "POST", url, json=body, headers=self._headers()
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    try:
                        body: Any = response.json()
                    except ValueError:
                        body = response.text
                    yield _event(emit, classify_http(
                        response.status_code, body=body, headers=response.headers
                    ))
                    return
        

                translator = AnthropicTranslator(emit)
                async for frame in iter_sse(response.aiter_bytes()):
                    for event in translator.push(frame):
                        yield event
                for event in translator.close():
                    yield event

        except httpx.RequestError as exc:
            yield _event(emit, classify_transport(exc))


def _event(emit: EventFactory, failure: Failure) -> ErrorEvent:
    return emit(
        ErrorEvent,
        kind=failure.kind,
        message=failure.message,
        retryable=failure.retryable,
        retry_after=failure.retry_after,
    )
