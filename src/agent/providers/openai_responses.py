from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from agent.auth.credentials import OpenAIApiKey
from agent.events import ErrorEvent, Event
from agent.providers.base import EventFactory, ProviderMetadata, ProviderRequest
from agent.providers.errors import classify_http, classify_transport, to_event
from agent.providers.openai_wire import OpenAIResponsesTranslator, build_body
from agent.providers.sse import iter_sse


DEFAULT_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=600.0,
    write=30.0,
    pool=10.0,
)


class OpenAIResponsesProvider:
    """Raw HTTP implementation of the OpenAI Responses streaming API."""

    name = "openai-responses"

    def __init__(
        self,
        credential: OpenAIApiKey,
        *,
        base_url: str = "https://api.openai.com",
        client: httpx.AsyncClient | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self._credential = credential
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout or DEFAULT_TIMEOUT
        )
        self.metadata = ProviderMetadata(
            provider_id="openai",
            base_url=self._base_url,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        return {
            **self._credential.headers(),
            "content-type": "application/json",
            "accept": "text/event-stream",
        }

    async def stream(
        self,
        request: ProviderRequest,
        emit: EventFactory,
    ) -> AsyncIterator[Event]:
        try:
            body = build_body(request)
        except ValueError as exc:
            yield emit(
                ErrorEvent,
                kind="invalid_request",
                message=str(exc),
                retryable=False,
            )
            return

        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/v1/responses",
                json=body,
                headers=self._headers(),
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    try:
                        failure_body: Any = response.json()
                    except ValueError:
                        failure_body = response.text
                    yield to_event(
                        emit,
                        classify_http(
                            response.status_code,
                            body=failure_body,
                            headers=response.headers,
                        ),
                    )
                    return

                translator = OpenAIResponsesTranslator(emit)
                async for frame in iter_sse(response.aiter_bytes()):
                    for event in translator.push(frame):
                        yield event
                for event in translator.close():
                    yield event

        except httpx.RequestError as exc:
            yield to_event(emit, classify_transport(exc))
