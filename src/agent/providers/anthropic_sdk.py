from __future__ import annotations

from typing import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from agent.events import ErrorEvent, Event
from agent.providers.anthropic_raw import build_body
from agent.providers.anthropic_wire import AnthropicTranslator
from agent.providers.base import EventFactory, ProviderRequest
from agent.providers.errors import classify_http, classify_transport, to_event


class AnthropicSDKProvider:
    name = "anthropic-sdk"

    def __init__(self,api_key: str | None = None, *, client: AsyncAnthropic | None = None, timeout: float = 600.0) -> None:
        self._owns_client = client is None
        self._client = client or AsyncAnthropic(
            api_key=api_key,
            max_retries=0,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()


    async def stream(self, request: ProviderRequest, emit: EventFactory) -> AsyncIterator[Event]:
        try:
            kwargs = build_body(request)
        except ValueError as exc:
            yield emit(
                ErrorEvent, kind="invalid_request", message=str(exc), retryable=False
            )
            return

        kwargs.pop("stream", None)

        translator = AnthropicTranslator(emit)
        try:
            stream = await self._client.messages.create(**kwargs, stream=True)
            async for sdk_event in stream:
                payload = sdk_event.model_dump(mode="json", exclude_none=True)
                for event in translator.push_payload(payload):
                    yield event
            for event in translator.close():
                yield event

        except anthropic.APIStatusError as exc:
            yield to_event(
                emit,
                classify_http(
                    exc.status_code, body=exc.body, headers=exc.response.headers
                ),
            )
        except anthropic.APIConnectionError as exc:
            yield to_event(emit, classify_transport(exc))