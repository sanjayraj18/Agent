from __future__ import annotations

from typing import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from agent.auth.credentials import ApiKey, Credential, OAUTH_BETA, OAuthToken
from agent.events import ErrorEvent, Event
from agent.providers.anthropic_raw import build_body
from agent.providers.anthropic_wire import AnthropicTranslator
from agent.providers.base import EventFactory, ProviderRequest
from agent.providers.errors import classify_http, classify_transport, to_event

def _client_for(credential: Credential, timeout: float) -> AsyncAnthropic:
    if isinstance(credential, ApiKey):
        return AsyncAnthropic(
            api_key=credential.value.get_secret_value(),
            max_retries=0,
            timeout=timeout,
        )
    if isinstance(credential, OAuthToken):
        return AsyncAnthropic(
            auth_token=credential.access_token.get_secret_value(),
            default_headers={"anthropic-beta": OAUTH_BETA},
            max_retries=0,
            timeout=timeout,
        )
    raise ValueError("AnthropicSDKProvider requires an Anthropic credential")

class AnthropicSDKProvider:
    name = "anthropic-sdk"

    def __init__(
        self,
        credential: Credential | None = None,
        *,
        client: AsyncAnthropic | None = None,
        timeout: float = 600.0,
    ) -> None:
        self._owns_client = client is None
        if client is not None:
            self._client = client
        elif credential is not None:
            self._client = _client_for(credential, timeout)
        else:
            raise ValueError("pass a credential or a client")
        

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
