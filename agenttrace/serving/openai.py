"""Async OpenAI-compatible HTTP client with streamed timing observations."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from agenttrace.config import EndpointConfig
from agenttrace.instrumentation.tokens import TokenCounter, WhitespaceTokenCounter
from agenttrace.serving.client import (
    InferenceError,
    InferenceRequest,
    InferenceResponse,
    StreamObservation,
)


class OpenAICompatibleClient:
    def __init__(
        self,
        config: EndpointConfig,
        *,
        token_counter: TokenCounter | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.counter = token_counter or WhitespaceTokenCounter()
        headers = {"Accept": "application/json"}
        if config.api_key is not None:
            headers["Authorization"] = f"Bearer {config.api_key.get_secret_value()}"
        self._http = httpx.AsyncClient(
            timeout=config.timeout_seconds,
            headers=headers,
            transport=transport,
        )
        self._url = str(config.base_url).rstrip("/") + "/chat/completions"

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        if request.stream:
            return await self._complete_stream(request)
        return await self._complete_json(request)

    async def _complete_json(self, request: InferenceRequest) -> InferenceResponse:
        started = time.monotonic()
        try:
            response = await self._http.post(
                self._url,
                json=request.openai_payload(),
                headers=self._request_headers(request),
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise self._convert_error(exc) from exc
        elapsed = time.monotonic() - started
        choice = body.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = body.get("usage") or {}
        return InferenceResponse(
            request_id=request.request_id,
            content=content,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            elapsed_seconds=elapsed,
            ttft_seconds=None,
            server_request_id=body.get("id"),
            finish_reason=choice.get("finish_reason"),
            raw_metadata={"created": body.get("created")},
        )

    async def _complete_stream(self, request: InferenceRequest) -> InferenceResponse:
        started = time.monotonic()
        chunks: list[StreamObservation] = []
        usage: dict[str, int] = {}
        server_request_id: str | None = None
        finish_reason: str | None = None
        try:
            async with self._http.stream(
                "POST",
                self._url,
                json=request.openai_payload(),
                headers=self._request_headers(request),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    server_request_id = server_request_id or event.get("id")
                    if event.get("usage"):
                        usage = event["usage"]
                    choice = (event.get("choices") or [{}])[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    text = choice.get("delta", {}).get("content") or ""
                    if text:
                        chunks.append(
                            StreamObservation(
                                elapsed_seconds=time.monotonic() - started,
                                text=text,
                                estimated_tokens=self.counter.count(text),
                            )
                        )
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise self._convert_error(exc) from exc
        elapsed = time.monotonic() - started
        content = "".join(chunk.text for chunk in chunks)
        return InferenceResponse(
            request_id=request.request_id,
            content=content,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            elapsed_seconds=elapsed,
            ttft_seconds=chunks[0].elapsed_seconds if chunks else None,
            chunks=chunks,
            server_request_id=server_request_id,
            finish_reason=finish_reason,
        )

    def _request_headers(self, request: InferenceRequest) -> dict[str, str]:
        return {"X-Request-ID": request.request_id, **request.extra_headers}

    @staticmethod
    def _convert_error(exc: Exception) -> InferenceError:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return InferenceError(
                f"inference endpoint returned HTTP {status}",
                status_code=status,
                retryable=status == 429 or status >= 500,
            )
        return InferenceError(str(exc), retryable=isinstance(exc, httpx.TransportError))

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> OpenAICompatibleClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
