import asyncio
import json

import httpx
import pytest

from agenttrace.config import EndpointConfig
from agenttrace.models import ChatMessage, MessageRole
from agenttrace.serving.client import InferenceError, InferenceRequest
from agenttrace.serving.openai import OpenAICompatibleClient


class DelayedSSE(httpx.AsyncByteStream):
    async def __aiter__(self):  # type: ignore[no-untyped-def]
        events = [
            {"id": "server-1", "choices": [{"delta": {"content": "hello"}}]},
            {"id": "server-1", "choices": [{"delta": {"content": " world"}}]},
            {
                "id": "server-1",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        ]
        for event in events:
            await asyncio.sleep(0.005)
            yield f"data: {json.dumps(event)}\n\n".encode()
        yield b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_streaming_client_records_ttft_chunks_and_usage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Request-ID"] == "req_1"
        return httpx.Response(200, stream=DelayedSSE())

    client = OpenAICompatibleClient(
        EndpointConfig(base_url="http://test/v1"), transport=httpx.MockTransport(handler)
    )
    response = await client.complete(
        InferenceRequest(
            request_id="req_1",
            model="mock",
            messages=[ChatMessage(role=MessageRole.USER, content="say hello")],
            stream=True,
        )
    )
    await client.close()
    assert response.content == "hello world"
    assert response.ttft_seconds is not None and response.ttft_seconds > 0
    assert len(response.chunks) == 2
    assert response.input_tokens == 4
    assert response.output_tokens == 2


@pytest.mark.asyncio
async def test_http_failure_is_retained_as_typed_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "overloaded"})

    client = OpenAICompatibleClient(
        EndpointConfig(base_url="http://test/v1"), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(InferenceError) as caught:
        await client.complete(
            InferenceRequest(
                request_id="req_fail",
                model="mock",
                messages=[ChatMessage(role=MessageRole.USER, content="hello")],
                stream=False,
            )
        )
    await client.close()
    assert caught.value.status_code == 503
    assert caught.value.retryable
