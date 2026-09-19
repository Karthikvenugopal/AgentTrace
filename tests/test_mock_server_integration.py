import threading

import httpx
import pytest

from agenttrace.config import EndpointConfig
from agenttrace.models import ChatMessage, MessageRole
from agenttrace.serving.client import InferenceRequest
from agenttrace.serving.mock_server import MockServerConfig, create_server
from agenttrace.serving.openai import OpenAICompatibleClient


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_http_stream_against_cpu_mock_server() -> None:
    try:
        server = create_server(
            MockServerConfig(
                port=0, first_token_delay_seconds=0.002, inter_chunk_delay_seconds=0.001
            )
        )
    except PermissionError:
        pytest.skip("local sandbox does not permit binding a loopback port")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    client = OpenAICompatibleClient(
        EndpointConfig(base_url=f"http://127.0.0.1:{port}/v1", model="agenttrace-mock")
    )
    try:
        response = await client.complete(
            InferenceRequest(
                request_id="integration-request",
                model="agenttrace-mock",
                messages=[ChatMessage(role=MessageRole.USER, content="hello")],
                max_tokens=6,
                stream=True,
            )
        )
        assert response.output_tokens == 6
        assert response.ttft_seconds is not None
        assert len(response.chunks) >= 2
        async with httpx.AsyncClient() as http:
            models = (await http.get(f"http://127.0.0.1:{port}/v1/models")).json()
        assert models["data"][0]["id"] == "agenttrace-mock"
    finally:
        await client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
