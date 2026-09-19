import httpx
import pytest

from agenttrace.serving.metrics import VLLMMetricsAdapter, metric_semantics


@pytest.mark.asyncio
async def test_metrics_adapter_marks_unsupported_values(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = """# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="mock"} 2
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="mock"} 100
"""

    async def fake_get(self, url):  # type: ignore[no-untyped-def]
        return httpx.Response(200, text=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    snapshot = await VLLMMetricsAdapter("http://server/metrics").collect()
    assert snapshot.available["running_requests"]["vllm:num_requests_running"] == 2
    assert "queue_time_seconds" in snapshot.unavailable
    assert "not an exact request" in metric_semantics()["queue_time_seconds"]
