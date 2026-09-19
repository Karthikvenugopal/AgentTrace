from prometheus_client import CollectorRegistry, generate_latest

from agenttrace.telemetry.metrics import ApplicationMetrics


def test_metrics_use_bounded_labels_and_client_namespace() -> None:
    registry = CollectorRegistry()
    metrics = ApplicationMetrics(registry)
    with metrics.track_request("replay", "mock"):
        metrics.observe_request(
            component="replay",
            model="mock",
            status="succeeded",
            latency_seconds=0.2,
            ttft_seconds=0.05,
            input_tokens=10,
            output_tokens=2,
        )
    metrics.observe_tool(tool="attacker-controlled-name", status="denied", duration_seconds=0.01)
    output = generate_latest(registry).decode()
    assert "agenttrace_client_requests_total" in output
    assert 'tool="other"' in output
    assert "request_id" not in output
