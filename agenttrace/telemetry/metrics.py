"""Low-cardinality client-side Prometheus metrics."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)


class ApplicationMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        selected_registry = registry or REGISTRY
        self.requests = Counter(
            "agenttrace_client_requests_total",
            "Inference attempts observed by AgentTrace clients.",
            ("component", "model", "status"),
            registry=selected_registry,
        )
        self.active = Gauge(
            "agenttrace_client_active_requests",
            "Currently active AgentTrace client requests.",
            ("component", "model"),
            registry=selected_registry,
        )
        self.latency = Histogram(
            "agenttrace_client_request_latency_seconds",
            "End-to-end latency observed by the client.",
            ("component", "model"),
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 120),
            registry=selected_registry,
        )
        self.ttft = Histogram(
            "agenttrace_client_ttft_seconds",
            "Time from request submission to first non-empty streamed chunk.",
            ("component", "model"),
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
            registry=selected_registry,
        )
        self.tokens = Counter(
            "agenttrace_client_tokens_total",
            "Tokens reported or estimated at the AgentTrace client.",
            ("component", "model", "direction"),
            registry=selected_registry,
        )
        self.tool_calls = Counter(
            "agenttrace_tool_calls_total",
            "Coding-agent tool calls by bounded tool name and status.",
            ("tool", "status"),
            registry=selected_registry,
        )
        self.tool_duration = Histogram(
            "agenttrace_tool_duration_seconds",
            "Coding-agent tool execution duration.",
            ("tool",),
            registry=selected_registry,
        )

    @contextmanager
    def track_request(self, component: str, model: str) -> Iterator[None]:
        labels = self.active.labels(component=component, model=model)
        labels.inc()
        try:
            yield
        finally:
            labels.dec()

    def observe_request(
        self,
        *,
        component: str,
        model: str,
        status: str,
        latency_seconds: float,
        ttft_seconds: float | None,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.requests.labels(component=component, model=model, status=status).inc()
        self.latency.labels(component=component, model=model).observe(latency_seconds)
        if ttft_seconds is not None:
            self.ttft.labels(component=component, model=model).observe(ttft_seconds)
        self.tokens.labels(component=component, model=model, direction="input").inc(input_tokens)
        self.tokens.labels(component=component, model=model, direction="output").inc(output_tokens)

    def observe_tool(self, *, tool: str, status: str, duration_seconds: float) -> None:
        bounded_tool = (
            tool
            if tool
            in {
                "read_file",
                "list_directory",
                "search",
                "write_file",
                "replace_text",
                "run_command",
            }
            else "other"
        )
        self.tool_calls.labels(tool=bounded_tool, status=status).inc()
        self.tool_duration.labels(tool=bounded_tool).observe(duration_seconds)


METRICS = ApplicationMetrics()


def serve_metrics(port: int = 9464, address: str = "127.0.0.1") -> None:
    start_http_server(port, addr=address)
