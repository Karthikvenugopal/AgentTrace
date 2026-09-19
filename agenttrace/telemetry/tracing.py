"""OpenTelemetry setup and correlation-safe span helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_configured = False


def configure_tracing(
    *, service_name: str = "agenttrace", otlp_http_endpoint: str | None = None
) -> TracerProvider:
    """Configure the SDK once; OTLP export is optional for local tests."""

    global _configured
    current = trace.get_tracer_provider()
    if _configured and isinstance(current, TracerProvider):
        return current
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if otlp_http_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as exc:
            raise RuntimeError("install agenttrace[telemetry] to export OTLP traces") from exc
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_http_endpoint))
        )
    trace.set_tracer_provider(provider)
    _configured = True
    return provider


def tracer() -> trace.Tracer:
    return trace.get_tracer("agenttrace", "0.1.0")


@contextmanager
def trace_span(name: str, attributes: Mapping[str, Any] | None = None) -> Iterator[trace.Span]:
    """Start a span and attach non-null identifiers and bounded attributes."""

    with tracer().start_as_current_span(name) as span:
        for key, value in (attributes or {}).items():
            if value is not None and isinstance(value, (str, bool, int, float)):
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
            raise
