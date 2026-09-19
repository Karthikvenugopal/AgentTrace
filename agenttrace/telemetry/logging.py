"""Structured logging with trace and workload correlation identifiers."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace

_correlation: ContextVar[dict[str, str] | None] = ContextVar("agenttrace_correlation", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        span = trace.get_current_span().get_span_context()
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
            **(_correlation.get() or {}),
        }
        if span.is_valid:
            payload["otel_trace_id"] = format(span.trace_id, "032x")
            payload["otel_span_id"] = format(span.span_id, "016x")
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


@contextmanager
def bind_correlation(**identifiers: str | None) -> Iterator[None]:
    bounded = {key: value for key, value in identifiers.items() if value is not None}
    token = _correlation.set({**(_correlation.get() or {}), **bounded})
    try:
        yield
    finally:
        _correlation.reset(token)
