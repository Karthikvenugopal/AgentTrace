"""Environment provenance captured for reproducible benchmark trials."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from datetime import UTC, datetime
from typing import Any

from agenttrace import __version__
from agenttrace.telemetry.device import collect_device_telemetry


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_provenance() -> dict[str, Any]:
    """Capture relevant versions without environment variables or credentials."""

    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "agenttrace_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "dependencies": {
            name: _version(name)
            for name in (
                "httpx",
                "pydantic",
                "prometheus-client",
                "opentelemetry-sdk",
                "transformers",
                "torch",
                "vllm",
            )
        },
        "device": collect_device_telemetry().as_dict(),
    }
