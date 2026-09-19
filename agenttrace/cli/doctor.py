"""Actionable environment and endpoint diagnostics."""

from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import asdict, dataclass

import httpx

from agenttrace.config import EndpointConfig
from agenttrace.serving.vllm import installed_vllm_version
from agenttrace.telemetry.device import collect_device_telemetry


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    action: str | None = None


async def run_doctor(endpoint: EndpointConfig | None = None) -> list[Check]:
    checks = [
        Check("python", "ok", platform.python_version()),
        _dependency("torch", optional=True, extra="gpu"),
        _dependency("vllm", optional=True, extra="gpu"),
        _dependency("transformers", optional=True, extra="tokenizers"),
    ]
    device = collect_device_telemetry()
    checks.append(
        Check(
            "cuda",
            "ok" if device.cuda_available else "unavailable",
            f"visible devices: {len(device.devices)}; {device.limitation}",
            None if device.cuda_available else "GPU benchmarks require CUDA; CPU tests do not.",
        )
    )
    otlp = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    checks.append(
        Check(
            "otel_export",
            "ok" if otlp else "not_configured",
            otlp or "OTEL_EXPORTER_OTLP_ENDPOINT is unset",
            "Set it to http://127.0.0.1:4318 after starting docker compose." if not otlp else None,
        )
    )
    if endpoint:
        models_url = str(endpoint.base_url).rstrip("/") + "/models"
        headers = {}
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key.get_secret_value()}"
        try:
            async with httpx.AsyncClient(timeout=min(5.0, endpoint.timeout_seconds)) as client:
                response = await client.get(models_url, headers=headers)
                response.raise_for_status()
                models = [item.get("id") for item in response.json().get("data", [])]
            status = "ok" if endpoint.model in models or not models else "warning"
            detail = f"reachable; advertised models: {models or 'not reported'}"
            action = None if status == "ok" else f"Configure one of {models}."
            checks.append(Check("inference_endpoint", status, detail, action))
        except Exception as exc:
            checks.append(
                Check(
                    "inference_endpoint",
                    "failed",
                    str(exc),
                    f"Start the server and verify {models_url}.",
                )
            )
    version = installed_vllm_version()
    checks.append(
        Check(
            "vllm_version",
            "ok" if version else "unavailable",
            version or "vLLM is not installed in this environment",
            None if version else "Install agenttrace[gpu] only on a supported GPU host.",
        )
    )
    return checks


def serialize_checks(checks: list[Check]) -> list[dict[str, str | None]]:
    return [asdict(check) for check in checks]


def _dependency(name: str, *, optional: bool, extra: str) -> Check:
    available = importlib.util.find_spec(name) is not None
    return Check(
        f"dependency:{name}",
        "ok" if available else ("optional_missing" if optional else "failed"),
        "installed" if available else "not installed",
        None if available else f"Install with: pip install 'agenttrace[{extra}]'",
    )
