"""Best-effort client-process device metadata.

These values describe the AgentTrace Python process and visible CUDA devices. They
must not be interpreted as vLLM server-process utilization when the server is a
different process or host.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceTelemetry:
    scope: str
    torch_available: bool
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    devices: list[dict[str, Any]]
    limitation: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_device_telemetry() -> DeviceTelemetry:
    limitation = (
        "client-process visibility only; this does not expose inference-server "
        "GPU memory or utilization for a separate vLLM process"
    )
    try:
        import torch
    except ImportError:
        return DeviceTelemetry(
            scope="agenttrace_client_process",
            torch_available=False,
            torch_version=None,
            cuda_available=False,
            cuda_version=None,
            devices=[],
            limitation=limitation,
        )
    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "client_allocated_bytes": torch.cuda.memory_allocated(index),
                    "client_reserved_bytes": torch.cuda.memory_reserved(index),
                }
            )
    return DeviceTelemetry(
        scope="agenttrace_client_process",
        torch_available=True,
        torch_version=importlib.metadata.version("torch"),
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda,
        devices=devices,
        limitation=limitation,
    )
