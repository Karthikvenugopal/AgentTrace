"""Version-aware adapter for vLLM's aggregate Prometheus metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from prometheus_client.parser import text_string_to_metric_families


@dataclass(frozen=True)
class MetricDefinition:
    semantic_name: str
    candidates: tuple[str, ...]
    semantics: str
    request_level: bool = False


DEFINITIONS = (
    MetricDefinition(
        "running_requests",
        ("vllm:num_requests_running", "vllm_num_requests_running"),
        "Current requests executing in the vLLM engine.",
    ),
    MetricDefinition(
        "waiting_requests",
        ("vllm:num_requests_waiting", "vllm_num_requests_waiting"),
        "Current requests waiting in the vLLM scheduler.",
    ),
    MetricDefinition(
        "gpu_kv_cache_usage",
        ("vllm:gpu_cache_usage_perc", "vllm_gpu_cache_usage_perc"),
        "Aggregate fraction of GPU KV-cache blocks in use.",
    ),
    MetricDefinition(
        "prompt_tokens_total",
        ("vllm:prompt_tokens_total", "vllm_prompt_tokens_total"),
        "Cumulative prompt tokens processed server-side.",
    ),
    MetricDefinition(
        "generation_tokens_total",
        ("vllm:generation_tokens_total", "vllm_generation_tokens_total"),
        "Cumulative generated tokens server-side.",
    ),
    MetricDefinition(
        "queue_time_seconds",
        ("vllm:request_queue_time_seconds", "vllm_request_queue_time_seconds"),
        "Aggregate histogram of time spent queued; not an exact request correlation.",
    ),
    MetricDefinition(
        "prefill_time_seconds",
        ("vllm:request_prefill_time_seconds", "vllm_request_prefill_time_seconds"),
        "Aggregate prefill duration when exposed by this vLLM release.",
    ),
    MetricDefinition(
        "decode_time_seconds",
        ("vllm:request_decode_time_seconds", "vllm_request_decode_time_seconds"),
        "Aggregate decode duration when exposed by this vLLM release.",
    ),
)


@dataclass(frozen=True)
class ServerMetricsSnapshot:
    collected_at: datetime
    endpoint: str
    available: dict[str, dict[str, float]]
    unavailable: dict[str, str]
    raw_metric_names: tuple[str, ...]


class VLLMMetricsAdapter:
    def __init__(self, metrics_url: str, *, timeout_seconds: float = 5.0) -> None:
        self.metrics_url = metrics_url
        self.timeout_seconds = timeout_seconds

    async def collect(self) -> ServerMetricsSnapshot:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(self.metrics_url)
            response.raise_for_status()
        samples: dict[str, float] = {}
        raw_names: set[str] = set()
        for family in text_string_to_metric_families(response.text):
            raw_names.add(family.name)
            for sample in family.samples:
                raw_names.add(sample.name)
                key = sample.name
                # Multiple workers/models are aggregated; cardinal labels are kept out.
                samples[key] = samples.get(key, 0.0) + float(sample.value)
        available: dict[str, dict[str, float]] = {}
        unavailable: dict[str, str] = {}
        for definition in DEFINITIONS:
            values: dict[str, float] = {}
            for candidate in definition.candidates:
                for name, value in samples.items():
                    if name == candidate or name.startswith(candidate + "_"):
                        values[name] = value
            if values:
                available[definition.semantic_name] = values
            else:
                unavailable[definition.semantic_name] = (
                    "not exported by the connected vLLM metrics endpoint"
                )
        return ServerMetricsSnapshot(
            collected_at=datetime.now(UTC),
            endpoint=self.metrics_url,
            available=available,
            unavailable=unavailable,
            raw_metric_names=tuple(sorted(raw_names)),
        )


def metric_semantics() -> dict[str, str]:
    return {definition.semantic_name: definition.semantics for definition in DEFINITIONS}
