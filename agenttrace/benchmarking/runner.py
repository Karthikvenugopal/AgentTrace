"""Reproducible benchmark matrix execution over the replay engine."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from agenttrace.benchmarking.matrix import BenchmarkCase, ordered_cases
from agenttrace.benchmarking.provenance import capture_provenance
from agenttrace.config import BenchmarkConfig, EndpointConfig, ReplayConfig
from agenttrace.instrumentation.tokens import TokenCounter, WhitespaceTokenCounter
from agenttrace.replay.engine import ReplayEngine
from agenttrace.replay.loader import load_workload
from agenttrace.replay.models import ReplaySessionResult, WorkloadRequest
from agenttrace.replay.transform import transform_workload
from agenttrace.serving.client import InferenceClient
from agenttrace.serving.metrics import ServerMetricsSnapshot, VLLMMetricsAdapter
from agenttrace.telemetry.tracing import trace_span

ClientFactory = Callable[[EndpointConfig], InferenceClient]


class BenchmarkRunner:
    def __init__(
        self,
        config: BenchmarkConfig,
        client_factory: ClientFactory,
        *,
        token_counter: TokenCounter | None = None,
        server_metrics: VLLMMetricsAdapter | None = None,
    ) -> None:
        self.config = config
        self.client_factory = client_factory
        self.counter = token_counter or WhitespaceTokenCounter()
        self.server_metrics = server_metrics
        self.experiment_dir = config.output_dir / config.experiment_id

    async def run(self) -> Path:
        source = load_workload(self.config.source_trace)
        if not source:
            raise ValueError("benchmark source trace has no inference requests")
        preflight = await self._preflight()
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now(UTC)
        self._write_metadata(started_at, preflight)
        observations = self.experiment_dir / "observations.jsonl"
        observations.write_text("", encoding="utf-8")
        with trace_span(
            "benchmark.experiment", {"agenttrace.experiment_id": self.config.experiment_id}
        ):
            for repetition in range(self.config.repetitions):
                for case in ordered_cases(self.config, repetition):
                    result, before, after, samples, metric_errors, replay = await self._run_case(
                        case, repetition, source
                    )
                    self._append_observation(
                        observations,
                        case,
                        repetition,
                        result,
                        before,
                        after,
                        samples,
                        metric_errors,
                        replay,
                    )
        completion = {
            "completed_at": datetime.now(UTC).isoformat(),
            "started_at": started_at.isoformat(),
            "experiment_id": self.config.experiment_id,
        }
        (self.experiment_dir / "completion.json").write_text(
            json.dumps(completion, indent=2) + "\n", encoding="utf-8"
        )
        return self.experiment_dir

    async def _run_case(
        self,
        case: BenchmarkCase,
        repetition: int,
        source: list[WorkloadRequest],
    ) -> tuple[
        ReplaySessionResult,
        ServerMetricsSnapshot | None,
        ServerMetricsSnapshot | None,
        list[ServerMetricsSnapshot],
        list[str],
        ReplayConfig,
    ]:
        replay = self.config.replay.model_copy(
            update={
                "mode": case.replay_mode,
                "prompt_tokens": case.context_tokens,
                "concurrent_agents": case.concurrent_agents,
                "active_subagents": (
                    max(0, case.concurrent_agents - 1) if case.pattern == "parent_subagents" else 0
                ),
                "seed": self.config.seed + repetition,
            }
        )
        workload = source
        if case.workload_type == "parameterized":
            workload = transform_workload(source, replay, self.counter)
        client = self.client_factory(replay.endpoint)
        engine = ReplayEngine(replay, client)
        samples: list[ServerMetricsSnapshot] = []
        metric_errors: list[str] = []
        stop_sampling = asyncio.Event()

        async def sample_metrics() -> None:
            if self.server_metrics is None:
                return
            while not stop_sampling.is_set():
                try:
                    samples.append(await self.server_metrics.collect())
                except Exception as exc:
                    metric_errors.append(f"{type(exc).__name__}: {exc}")
                try:
                    await asyncio.wait_for(
                        stop_sampling.wait(),
                        timeout=self.config.server_metrics_interval_seconds,
                    )
                except TimeoutError:
                    pass

        try:
            # Warm each configuration once. Re-warming every repetition would add a
            # configuration-dependent amount of unmeasured server work.
            if repetition == 0 and self.config.warmup_requests:
                warmup = workload[: self.config.warmup_requests]
                if warmup:
                    await engine.run_closed_loop(warmup)
            before = await self._safe_server_snapshot(metric_errors)
            sampling_task = asyncio.create_task(sample_metrics())
            try:
                if case.replay_mode == "open_loop":
                    result = await engine.run_open_loop(workload)
                else:
                    result = await engine.run_closed_loop(workload)
            finally:
                stop_sampling.set()
                await sampling_task
            after = await self._safe_server_snapshot(metric_errors)
            return result, before, after, samples, metric_errors, replay
        finally:
            await client.close()

    async def _safe_server_snapshot(self, errors: list[str]) -> ServerMetricsSnapshot | None:
        if self.server_metrics is None:
            return None
        try:
            return await self.server_metrics.collect()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            return None

    async def _preflight(self) -> dict[str, Any]:
        if self.config.measurement_label != "real_inference":
            return {"classification": "mock", "vllm_metrics_verified": False}
        if self.server_metrics is None:
            raise ValueError(
                "real_inference requires AGENTTRACE_VLLM_METRICS_URL so the target can be "
                "verified as vLLM"
            )
        if self.config.replay.tokenizer is None:
            raise ValueError("real_inference requires an exact model tokenizer")
        snapshot = await self.server_metrics.collect()
        vllm_names = [
            name
            for name in snapshot.raw_metric_names
            if name.startswith("vllm:") or name.startswith("vllm_")
        ]
        if not vllm_names:
            raise ValueError(
                "real_inference target did not expose vLLM metric families; refusing to label "
                "the run as vLLM inference"
            )
        base_url = str(self.config.replay.endpoint.base_url).rstrip("/")
        headers: dict[str, str] = {}
        api_key = self.config.replay.endpoint.api_key
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"
        async with httpx.AsyncClient(
            timeout=self.config.replay.endpoint.timeout_seconds, headers=headers
        ) as client:
            response = await client.get(f"{base_url}/models")
            response.raise_for_status()
            body = response.json()
        advertised = sorted(
            str(item.get("id"))
            for item in body.get("data", [])
            if isinstance(item, dict) and item.get("id")
        )
        configured_model = self.config.replay.endpoint.model
        if configured_model not in advertised:
            raise ValueError(
                f"configured model {configured_model!r} is not advertised by the endpoint: "
                f"{advertised}"
            )
        return {
            "classification": "real_vllm_inference",
            "models_endpoint": f"{base_url}/models",
            "configured_served_model": configured_model,
            "advertised_models": advertised,
            "metrics_endpoint": snapshot.endpoint,
            "vllm_metrics_verified": True,
            "vllm_metric_family_count": len(vllm_names),
        }

    def _write_metadata(self, started_at: datetime, preflight: dict[str, Any]) -> None:
        config = self.config.model_dump(mode="json", exclude={"replay": {"endpoint": {"api_key"}}})
        metadata = {
            "experiment_id": self.config.experiment_id,
            "started_at": started_at.isoformat(),
            "provenance": capture_provenance(),
            "tokenizer": self.counter.identity,
            "token_count_method": self.counter.method,
            "configuration": config,
            "server_configuration": self.config.server_metadata,
            "preflight": preflight,
            "warmups_excluded_from_measurements": True,
        }
        (self.experiment_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _snapshot(snapshot: ServerMetricsSnapshot | None) -> dict[str, Any] | None:
        if snapshot is None:
            return None
        return {
            "collected_at": snapshot.collected_at.isoformat(),
            "endpoint": snapshot.endpoint,
            "available": snapshot.available,
            "unavailable": snapshot.unavailable,
            "raw_metric_names": snapshot.raw_metric_names,
        }

    def _append_observation(
        self,
        path: Path,
        case: BenchmarkCase,
        repetition: int,
        result: ReplaySessionResult,
        before: ServerMetricsSnapshot | None,
        after: ServerMetricsSnapshot | None,
        samples: list[ServerMetricsSnapshot],
        metric_errors: list[str],
        replay_config: ReplayConfig,
    ) -> None:
        record = {
            "record_type": "benchmark_trial",
            "experiment_id": self.config.experiment_id,
            "case": case.model_dump(mode="json"),
            "repetition": repetition,
            "replay": result.model_dump(mode="json"),
            "server_metrics_before": self._snapshot(before),
            "server_metrics_after": self._snapshot(after),
            "server_metrics_samples": [self._snapshot(sample) for sample in samples],
            "server_metrics_errors": metric_errors,
            "effective_replay": replay_config.model_dump(
                mode="json", exclude={"endpoint": {"api_key"}}
            ),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
