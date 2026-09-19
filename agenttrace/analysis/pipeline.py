"""Workload analysis and chart generation from recorded observations."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agenttrace.benchmarking.aggregate import aggregate_experiment, distribution
from agenttrace.tracing.schema import AgentRecord, RequestRecord, ToolCallRecord
from agenttrace.tracing.storage import iter_trace


def analyze_trace(path: Path) -> dict[str, Any]:
    records = list(iter_trace(path))
    requests = [record for record in records if isinstance(record, RequestRecord)]
    tools = [record for record in records if isinstance(record, ToolCallRecord)]
    agents = [record for record in records if isinstance(record, AgentRecord)]
    concurrency = _concurrency_at_submissions(requests)
    per_agent: dict[str, dict[str, Any]] = {}
    for agent_id in sorted({request.agent_id for request in requests}):
        stream = sorted(
            (request for request in requests if request.agent_id == agent_id),
            key=lambda request: request.sequence_number,
        )
        per_agent[agent_id] = {
            "steps": len(stream),
            "prompt_tokens_by_step": [request.input_tokens for request in stream],
            "context_growth_tokens_by_step": [request.context_growth_tokens for request in stream],
            "cumulative_prompt_tokens": _cumulative([request.input_tokens for request in stream]),
            "total_task_tokens": sum(
                request.input_tokens + request.output_tokens for request in stream
            ),
        }
    return {
        "request_count": len(requests),
        "agent_count": len(agents),
        "subagent_count": sum(agent.parent_agent_id is not None for agent in agents),
        "ttft_seconds": distribution(
            [request.ttft_seconds for request in requests if request.ttft_seconds is not None]
        ),
        "latency_seconds": distribution([request.elapsed_seconds for request in requests]),
        "inter_chunk_arrival_seconds": distribution(_chunk_intervals(requests)),
        "tool_duration_seconds": distribution([tool.duration_seconds for tool in tools]),
        "tool_output_tokens": sum(tool.output_tokens for tool in tools),
        "agent_idle_waiting_for_tools_seconds": sum(tool.duration_seconds for tool in tools),
        "maximum_inference_request_concurrency": max(concurrency, default=0),
        "per_agent": per_agent,
        "stream_timing_semantics": (
            "inter-chunk arrivals are transport observations; chunks containing multiple tokens "
            "do not reveal individual token decode latency"
        ),
    }


def analyze_experiment(experiment_dir: Path, *, source_trace: Path | None = None) -> Path:
    aggregate_path = experiment_dir / "aggregates.json"
    aggregates = (
        json.loads(aggregate_path.read_text(encoding="utf-8"))
        if aggregate_path.exists()
        else aggregate_experiment(experiment_dir)
    )
    trace_analysis = analyze_trace(source_trace) if source_trace else None
    charts_dir = experiment_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    chart_results = {
        "context_vs_ttft": _plot_case_metric(
            aggregates,
            x_key="context_tokens",
            y_path=("ttft_seconds", "median"),
            x_label="Prompt context (content tokens)",
            y_label="Median client-observed TTFT (seconds)",
            output=charts_dir / "context-vs-ttft.png",
        ),
        "concurrency_vs_latency": _plot_case_metric(
            aggregates,
            x_key="concurrent_agents",
            y_path=("latency_seconds", "median"),
            x_label="Configured concurrent agents",
            y_label="Median end-to-end client latency (seconds)",
            output=charts_dir / "concurrency-vs-latency.png",
        ),
        "concurrency_vs_generation_throughput": _plot_case_metric(
            aggregates,
            x_key="concurrent_agents",
            y_path=("output_token_throughput_per_second",),
            x_label="Configured concurrent agents",
            y_label="Aggregate output-token throughput (tokens/second)",
            output=charts_dir / "concurrency-vs-output-throughput.png",
        ),
    }
    unavailable = {
        "tool_wait_vs_server_utilization": (
            "requires aligned time-series vLLM utilization samples; before/after counters are "
            "not sufficient for a utilization curve"
        ),
        "request_level_server_queueing": (
            "vLLM Prometheus queue histograms are aggregate and cannot be assigned exactly to "
            "individual requests without backend-supported correlation"
        ),
    }
    result = {
        "experiment_id": aggregates["experiment_id"],
        "measured_aggregate_results": aggregates,
        "source_trace_analysis": trace_analysis,
        "charts": chart_results,
        "unavailable_analyses": unavailable,
        "interpretation": (
            "This file reports observations only. It does not claim causal improvements or "
            "generalize beyond the recorded configuration."
        ),
    }
    output = experiment_dir / "analysis.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _plot_case_metric(
    aggregates: dict[str, Any],
    *,
    x_key: str,
    y_path: tuple[str, ...],
    x_label: str,
    y_label: str,
    output: Path,
) -> str:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for case in aggregates["cases"]:
        value: Any = case
        for key in y_path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is None:
            continue
        config = case["configuration"]
        label = f"{config['replay_mode']} / {config['pattern']}"
        series[label].append((float(config[x_key]), float(value)))
    if not series:
        return "unavailable: no supported observations"
    figure, axis = plt.subplots(figsize=(8, 5))
    for label, points in sorted(series.items()):
        points.sort()
        axis.plot([point[0] for point in points], [point[1] for point in points], "o-", label=label)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(f"AgentTrace experiment: {aggregates['experiment_id']}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return str(output)


def _chunk_intervals(requests: list[RequestRecord]) -> list[float]:
    intervals: list[float] = []
    for request in requests:
        arrivals = request.stream_chunk_arrivals_seconds
        intervals.extend(right - left for left, right in pairwise(arrivals))
    return intervals


def _cumulative(values: list[int]) -> list[int]:
    total = 0
    result = []
    for value in values:
        total += value
        result.append(total)
    return result


def _concurrency_at_submissions(requests: list[RequestRecord]) -> list[int]:
    return [request.concurrency_at_submission for request in requests]
