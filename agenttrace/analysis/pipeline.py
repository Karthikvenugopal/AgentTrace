"""Workload analysis and chart generation from recorded observations."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agenttrace.benchmarking.aggregate import aggregate_experiment, distribution
from agenttrace.tracing.schema import AgentRecord, OutcomeRecord, RequestRecord, ToolCallRecord
from agenttrace.tracing.storage import iter_trace


def analyze_trace(path: Path) -> dict[str, Any]:
    records = list(iter_trace(path))
    requests = [record for record in records if isinstance(record, RequestRecord)]
    tools = [record for record in records if isinstance(record, ToolCallRecord)]
    agents = [record for record in records if isinstance(record, AgentRecord)]
    outcomes = [record for record in records if isinstance(record, OutcomeRecord)]
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
        "maximum_active_agent_concurrency": _maximum_active_agents(agents, outcomes),
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
    observations = _read_observations(experiment_dir / "observations.jsonl")
    charts_dir = experiment_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    chart_results = {
        "p95_ttft_vs_context": _plot_case_metric(
            aggregates,
            x_key="context_tokens",
            y_path=("ttft_seconds", "p95"),
            x_label="Prompt context (content tokens)",
            y_label="p95 client-observed TTFT (seconds)",
            output=charts_dir / "p95-ttft-vs-context.png",
        ),
        "p95_latency_vs_concurrency": _plot_case_metric(
            aggregates,
            x_key="concurrent_agents",
            y_path=("latency_seconds", "p95"),
            x_label="Configured concurrent agents",
            y_label="p95 client end-to-end latency (seconds)",
            output=charts_dir / "p95-latency-vs-concurrency.png",
        ),
        "output_throughput_vs_concurrency": _plot_case_metric(
            aggregates,
            x_key="concurrent_agents",
            y_path=("output_token_throughput_per_second",),
            x_label="Configured concurrent agents",
            y_label="Aggregate output-token throughput (tokens/second)",
            output=charts_dir / "output-throughput-vs-concurrency.png",
        ),
        "tool_wait_vs_server_utilization": _plot_tool_wait_utilization(
            observations,
            charts_dir / "tool-wait-vs-server-utilization.png",
            aggregates["experiment_id"],
        ),
        "subagent_activity_vs_latency": _plot_subagent_case_latency(
            aggregates,
            charts_dir / "subagent-activity-vs-latency.png",
        ),
    }
    if source_trace:
        chart_results["execution_length_vs_cumulative_prompt_tokens"] = _plot_execution_growth(
            source_trace,
            charts_dir / "execution-length-vs-cumulative-prompt-tokens.png",
            aggregates["experiment_id"],
        )
        chart_results["subagent_arrivals_vs_latency"] = _plot_subagent_arrivals(
            source_trace,
            charts_dir / "subagent-arrivals-vs-latency.png",
            aggregates["experiment_id"],
        )
    unavailable = {
        "request_level_server_queueing": (
            "vLLM Prometheus queue histograms are aggregate and cannot be assigned exactly to "
            "individual requests without backend-supported correlation"
        )
    }
    if chart_results["tool_wait_vs_server_utilization"].startswith("unavailable"):
        unavailable["tool_wait_vs_server_utilization"] = (
            "requires at least two tool-wait settings and aligned vLLM GPU KV-cache samples; "
            "the current experiment did not provide both"
        )
    result = {
        "experiment_id": aggregates["experiment_id"],
        "measured_aggregate_results": aggregates,
        "source_trace_analysis": trace_analysis,
        "charts": chart_results,
        "comparisons": _benchmark_comparisons(aggregates),
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
        if x_key == "context_tokens":
            label = (
                f"concurrency={config['concurrent_agents']} / "
                f"{config['replay_mode']} / {config['pattern']}"
            )
        else:
            label = (
                f"context={config['context_tokens']} / "
                f"{config['replay_mode']} / {config['pattern']}"
            )
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


def _plot_tool_wait_utilization(
    observations: list[dict[str, Any]], output: Path, experiment_id: str
) -> str:
    points: list[tuple[float, float]] = []
    for observation in observations:
        wait = (observation.get("effective_replay") or {}).get("tool_wait_seconds")
        if wait is None:
            continue
        values: list[float] = []
        for sample in observation.get("server_metrics_samples", []):
            metric = (sample or {}).get("available", {}).get("gpu_kv_cache_usage", {})
            values.extend(float(value) for value in metric.values())
        if values:
            points.append((float(wait), sum(values) / len(values)))
    if len({point[0] for point in points}) < 2:
        return "unavailable: requires two tool-wait settings and sampled GPU KV-cache usage"
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter([point[0] for point in points], [point[1] for point in points])
    axis.set_xlabel("Configured tool-wait duration (seconds)")
    axis.set_ylabel("Mean sampled server GPU KV-cache utilization (fraction)")
    axis.set_title(f"Tool waits and server utilization: {experiment_id}")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return str(output)


def _plot_subagent_case_latency(aggregates: dict[str, Any], output: Path) -> str:
    points = [
        (
            int(case["configuration"]["concurrent_agents"]),
            float(case["latency_seconds"]["median"]),
            case["configuration"]["replay_mode"],
        )
        for case in aggregates["cases"]
        if case["configuration"]["pattern"] == "parent_subagents"
        and case["latency_seconds"]["median"] is not None
    ]
    if not points:
        return "unavailable: benchmark has no parent-subagent cases"
    figure, axis = plt.subplots(figsize=(8, 5))
    for mode in sorted({point[2] for point in points}):
        selected = sorted(point for point in points if point[2] == mode)
        axis.plot(
            [point[0] for point in selected],
            [point[1] for point in selected],
            "o-",
            label=mode,
        )
    axis.set_xlabel("Active parent and subagent count")
    axis.set_ylabel("Median client end-to-end latency (seconds)")
    axis.set_title(f"Concurrent subagents: {aggregates['experiment_id']}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return str(output)


def _read_observations(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def _maximum_active_agents(agents: list[AgentRecord], outcomes: list[OutcomeRecord]) -> int:
    completed = {outcome.agent_id: outcome.completed_at for outcome in outcomes}
    events: list[tuple[datetime, int]] = []
    for agent in agents:
        events.append((agent.started_at, 1))
        if agent.agent_id in completed:
            events.append((completed[agent.agent_id], -1))
    # End events sort before start events at the same timestamp.
    events.sort(key=lambda event: (event[0], event[1]))
    active = maximum = 0
    for _, delta in events:
        active += delta
        maximum = max(maximum, active)
    return maximum


def _plot_execution_growth(path: Path, output: Path, experiment_id: str) -> str:
    requests = [record for record in iter_trace(path) if isinstance(record, RequestRecord)]
    streams: dict[str, list[RequestRecord]] = defaultdict(list)
    for request in requests:
        streams[request.agent_id].append(request)
    if not streams:
        return "unavailable: source trace has no requests"
    figure, axis = plt.subplots(figsize=(8, 5))
    for agent_id, stream in sorted(streams.items()):
        stream.sort(key=lambda request: request.sequence_number)
        cumulative = _cumulative([request.input_tokens for request in stream])
        axis.plot(range(1, len(stream) + 1), cumulative, "o-", label=agent_id)
    axis.set_xlabel("Agent execution step (request count)")
    axis.set_ylabel("Cumulative prompt tokens consumed")
    axis.set_title(f"AgentTrace experiment: {experiment_id}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return str(output)


def _plot_subagent_arrivals(path: Path, output: Path, experiment_id: str) -> str:
    records = list(iter_trace(path))
    agents = {record.agent_id: record for record in records if isinstance(record, AgentRecord)}
    requests = [record for record in records if isinstance(record, RequestRecord)]
    if not requests or not any(agent.parent_agent_id for agent in agents.values()):
        return "unavailable: source trace has no parent-child agent activity"
    origin = min(request.submitted_at for request in requests)
    figure, axis = plt.subplots(figsize=(8, 5))
    for relation, marker in (("parent", "o"), ("subagent", "^")):
        selected = [
            request
            for request in requests
            if (agents[request.agent_id].parent_agent_id is not None) == (relation == "subagent")
        ]
        axis.scatter(
            [(request.submitted_at - origin).total_seconds() for request in selected],
            [request.elapsed_seconds for request in selected],
            label=relation,
            marker=marker,
        )
    axis.set_xlabel("Request arrival offset (seconds)")
    axis.set_ylabel("Client end-to-end latency (seconds)")
    axis.set_title(f"Subagent arrival pattern: {experiment_id}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return str(output)


def _benchmark_comparisons(aggregates: dict[str, Any]) -> dict[str, Any]:
    cases = _preferred_study_cases(aggregates["cases"])
    context_cases = [
        case
        for case in cases
        if case["configuration"]["concurrent_agents"] == 1
        and _stable(case)
        and case["ttft_seconds"]["p95"] is not None
        and case["latency_seconds"]["p95"] is not None
    ]
    context_scaling: dict[str, Any] | None = None
    if len(context_cases) >= 2:
        smallest = min(context_cases, key=lambda case: case["configuration"]["context_tokens"])
        largest = max(context_cases, key=lambda case: case["configuration"]["context_tokens"])
        context_scaling = {
            "baseline": _comparison_values(smallest),
            "comparison": _comparison_values(largest),
            "p95_ttft_percent_change": _percent_change(
                smallest["ttft_seconds"]["p95"], largest["ttft_seconds"]["p95"]
            ),
            "p95_latency_percent_change": _percent_change(
                smallest["latency_seconds"]["p95"], largest["latency_seconds"]["p95"]
            ),
            "output_throughput_percent_change": _percent_change(
                smallest["output_token_throughput_per_second"],
                largest["output_token_throughput_per_second"],
            ),
        }

    available_contexts = sorted({case["configuration"]["context_tokens"] for case in cases})
    representative = (
        min(available_contexts, key=lambda value: abs(value - 8192)) if available_contexts else None
    )
    concurrency_scaling: dict[str, Any] | None = None
    if representative is not None:
        candidates = [
            case
            for case in cases
            if case["configuration"]["context_tokens"] == representative
            and _stable(case)
            and case["ttft_seconds"]["p95"] is not None
            and case["latency_seconds"]["p95"] is not None
        ]
        baseline = next(
            (case for case in candidates if case["configuration"]["concurrent_agents"] == 1),
            None,
        )
        highest = max(
            candidates,
            key=lambda case: case["configuration"]["concurrent_agents"],
            default=None,
        )
        if baseline is not None and highest is not None and highest is not baseline:
            concurrency_scaling = {
                "representative_context_tokens": representative,
                "baseline": _comparison_values(baseline),
                "comparison": _comparison_values(highest),
                "output_throughput_percent_change": _percent_change(
                    baseline["output_token_throughput_per_second"],
                    highest["output_token_throughput_per_second"],
                ),
                "p95_ttft_percent_change": _percent_change(
                    baseline["ttft_seconds"]["p95"], highest["ttft_seconds"]["p95"]
                ),
                "p95_latency_percent_change": _percent_change(
                    baseline["latency_seconds"]["p95"],
                    highest["latency_seconds"]["p95"],
                ),
            }

    stable = [case for case in cases if _stable(case)]
    best = max(
        stable,
        key=lambda case: case["output_token_throughput_per_second"] or float("-inf"),
        default=None,
    )
    tradeoffs: list[dict[str, Any]] = []
    for context in available_contexts:
        same_context = [
            case
            for case in stable
            if case["configuration"]["context_tokens"] == context
            and case["latency_seconds"]["p95"] is not None
        ]
        baseline = next(
            (case for case in same_context if case["configuration"]["concurrent_agents"] == 1),
            None,
        )
        if baseline is None:
            continue
        for candidate in same_context:
            throughput_change = _percent_change(
                baseline["output_token_throughput_per_second"],
                candidate["output_token_throughput_per_second"],
            )
            latency_change = _percent_change(
                baseline["latency_seconds"]["p95"], candidate["latency_seconds"]["p95"]
            )
            if (throughput_change or 0) > 0 and (latency_change or 0) > 0:
                tradeoffs.append(
                    {
                        "baseline": _comparison_values(baseline),
                        "comparison": _comparison_values(candidate),
                        "output_throughput_percent_change": throughput_change,
                        "p95_latency_percent_change": latency_change,
                    }
                )
    largest_tradeoff = max(
        tradeoffs,
        key=lambda comparison: comparison["p95_latency_percent_change"] or float("-inf"),
        default=None,
    )
    return {
        "context_scaling_at_concurrency_1": context_scaling,
        "concurrency_scaling_at_representative_context": concurrency_scaling,
        "best_stable_output_throughput": _comparison_values(best) if best else None,
        "largest_observed_latency_tradeoff": largest_tradeoff,
        "stable_definition": "zero failed logical requests across all repetitions",
    }


def _preferred_study_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferred = [
        case
        for case in cases
        if case["configuration"]["replay_mode"] == "open_loop"
        and case["configuration"]["pattern"] == "concurrent"
        and case["configuration"]["workload_type"] == "parameterized"
    ]
    return preferred or cases


def _comparison_values(case: dict[str, Any] | None) -> dict[str, Any]:
    if case is None:
        return {}
    config = case["configuration"]
    return {
        "case_id": case["case_id"],
        "context_tokens": config["context_tokens"],
        "concurrent_agents": config["concurrent_agents"],
        "successful_requests": case["successful_requests"],
        "failed_requests": case["failed_requests"],
        "failure_rate": case["failure_rate"],
        "p95_ttft_seconds": case["ttft_seconds"]["p95"],
        "p95_latency_seconds": case["latency_seconds"]["p95"],
        "output_tokens_per_second": case["output_token_throughput_per_second"],
    }


def _stable(case: dict[str, Any]) -> bool:
    return bool(case["successful_requests"] > 0 and case["failed_requests"] == 0)


def _percent_change(baseline: Any, comparison: Any) -> float | None:
    if baseline is None or comparison is None or float(baseline) == 0:
        return None
    return (float(comparison) - float(baseline)) / float(baseline) * 100.0
