"""Aggregate benchmark trials without hiding failed or retried attempts."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    """Only report tail percentiles when sample sizes make them interpretable."""

    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95) if len(values) >= 20 else None,
        "p99": percentile(values, 0.99) if len(values) >= 100 else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def aggregate_experiment(experiment_dir: Path) -> dict[str, Any]:
    """Aggregate measured attempts and emit JSON plus request/configuration CSV files."""

    observation_path = experiment_dir / "observations.jsonl"
    records = [
        json.loads(line)
        for line in observation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["case"]["case_id"]].append(record)

    cases: list[dict[str, Any]] = []
    for case_id, trials in groups.items():
        attempts = [attempt for trial in trials for attempt in trial["replay"]["attempts"]]
        successes = [attempt for attempt in attempts if attempt["status"] == "succeeded"]
        latencies = [float(attempt["latency_seconds"]) for attempt in successes]
        ttfts = [
            float(attempt["ttft_seconds"])
            for attempt in successes
            if attempt["ttft_seconds"] is not None
        ]
        input_values = [int(attempt["input_tokens"] or 0) for attempt in successes]
        output_values = [int(attempt["output_tokens"] or 0) for attempt in successes]
        input_tokens = sum(input_values)
        output_tokens = sum(output_values)
        input_methods = sorted(
            {str(attempt.get("input_token_method") or "unspecified") for attempt in successes}
        )
        output_methods = sorted(
            {str(attempt.get("output_token_method") or "unspecified") for attempt in successes}
        )
        total_throughput_valid = (
            bool(successes)
            and input_methods == ["server_usage"]
            and output_methods == ["server_usage"]
        )
        trial_durations = [_measured_request_window(trial) for trial in trials]
        total_duration = sum(trial_durations)
        logical_requests = _logical_request_attempts(attempts)
        failed_requests = sum(
            not any(attempt["status"] == "succeeded" for attempt in request_attempts)
            for request_attempts in logical_requests.values()
        )
        retried_requests = sum(
            max(int(attempt["attempt_number"]) for attempt in request_attempts) > 1
            for request_attempts in logical_requests.values()
        )
        concurrency = [_trial_concurrency(trial) for trial in trials]
        total_attempt_busy_time = sum(item["attempt_busy_seconds"] for item in concurrency)
        sampled = _sampled_server_metrics(trials)
        cases.append(
            {
                "case_id": case_id,
                "configuration": trials[0]["case"],
                "trials": len(trials),
                "attempts": len(attempts),
                "logical_requests": len(logical_requests),
                "successful_requests": len(successes),
                "failed_requests": failed_requests,
                "failed_attempts": sum(a["status"] == "failed" for a in attempts),
                "timeout_attempts": sum(a["status"] == "timeout" for a in attempts),
                "retried_requests": retried_requests,
                "retried_attempts": sum(int(a["attempt_number"]) > 1 for a in attempts),
                "failure_rate": failed_requests / len(logical_requests)
                if logical_requests
                else None,
                "latency_seconds": distribution(latencies),
                "ttft_seconds": distribution(ttfts),
                "input_tokens_per_request": distribution([float(value) for value in input_values]),
                "output_tokens_per_request": distribution(
                    [float(value) for value in output_values]
                ),
                "measured_duration_seconds": total_duration,
                "request_throughput_per_second": _rate(len(successes), total_duration),
                "input_token_throughput_per_second": _rate(input_tokens, total_duration),
                "output_token_throughput_per_second": _rate(output_tokens, total_duration),
                "total_token_throughput_per_second": _rate(
                    input_tokens + output_tokens, total_duration
                )
                if total_throughput_valid
                else None,
                "total_token_throughput_valid": total_throughput_valid,
                "input_token_methods": input_methods,
                "output_token_methods": output_methods,
                "total_input_tokens": input_tokens,
                "total_output_tokens": output_tokens,
                "maximum_request_concurrency": max(
                    (int(item["maximum"]) for item in concurrency), default=0
                ),
                "effective_request_concurrency": (
                    total_attempt_busy_time / total_duration if total_duration > 0 else None
                ),
                "server_metrics": {
                    **sampled,
                    "queue_time_window": _histogram_window_delta(trials, "queue_time_seconds"),
                    "prefill_time_window": _histogram_window_delta(trials, "prefill_time_seconds"),
                    "decode_time_window": _histogram_window_delta(trials, "decode_time_seconds"),
                    "correlation_scope": (
                        "aggregate vLLM samples over each measured case window; not "
                        "request-level correlation"
                    ),
                },
                "itl": {
                    "available": False,
                    "reason": (
                        "ITL unavailable with current instrumentation: streamed chunks may "
                        "contain multiple tokens"
                    ),
                },
                "aggregation_policy": (
                    "successful attempts only for latency and throughput; all failed, timeout, "
                    "and retry attempts counted separately; throughput uses earliest measured "
                    "submission through latest measured completion per trial"
                ),
            }
        )
    result = {
        "experiment_id": experiment_dir.name,
        "case_count": len(cases),
        "cases": sorted(cases, key=lambda case: case["case_id"]),
    }
    (experiment_dir / "aggregates.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from agenttrace.benchmarking.export import write_csv_artifacts

    write_csv_artifacts(experiment_dir, records, result)
    return result


def _rate(numerator: int, duration: float) -> float | None:
    return numerator / duration if duration > 0 else None


def _logical_request_attempts(
    attempts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, attempt in enumerate(attempts):
        key = str(attempt.get("replay_request_id") or f"legacy-attempt-{index}")
        grouped[key].append(attempt)
    return grouped


def _measured_request_window(trial: dict[str, Any]) -> float:
    attempts = trial["replay"]["attempts"]
    timestamps = [
        (_parse_time(attempt["submitted_at"]), _parse_time(attempt["completed_at"]))
        for attempt in attempts
        if attempt.get("submitted_at") and attempt.get("completed_at")
    ]
    if timestamps:
        start = min(item[0] for item in timestamps)
        end = max(item[1] for item in timestamps)
        return max(0.0, (end - start).total_seconds())
    return max(
        0.0,
        (
            _parse_time(trial["replay"]["completed_at"])
            - _parse_time(trial["replay"]["started_at"])
        ).total_seconds(),
    )


def _trial_concurrency(trial: dict[str, Any]) -> dict[str, float | int]:
    events: list[tuple[datetime, int]] = []
    busy_seconds = 0.0
    for attempt in trial["replay"]["attempts"]:
        if not attempt.get("submitted_at") or not attempt.get("completed_at"):
            continue
        start = _parse_time(attempt["submitted_at"])
        end = _parse_time(attempt["completed_at"])
        events.extend(((start, 1), (end, -1)))
        busy_seconds += max(0.0, (end - start).total_seconds())
    events.sort(key=lambda item: (item[0], item[1]))
    active = maximum = 0
    for _, delta in events:
        active += delta
        maximum = max(maximum, active)
    return {"maximum": maximum, "attempt_busy_seconds": busy_seconds}


def _sampled_server_metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for semantic in ("running_requests", "waiting_requests", "gpu_kv_cache_usage"):
        values: list[float] = []
        for trial in trials:
            snapshots = trial.get("server_metrics_samples") or []
            if not snapshots:
                snapshots = [
                    trial.get("server_metrics_before"),
                    trial.get("server_metrics_after"),
                ]
            for snapshot in snapshots:
                metric = ((snapshot or {}).get("available") or {}).get(semantic) or {}
                values.extend(float(value) for value in metric.values())
        result[semantic] = {
            "sample_count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "maximum": max(values) if values else None,
        }
    return result


def _histogram_window_delta(
    trials: list[dict[str, Any]], semantic: str
) -> dict[str, float | int | str | None]:
    total_sum = 0.0
    total_count = 0.0
    available_trials = 0
    for trial in trials:
        before = ((trial.get("server_metrics_before") or {}).get("available") or {}).get(semantic)
        after = ((trial.get("server_metrics_after") or {}).get("available") or {}).get(semantic)
        if not before or not after:
            continue
        before_sum = sum(float(value) for name, value in before.items() if name.endswith("_sum"))
        after_sum = sum(float(value) for name, value in after.items() if name.endswith("_sum"))
        before_count = sum(
            float(value) for name, value in before.items() if name.endswith("_count")
        )
        after_count = sum(float(value) for name, value in after.items() if name.endswith("_count"))
        delta_count = max(0.0, after_count - before_count)
        delta_sum = max(0.0, after_sum - before_sum)
        if delta_count:
            available_trials += 1
            total_count += delta_count
            total_sum += delta_sum
    return {
        "scope": "aggregate_server_window",
        "available_trials": available_trials,
        "observation_count": int(total_count),
        "total_seconds": total_sum if available_trials else None,
        "mean_seconds": total_sum / total_count if total_count else None,
    }


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)
