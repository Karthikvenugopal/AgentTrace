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
    observation_path = experiment_dir / "observations.jsonl"
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in observation_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            groups[record["case"]["case_id"]].append(record)
    cases: list[dict[str, Any]] = []
    for case_id, trials in groups.items():
        attempts = [attempt for trial in trials for attempt in trial["replay"]["attempts"]]
        successes = [attempt for attempt in attempts if attempt["status"] == "succeeded"]
        # Only successful attempts enter latency/throughput distributions. Failures remain explicit.
        latencies = [float(attempt["latency_seconds"]) for attempt in successes]
        ttfts = [
            float(attempt["ttft_seconds"])
            for attempt in successes
            if attempt["ttft_seconds"] is not None
        ]
        input_tokens = sum(int(attempt["input_tokens"] or 0) for attempt in successes)
        output_tokens = sum(int(attempt["output_tokens"] or 0) for attempt in successes)
        trial_durations = [
            max(
                0.0,
                (
                    _parse_time(trial["replay"]["completed_at"])
                    - _parse_time(trial["replay"]["started_at"])
                ).total_seconds(),
            )
            for trial in trials
        ]
        total_duration = sum(trial_durations)
        cases.append(
            {
                "case_id": case_id,
                "configuration": trials[0]["case"],
                "trials": len(trials),
                "attempts": len(attempts),
                "successful_requests": len(successes),
                "failed_attempts": sum(a["status"] == "failed" for a in attempts),
                "timeout_attempts": sum(a["status"] == "timeout" for a in attempts),
                "retried_attempts": sum(int(a["attempt_number"]) > 1 for a in attempts),
                "latency_seconds": distribution(latencies),
                "ttft_seconds": distribution(ttfts),
                "measured_duration_seconds": total_duration,
                "request_throughput_per_second": (
                    len(successes) / total_duration if total_duration > 0 else None
                ),
                "input_token_throughput_per_second": (
                    input_tokens / total_duration if total_duration > 0 else None
                ),
                "output_token_throughput_per_second": (
                    output_tokens / total_duration if total_duration > 0 else None
                ),
                "total_input_tokens": input_tokens,
                "total_output_tokens": output_tokens,
                "aggregation_policy": (
                    "successful attempts only for latency and throughput; all failed, timeout, "
                    "and retry attempts counted separately"
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
    return result


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)
