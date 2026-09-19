import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agenttrace.benchmarking.aggregate import aggregate_experiment, distribution


def test_tail_percentiles_require_enough_samples() -> None:
    small = distribution([float(value) for value in range(10)])
    assert small["median"] == 4.5
    assert small["p95"] is None
    medium = distribution([float(value) for value in range(20)])
    assert medium["p95"] is not None
    assert medium["p99"] is None


def test_aggregation_counts_failures_but_excludes_them_from_latency(tmp_path: Path) -> None:
    start = datetime.now(UTC)
    attempts = [
        {
            "status": "failed",
            "attempt_number": 1,
            "latency_seconds": 9.0,
            "ttft_seconds": None,
            "input_tokens": None,
            "output_tokens": None,
        },
        {
            "status": "succeeded",
            "attempt_number": 2,
            "latency_seconds": 0.2,
            "ttft_seconds": 0.05,
            "input_tokens": 100,
            "output_tokens": 10,
        },
    ]
    record = {
        "case": {
            "case_id": "case-a",
            "context_tokens": 100,
            "concurrent_agents": 1,
            "pattern": "sequential",
            "workload_type": "parameterized",
            "replay_mode": "open_loop",
        },
        "replay": {
            "started_at": start.isoformat(),
            "completed_at": (start + timedelta(seconds=1)).isoformat(),
            "attempts": attempts,
        },
    }
    (tmp_path / "observations.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    result = aggregate_experiment(tmp_path)
    case = result["cases"][0]
    assert case["failed_attempts"] == 1
    assert case["retried_attempts"] == 1
    assert case["latency_seconds"]["median"] == 0.2
    assert case["request_throughput_per_second"] == 1.0
