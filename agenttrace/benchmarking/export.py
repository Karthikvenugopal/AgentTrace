"""CSV exports for audit-friendly benchmark analysis."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

REQUEST_COLUMNS = (
    "experiment_id",
    "case_id",
    "repetition",
    "target_context_tokens",
    "configured_concurrent_agents",
    "pattern",
    "workload_type",
    "replay_mode",
    "replay_session_id",
    "replay_request_id",
    "source_trace_id",
    "source_request_id",
    "agent_id",
    "parent_agent_id",
    "sequence_number",
    "attempt_number",
    "is_retry_attempt",
    "status",
    "submitted_at",
    "completed_at",
    "scheduled_offset_seconds",
    "submitted_offset_seconds",
    "client_scheduling_delay_seconds",
    "latency_seconds",
    "ttft_seconds",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_token_method",
    "output_token_method",
    "server_request_id",
    "finish_reason",
    "concurrency_at_submission",
    "error_type",
    "error_message",
)


SUMMARY_COLUMNS = (
    "experiment_id",
    "case_id",
    "target_context_tokens",
    "configured_concurrent_agents",
    "pattern",
    "workload_type",
    "replay_mode",
    "trials",
    "logical_requests",
    "successful_requests",
    "failed_requests",
    "failed_attempts",
    "timeout_attempts",
    "retried_requests",
    "failure_rate",
    "actual_input_tokens_p50",
    "actual_output_tokens_p50",
    "latency_p50_seconds",
    "latency_p95_seconds",
    "ttft_p50_seconds",
    "ttft_p95_seconds",
    "request_throughput_per_second",
    "input_token_throughput_per_second",
    "output_token_throughput_per_second",
    "total_token_throughput_per_second",
    "total_token_throughput_valid",
    "input_token_methods",
    "output_token_methods",
    "maximum_request_concurrency",
    "effective_request_concurrency",
    "server_running_requests_max",
    "server_waiting_requests_max",
    "server_gpu_kv_cache_usage_mean",
    "server_gpu_kv_cache_usage_max",
    "server_queue_mean_seconds",
    "itl_status",
)


def write_csv_artifacts(
    experiment_dir: Path,
    observations: list[dict[str, Any]],
    aggregates: dict[str, Any],
) -> tuple[Path, Path]:
    requests_path = experiment_dir / "requests.csv"
    summary_path = experiment_dir / "summary.csv"
    with requests_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUEST_COLUMNS)
        writer.writeheader()
        for observation in observations:
            for row in _request_rows(observation):
                writer.writerow(row)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for case in aggregates["cases"]:
            writer.writerow(_summary_row(aggregates["experiment_id"], case))
    return requests_path, summary_path


def _request_rows(observation: dict[str, Any]) -> list[dict[str, Any]]:
    case = observation["case"]
    attempts = observation["replay"]["attempts"]
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        input_tokens = attempt.get("input_tokens")
        output_tokens = attempt.get("output_tokens")
        rows.append(
            {
                "experiment_id": observation.get("experiment_id", ""),
                "case_id": case["case_id"],
                "repetition": observation.get("repetition", 0),
                "target_context_tokens": case["context_tokens"],
                "configured_concurrent_agents": case["concurrent_agents"],
                "pattern": case["pattern"],
                "workload_type": case["workload_type"],
                "replay_mode": case["replay_mode"],
                "replay_session_id": attempt.get("replay_session_id", ""),
                "replay_request_id": attempt.get("replay_request_id", ""),
                "source_trace_id": attempt.get("source_trace_id", ""),
                "source_request_id": attempt.get("source_request_id", ""),
                "agent_id": attempt.get("agent_id", ""),
                "parent_agent_id": attempt.get("parent_agent_id") or "",
                "sequence_number": attempt.get("sequence_number", ""),
                "attempt_number": attempt["attempt_number"],
                "is_retry_attempt": int(attempt["attempt_number"]) > 1,
                "status": attempt["status"],
                "submitted_at": attempt.get("submitted_at", ""),
                "completed_at": attempt.get("completed_at", ""),
                "scheduled_offset_seconds": attempt.get("scheduled_offset_seconds", ""),
                "submitted_offset_seconds": attempt.get("submitted_offset_seconds", ""),
                "client_scheduling_delay_seconds": attempt.get(
                    "client_scheduling_delay_seconds", ""
                ),
                "latency_seconds": attempt["latency_seconds"],
                "ttft_seconds": attempt.get("ttft_seconds", ""),
                "input_tokens": "" if input_tokens is None else input_tokens,
                "output_tokens": "" if output_tokens is None else output_tokens,
                "total_tokens": (
                    ""
                    if input_tokens is None or output_tokens is None
                    else int(input_tokens) + int(output_tokens)
                ),
                "input_token_method": attempt.get("input_token_method") or "",
                "output_token_method": attempt.get("output_token_method") or "",
                "server_request_id": attempt.get("server_request_id") or "",
                "finish_reason": attempt.get("finish_reason") or "",
                "concurrency_at_submission": _concurrency_at_submission(attempt, attempts),
                "error_type": attempt.get("error_type") or "",
                "error_message": attempt.get("error_message") or "",
            }
        )
    return rows


def _concurrency_at_submission(target: dict[str, Any], attempts: list[dict[str, Any]]) -> int | str:
    if not target.get("submitted_at"):
        return ""
    submitted = _parse_time(target["submitted_at"])
    active = 0
    for attempt in attempts:
        if not attempt.get("submitted_at") or not attempt.get("completed_at"):
            continue
        if _parse_time(attempt["submitted_at"]) <= submitted < _parse_time(attempt["completed_at"]):
            active += 1
    return active


def _summary_row(experiment_id: str, case: dict[str, Any]) -> dict[str, Any]:
    config = case["configuration"]
    server = case["server_metrics"]
    return {
        "experiment_id": experiment_id,
        "case_id": case["case_id"],
        "target_context_tokens": config["context_tokens"],
        "configured_concurrent_agents": config["concurrent_agents"],
        "pattern": config["pattern"],
        "workload_type": config["workload_type"],
        "replay_mode": config["replay_mode"],
        "trials": case["trials"],
        "logical_requests": case["logical_requests"],
        "successful_requests": case["successful_requests"],
        "failed_requests": case["failed_requests"],
        "failed_attempts": case["failed_attempts"],
        "timeout_attempts": case["timeout_attempts"],
        "retried_requests": case["retried_requests"],
        "failure_rate": case["failure_rate"],
        "actual_input_tokens_p50": case["input_tokens_per_request"]["median"],
        "actual_output_tokens_p50": case["output_tokens_per_request"]["median"],
        "latency_p50_seconds": case["latency_seconds"]["median"],
        "latency_p95_seconds": case["latency_seconds"]["p95"],
        "ttft_p50_seconds": case["ttft_seconds"]["median"],
        "ttft_p95_seconds": case["ttft_seconds"]["p95"],
        "request_throughput_per_second": case["request_throughput_per_second"],
        "input_token_throughput_per_second": case["input_token_throughput_per_second"],
        "output_token_throughput_per_second": case["output_token_throughput_per_second"],
        "total_token_throughput_per_second": case["total_token_throughput_per_second"],
        "total_token_throughput_valid": case["total_token_throughput_valid"],
        "input_token_methods": ";".join(case["input_token_methods"]),
        "output_token_methods": ";".join(case["output_token_methods"]),
        "maximum_request_concurrency": case["maximum_request_concurrency"],
        "effective_request_concurrency": case["effective_request_concurrency"],
        "server_running_requests_max": server["running_requests"]["maximum"],
        "server_waiting_requests_max": server["waiting_requests"]["maximum"],
        "server_gpu_kv_cache_usage_mean": server["gpu_kv_cache_usage"]["mean"],
        "server_gpu_kv_cache_usage_max": server["gpu_kv_cache_usage"]["maximum"],
        "server_queue_mean_seconds": server["queue_time_window"]["mean_seconds"],
        "itl_status": case["itl"]["reason"],
    }


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)
