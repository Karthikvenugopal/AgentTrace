from typing import Any

from agenttrace.analysis.pipeline import _benchmark_comparisons


def _case(
    context: int, concurrency: int, ttft: float, latency: float, rate: float
) -> dict[str, Any]:
    return {
        "case_id": f"ctx-{context}-c-{concurrency}",
        "configuration": {
            "context_tokens": context,
            "concurrent_agents": concurrency,
            "pattern": "concurrent",
            "workload_type": "parameterized",
            "replay_mode": "closed_loop",
        },
        "successful_requests": 20,
        "failed_requests": 0,
        "failure_rate": 0.0,
        "ttft_seconds": {"p95": ttft},
        "latency_seconds": {"p95": latency},
        "output_token_throughput_per_second": rate,
    }


def test_resume_comparisons_retain_source_values_and_calculate_changes() -> None:
    cases = [
        _case(2048, 1, 0.1, 1.0, 100.0),
        _case(32000, 1, 0.2, 2.0, 80.0),
        _case(8192, 1, 0.15, 1.2, 100.0),
        _case(8192, 8, 0.30, 2.4, 180.0),
    ]
    comparisons = _benchmark_comparisons({"cases": cases})
    context = comparisons["context_scaling_at_concurrency_1"]
    assert context["baseline"]["case_id"] == "ctx-2048-c-1"
    assert context["comparison"]["case_id"] == "ctx-32000-c-1"
    assert context["p95_ttft_percent_change"] == 100.0
    concurrency = comparisons["concurrency_scaling_at_representative_context"]
    assert concurrency["representative_context_tokens"] == 8192
    assert concurrency["output_throughput_percent_change"] == 80.0
    assert concurrency["p95_latency_percent_change"] == 100.0
    assert comparisons["largest_observed_latency_tradeoff"] is not None
