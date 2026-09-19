import json
from pathlib import Path

from agenttrace.analysis.report import generate_report


def test_report_labels_mock_results_and_exposes_failures(tmp_path: Path) -> None:
    metadata = {
        "experiment_id": "report-test",
        "started_at": "2026-01-01T00:00:00Z",
        "tokenizer": "test",
        "token_count_method": "fixture",
        "configuration": {"measurement_label": "mock", "repetitions": 1},
    }
    case = {
        "case_id": "case-1",
        "configuration": {
            "context_tokens": 128,
            "concurrent_agents": 2,
            "replay_mode": "open_loop",
            "pattern": "concurrent",
        },
        "successful_requests": 3,
        "failed_attempts": 1,
        "timeout_attempts": 2,
        "ttft_seconds": {"median": 0.1},
        "latency_seconds": {"median": 0.4},
        "output_token_throughput_per_second": 20,
    }
    analysis = {
        "measured_aggregate_results": {"cases": [case]},
        "charts": {"context_vs_ttft": "unavailable: fixture"},
        "unavailable_analyses": {"queueing": "not correlated"},
        "interpretation": "fixture observations only",
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    report = generate_report(tmp_path).read_text(encoding="utf-8")
    assert "MOCK-SERVER DEMONSTRATION" in report
    assert "1/2" in report
    assert "not correlated" in report
    assert "fixture observations only" in report
