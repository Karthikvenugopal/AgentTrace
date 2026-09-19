from pathlib import Path

from agenttrace.benchmarking.matrix import expand_matrix, ordered_cases
from agenttrace.config import BenchmarkConfig


def test_matrix_expands_actual_configured_axes() -> None:
    config = BenchmarkConfig(
        experiment_id="matrix",
        source_trace=Path("trace.jsonl"),
        matrix={
            "context_tokens": [128, 1024],
            "concurrent_agents": [1, 4],
            "patterns": ["sequential"],
            "workload_types": ["parameterized"],
            "replay_modes": ["open_loop", "closed_loop"],
        },
    )
    cases = expand_matrix(config)
    assert len(cases) == 8
    assert len({case.case_id for case in cases}) == 8
    assert ordered_cases(config, 0) == ordered_cases(config, 0)
    assert ordered_cases(config, 0) != ordered_cases(config, 1)
