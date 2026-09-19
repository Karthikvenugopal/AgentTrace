"""Human-readable benchmark reports grounded in generated measurements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenttrace.analysis.pipeline import analyze_experiment


def generate_report(experiment_dir: Path) -> Path:
    metadata = json.loads((experiment_dir / "metadata.json").read_text(encoding="utf-8"))
    analysis_path = experiment_dir / "analysis.json"
    if not analysis_path.exists():
        analyze_experiment(experiment_dir)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    cases = analysis["measured_aggregate_results"]["cases"]
    label = metadata["configuration"].get("measurement_label", "mock")
    classification = (
        "REAL INFERENCE MEASUREMENTS"
        if label == "real_inference"
        else "MOCK-SERVER DEMONSTRATION — NOT REAL INFERENCE PERFORMANCE"
    )
    lines = [
        f"# AgentTrace benchmark report: {metadata['experiment_id']}",
        "",
        f"> **{classification}**",
        "",
        f"Started: `{metadata['started_at']}`  ",
        f"Tokenizer: `{metadata['tokenizer']}` ({metadata['token_count_method']})  ",
        f"Repeated trials: `{metadata['configuration']['repetitions']}`",
        "",
        "## Measurement semantics",
        "",
        "Open-loop replay schedules arrivals by trace time without waiting for prior responses. "
        "Closed-loop replay advances each agent stream only after its preceding request completes "
        "and the recorded delay elapses. Client TTFT ends at the first non-empty streamed chunk; "
        "it is not treated as exact server queueing time. Failed and timeout attempts are shown "
        "but excluded from successful-request latency and throughput aggregates.",
        "",
        "## Aggregate observations",
        "",
        "| Case | Context tokens | Agents | Mode | Pattern | n | Median TTFT (s) | Median latency (s) | Output tok/s | Failed/timeouts |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        config = case["configuration"]
        lines.append(
            "| {case_id} | {context} | {agents} | {mode} | {pattern} | {count} | {ttft} | "
            "{latency} | {throughput} | {failed}/{timeouts} |".format(
                case_id=case["case_id"],
                context=config["context_tokens"],
                agents=config["concurrent_agents"],
                mode=config["replay_mode"],
                pattern=config["pattern"],
                count=case["successful_requests"],
                ttft=_number(case["ttft_seconds"]["median"]),
                latency=_number(case["latency_seconds"]["median"]),
                throughput=_number(case["output_token_throughput_per_second"]),
                failed=case["failed_attempts"],
                timeouts=case["timeout_attempts"],
            )
        )
    lines.extend(
        [
            "",
            "Tail percentiles are omitted when fewer than 20 observations are available for p95 "
            "or fewer than 100 for p99.",
            "",
            "## Charts",
            "",
        ]
    )
    for name, path in analysis["charts"].items():
        if path.startswith("unavailable"):
            lines.append(f"- `{name}`: {path}")
        else:
            relative = Path(path).relative_to(experiment_dir)
            lines.append(f"- [{name}]({relative.as_posix()})")
    lines.extend(["", "## Unavailable measurements", ""])
    for name, reason in analysis["unavailable_analyses"].items():
        lines.append(f"- `{name}`: {reason}")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            analysis["interpretation"],
            "",
            "Machine-readable inputs: `metadata.json`, `observations.jsonl`, `aggregates.json`, "
            "and `analysis.json`.",
        ]
    )
    output = experiment_dir / "report.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"
