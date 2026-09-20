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
    configuration = metadata["configuration"]
    label = configuration.get("measurement_label", "mock")
    verified_real = bool((metadata.get("preflight") or {}).get("vllm_metrics_verified"))
    if label == "real_inference" and verified_real:
        classification = "REAL vLLM/GPU INFERENCE MEASUREMENTS"
    elif label == "real_inference":
        classification = "UNVERIFIED INFERENCE ARTIFACT — NOT ELIGIBLE FOR PERFORMANCE CLAIMS"
    else:
        classification = "MOCK-SERVER DEMONSTRATION — NOT REAL INFERENCE PERFORMANCE"
    replay_modes = sorted({case["configuration"].get("replay_mode", "unknown") for case in cases})
    lines = [
        f"# AgentTrace benchmark report: {metadata['experiment_id']}",
        "",
        f"> **{classification}**",
        "",
        "## Research question",
        "",
        "How do prompt/context length and concurrent coding-agent requests affect inference "
        "latency and throughput?",
        "",
        "## Environment and reproducibility",
        "",
        f"- Started: `{metadata['started_at']}`",
        f"- Tokenizer: `{metadata['tokenizer']}` (`{metadata['token_count_method']}`)",
        f"- Repetitions per configuration: `{configuration['repetitions']}`",
        f"- Warm-up requests per case: `{configuration.get('warmup_requests', 0)}`; excluded",
        f"- Matrix order randomized: `{configuration.get('randomize_order', False)}`",
        f"- Per-repetition rotation: `{configuration.get('rotate_order_each_repetition', False)}`",
        f"- Workload seed: `{configuration.get('seed')}`",
        f"- Replay mode(s): `{', '.join(replay_modes)}`",
    ]
    server = metadata.get("server_configuration") or {}
    if server:
        lines.extend(["", "| Server parameter | Recorded value |", "|---|---|"])
        for key in (
            "backend",
            "vllm_version",
            "expected_vllm_image",
            "model",
            "model_revision",
            "gpu",
            "dtype",
            "tensor_parallel_size",
            "max_model_len",
            "max_num_seqs",
            "gpu_memory_utilization",
            "prefix_caching",
        ):
            if key in server:
                lines.append(f"| `{key}` | `{server[key]}` |")

    lines.extend(
        [
            "",
            "## Methodology",
            "",
            "Each case uses parameterized request replay with prompt text constructed by the "
            "configured tokenizer (or a labeled fallback in mock mode). Closed-loop replay "
            "keeps one request active per agent "
            "stream; open-loop replay, when configured, submits by scheduled arrival time. "
            "`max_tokens` is fixed across configurations; actual server-reported input and "
            "output tokens remain in the raw data.",
            "",
            "Client TTFT runs from request submission to the first non-empty streamed content "
            "chunk. End-to-end latency ends when the stream closes. Throughput uses the window "
            "from the earliest measured request submission to the latest measured completion "
            "within each trial, then pools token counts and durations across repetitions.",
            "",
            "Failed and timeout attempts are retained but excluded from successful-request "
            "latency and throughput. A stable configuration has zero failed logical requests. "
            "p95 is suppressed below 20 successful observations.",
            "",
            "**ITL unavailable with current instrumentation:** streamed chunks may contain "
            "multiple tokens, so chunk-arrival intervals are not per-token decode latency.",
            "",
            "## Per-configuration results",
            "",
            "| Context target | Agents | Success | Failed/timeouts/retried | Input/output tokens | "
            "Latency p50/p95 (s) | TTFT p50/p95 (s) | Output/total tok/s | Req/s | "
            "Max/effective concurrency |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in sorted(
        cases,
        key=lambda item: (
            item["configuration"]["context_tokens"],
            item["configuration"]["concurrent_agents"],
        ),
    ):
        config = case["configuration"]
        latency = case["latency_seconds"]
        ttft = case["ttft_seconds"]
        lines.append(
            "| {context} | {agents} | {success} | {failed}/{timeouts}/{retried} | "
            "{input}/{output} | "
            "{latency50}/{latency95} | {ttft50}/{ttft95} | {out_rate}/{total_rate} | "
            "{request_rate} | {maximum}/{effective} |".format(
                context=config["context_tokens"],
                agents=config["concurrent_agents"],
                success=case["successful_requests"],
                failed=case.get("failed_requests", case.get("failed_attempts", 0)),
                timeouts=case.get("timeout_attempts", 0),
                retried=case.get("retried_requests", case.get("retried_attempts", 0)),
                input=case.get("total_input_tokens", "n/a"),
                output=case.get("total_output_tokens", "n/a"),
                latency50=_number(latency.get("median")),
                latency95=_number(latency.get("p95")),
                ttft50=_number(ttft.get("median")),
                ttft95=_number(ttft.get("p95")),
                out_rate=_number(case.get("output_token_throughput_per_second")),
                total_rate=_number(case.get("total_token_throughput_per_second")),
                request_rate=_number(case.get("request_throughput_per_second")),
                maximum=case.get("maximum_request_concurrency", "n/a"),
                effective=_number(case.get("effective_request_concurrency")),
            )
        )

    lines.extend(["", "## Server metrics", ""])
    server_rows = [case for case in cases if case.get("server_metrics")]
    if server_rows:
        lines.extend(
            [
                "vLLM metrics below are aggregate samples for each measurement window, not "
                "request-correlated queueing observations.",
                "",
                "| Case | Waiting max | Running max | KV-cache mean/max | Queue mean (s) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for case in server_rows:
            metrics = case["server_metrics"]
            lines.append(
                "| {case_id} | {waiting} | {running} | {kv_mean}/{kv_max} | {queue} |".format(
                    case_id=case["case_id"],
                    waiting=_number(metrics["waiting_requests"]["maximum"]),
                    running=_number(metrics["running_requests"]["maximum"]),
                    kv_mean=_number(metrics["gpu_kv_cache_usage"]["mean"]),
                    kv_max=_number(metrics["gpu_kv_cache_usage"]["maximum"]),
                    queue=_number(metrics["queue_time_window"]["mean_seconds"]),
                )
            )
    else:
        lines.append("Server-side metrics were unavailable in this result set.")

    lines.extend(["", "## Controlled comparisons", ""])
    comparisons = analysis.get("comparisons") or {}
    lines.extend(_comparison_lines(comparisons))
    lines.extend(["", "## Charts", ""])
    for name, path in analysis["charts"].items():
        if path.startswith("unavailable"):
            lines.append(f"- `{name}`: {path}")
        else:
            relative = Path(path).relative_to(experiment_dir)
            lines.append(f"- [{name}]({relative.as_posix()})")

    lines.extend(["", "## Limitations", ""])
    lines.extend(
        [
            "- Parameterized prompts isolate token length but do not reproduce the semantic "
            "content distribution or model-driven branching of a live coding agent.",
            "- Client TTFT includes transport, server queueing, prefill, first-token decode, and "
            "stream buffering; it is not relabeled as queueing time.",
            "- Aggregate Prometheus histograms cannot be assigned exactly to individual requests.",
            "- Repeated trials reduce but do not remove cache, thermal, clock, and background-load "
            "effects. Results apply only to the recorded model, server flags, and hardware.",
        ]
    )
    for name, reason in analysis["unavailable_analyses"].items():
        lines.append(f"- `{name}`: {reason}")

    lines.extend(["", "## Resume-ready metrics", ""])
    if label != "real_inference" or not verified_real:
        lines.append(
            "Unavailable: this artifact is not a verified real vLLM/GPU run. Mock measurements "
            "must not be used as model inference performance."
        )
    else:
        lines.extend(_resume_metrics(cases, comparisons))

    lines.extend(
        [
            "",
            "## Audit artifacts",
            "",
            "Machine-readable sources: `requests.csv` (one row per measured attempt), "
            "`summary.csv` (one row per configuration), `observations.jsonl`, "
            "`aggregates.json`, `analysis.json`, and `metadata.json`.",
            "",
            analysis["interpretation"],
        ]
    )
    output = experiment_dir / "report.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _comparison_lines(comparisons: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    context = comparisons.get("context_scaling_at_concurrency_1")
    if context:
        lines.append(_comparison_sentence("Context scaling", context))
    else:
        lines.append(
            "- Context scaling: unavailable; two concurrency=1 cases with p95 are required."
        )
    concurrency = comparisons.get("concurrency_scaling_at_representative_context")
    if concurrency:
        lines.append(_comparison_sentence("Concurrency scaling", concurrency))
    else:
        lines.append(
            "- Concurrency scaling: unavailable; stable baseline and higher-concurrency p95 "
            "are required."
        )
    best = comparisons.get("best_stable_output_throughput")
    if best:
        lines.append(
            f"- Best stable throughput: context `{best['context_tokens']}`, concurrency "
            f"`{best['concurrent_agents']}`, `{_number(best['output_tokens_per_second'])}` output "
            f"tokens/s, p95 TTFT `{_number(best['p95_ttft_seconds'])}` s, failure rate "
            f"`{_percent(best['failure_rate'])}` (case `{best['case_id']}`)."
        )
    else:
        lines.append("- Best stable throughput: unavailable; no zero-failure case.")
    tradeoff = comparisons.get("largest_observed_latency_tradeoff")
    if tradeoff:
        lines.append(_comparison_sentence("Largest observed latency trade-off", tradeoff))
    else:
        lines.append(
            "- Largest latency trade-off: not observed; no tested stable case both increased "
            "output throughput and p95 latency relative to concurrency=1."
        )
    return lines


def _comparison_sentence(label: str, comparison: dict[str, Any]) -> str:
    baseline = comparison["baseline"]
    target = comparison["comparison"]
    return (
        f"- {label}: `{baseline['context_tokens']}` tokens / concurrency "
        f"`{baseline['concurrent_agents']}` (case `{baseline['case_id']}`) to "
        f"`{target['context_tokens']}` tokens / concurrency `{target['concurrent_agents']}` "
        f"(case `{target['case_id']}`): output throughput "
        f"`{_number(baseline['output_tokens_per_second'])}` → "
        f"`{_number(target['output_tokens_per_second'])}` tokens/s "
        f"(`{_signed_percent(comparison.get('output_throughput_percent_change'))}`); "
        f"p95 TTFT `{_number(baseline['p95_ttft_seconds'])}` → "
        f"`{_number(target['p95_ttft_seconds'])}` s "
        f"(`{_signed_percent(comparison.get('p95_ttft_percent_change'))}`); p95 latency "
        f"`{_number(baseline['p95_latency_seconds'])}` → "
        f"`{_number(target['p95_latency_seconds'])}` s "
        f"(`{_signed_percent(comparison.get('p95_latency_percent_change'))}`)."
    )


def _resume_metrics(cases: list[dict[str, Any]], comparisons: dict[str, Any]) -> list[str]:
    request_count = sum(int(case.get("logical_requests", 0)) for case in cases)
    context_count = len({case["configuration"]["context_tokens"] for case in cases})
    concurrency_count = len({case["configuration"]["concurrent_agents"] for case in cases})
    metrics = [
        f"- Replayed `{request_count}` measured inference requests across "
        f"`{context_count}` context "
        f"lengths and `{concurrency_count}` concurrency levels. Source: all rows in `summary.csv`."
    ]
    context = comparisons.get("context_scaling_at_concurrency_1")
    if context:
        metrics.append(_comparison_sentence("Measured context effect", context))
    concurrency = comparisons.get("concurrency_scaling_at_representative_context")
    if concurrency:
        metrics.append(_comparison_sentence("Measured concurrency effect", concurrency))
    best = comparisons.get("best_stable_output_throughput")
    if best:
        metrics.append(
            f"- Highest stable output throughput was `{_number(best['output_tokens_per_second'])}` "
            f"tokens/s at `{best['context_tokens']}` context tokens and concurrency "
            f"`{best['concurrent_agents']}`; p95 TTFT was "
            f"`{_number(best['p95_ttft_seconds'])}` s and failure rate was "
            f"`{_percent(best['failure_rate'])}`. Source case: `{best['case_id']}`."
        )
    return metrics[:5]


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.2f}%"


def _signed_percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.2f}%"
