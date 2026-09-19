"""AgentTrace command-line interface."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer

from agenttrace.agent.orchestration import run_agent_group
from agenttrace.analysis.pipeline import analyze_experiment
from agenttrace.analysis.report import generate_report
from agenttrace.benchmarking.aggregate import aggregate_experiment
from agenttrace.benchmarking.runner import BenchmarkRunner
from agenttrace.cli.doctor import run_doctor, serialize_checks
from agenttrace.config import (
    AgentConfig,
    BenchmarkConfig,
    ReplayConfig,
    SyntheticConfig,
    load_config,
)
from agenttrace.instrumentation.tokens import build_token_counter
from agenttrace.replay.engine import ReplayEngine
from agenttrace.replay.loader import load_workload
from agenttrace.replay.transform import transform_workload, write_transformation_manifest
from agenttrace.serving.metrics import VLLMMetricsAdapter
from agenttrace.serving.mock_server import MockServerConfig, serve_mock
from agenttrace.serving.openai import OpenAICompatibleClient
from agenttrace.telemetry.logging import configure_logging
from agenttrace.telemetry.metrics import serve_metrics
from agenttrace.telemetry.tracing import configure_tracing
from agenttrace.tracing.summary import summarize_trace
from agenttrace.tracing.synthetic import generate_synthetic_trace
from agenttrace.tracing.validation import validate_file

app = typer.Typer(help="Profile coding-agent inference workloads.", no_args_is_help=True)
agent_app = typer.Typer(help="Execute bounded coding agents.")
trace_app = typer.Typer(help="Generate, validate, and inspect traces.")
replay_app = typer.Typer(help="Replay inference workloads.")
benchmark_app = typer.Typer(help="Run and report benchmark matrices.")
app.add_typer(agent_app, name="agent")
app.add_typer(trace_app, name="trace")
app.add_typer(replay_app, name="replay")
app.add_typer(benchmark_app, name="benchmark")


@app.callback()
def initialize_telemetry() -> None:
    """Enable optional exporters from environment variables without storing secrets."""

    if os.getenv("AGENTTRACE_JSON_LOGS") == "1":
        configure_logging()
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        endpoint = otlp_endpoint.rstrip("/")
        if not endpoint.endswith("/v1/traces"):
            endpoint += "/v1/traces"
        configure_tracing(otlp_http_endpoint=endpoint)
    metrics_port = os.getenv("AGENTTRACE_METRICS_PORT")
    if metrics_port:
        serve_metrics(int(metrics_port), os.getenv("AGENTTRACE_METRICS_HOST", "127.0.0.1"))


@app.command("mock-server")
def mock_server(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8010,
    first_token_delay: Annotated[float, typer.Option(min=0)] = 0.01,
    inter_chunk_delay: Annotated[float, typer.Option(min=0)] = 0.005,
    fail_every: Annotated[int, typer.Option(min=0)] = 0,
) -> None:
    """Run the CPU-only test server; its timings are simulated, not model performance."""

    typer.echo(f"AgentTrace mock server listening on http://{host}:{port}/v1")
    serve_mock(
        MockServerConfig(
            host=host,
            port=port,
            first_token_delay_seconds=first_token_delay,
            inter_chunk_delay_seconds=inter_chunk_delay,
            fail_every=fail_every,
        )
    )


@agent_app.command("run")
def agent_run(config: Annotated[Path, typer.Option(exists=True, dir_okay=False)]) -> None:
    settings = load_config(config, AgentConfig)

    async def execute() -> None:
        outcomes = await run_agent_group(
            [settings], lambda endpoint: OpenAICompatibleClient(endpoint)
        )
        typer.echo(json.dumps([outcome.model_dump(mode="json") for outcome in outcomes], indent=2))

    asyncio.run(execute())


@trace_app.command("validate")
def trace_validate(input: Annotated[Path, typer.Option(exists=True, dir_okay=False)]) -> None:
    report = validate_file(input)
    typer.echo(
        json.dumps(
            {"valid": report.valid, "errors": report.errors, "warnings": report.warnings}, indent=2
        )
    )
    if not report.valid:
        raise typer.Exit(1)


@trace_app.command("summarize")
def trace_summarize(input: Annotated[Path, typer.Option(exists=True, dir_okay=False)]) -> None:
    typer.echo(json.dumps(summarize_trace(input).as_dict(), indent=2))


@trace_app.command("generate")
def trace_generate(config: Annotated[Path, typer.Option(exists=True, dir_okay=False)]) -> None:
    settings = load_config(config, SyntheticConfig)
    generate_synthetic_trace(settings)
    typer.echo(f"wrote synthetic trace: {settings.output}")


@replay_app.command("run")
def replay_run(
    trace: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()] = Path("results/replay.json"),
) -> None:
    settings = load_config(config, ReplayConfig)

    async def execute() -> None:
        workload = load_workload(trace)
        counter = build_token_counter(settings.tokenizer)
        if settings.mode == "parameterized":
            workload = transform_workload(workload, settings, counter)
            write_transformation_manifest(
                output.with_suffix(".manifest.json"),
                source_trace=trace,
                source_trace_id=workload[0].source_trace_id if workload else "empty",
                config=settings,
                requests=workload,
            )
        client = OpenAICompatibleClient(settings.endpoint, token_counter=counter)
        try:
            engine = ReplayEngine(settings, client)
            if settings.mode == "closed_loop":
                result = await engine.run_closed_loop(workload)
            else:
                result = await engine.run_open_loop(workload)
            if settings.mode == "parameterized":
                result = result.model_copy(
                    update={
                        "mode": "parameterized",
                        "attempts": [
                            attempt.model_copy(update={"mode": "parameterized"})
                            for attempt in result.attempts
                        ],
                    }
                )
        finally:
            await client.close()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        typer.echo(f"wrote replay observations: {output}")

    asyncio.run(execute())


@benchmark_app.command("run")
def benchmark_run(config: Annotated[Path, typer.Option(exists=True, dir_okay=False)]) -> None:
    settings = load_config(config, BenchmarkConfig)
    counter = build_token_counter(settings.replay.tokenizer)
    metrics_url = os.getenv("AGENTTRACE_VLLM_METRICS_URL")
    metrics = VLLMMetricsAdapter(metrics_url) if metrics_url else None

    async def execute() -> None:
        runner = BenchmarkRunner(
            settings,
            lambda endpoint: OpenAICompatibleClient(endpoint, token_counter=counter),
            token_counter=counter,
            server_metrics=metrics,
        )
        directory = await runner.run()
        aggregate_experiment(directory)
        analyze_experiment(directory, source_trace=settings.source_trace)
        report = generate_report(directory)
        typer.echo(f"wrote benchmark report: {report}")

    asyncio.run(execute())


@benchmark_app.command("report")
def benchmark_report(results: Annotated[Path, typer.Option(exists=True, file_okay=False)]) -> None:
    aggregate_experiment(results)
    analyze_experiment(results)
    typer.echo(str(generate_report(results)))


@app.command("doctor")
def doctor(
    config: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    endpoint = load_config(config, ReplayConfig).endpoint if config else None
    checks = asyncio.run(run_doctor(endpoint))
    typer.echo(json.dumps(serialize_checks(checks), indent=2))
    if any(check.status == "failed" for check in checks):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
