"""Deterministic synthetic traces for scheduling tests and controlled experiments."""

from __future__ import annotations

import platform
from datetime import UTC, datetime, timedelta

from agenttrace import __version__
from agenttrace.config import SyntheticConfig
from agenttrace.instrumentation.tokens import WhitespaceTokenCounter
from agenttrace.models import ExecutionStatus, RequestStatus
from agenttrace.tracing.schema import (
    AgentRecord,
    EnvironmentInfo,
    OutcomeRecord,
    PromptTokenBreakdown,
    RequestRecord,
    TraceHeader,
)
from agenttrace.tracing.storage import TraceWriter, write_manifest


def generate_synthetic_trace(config: SyntheticConfig) -> None:
    """Create a clearly labeled non-measured workload with stable contents."""

    base = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=config.seed)
    trace_id = f"trace_synthetic_{config.seed}"
    counter = WhitespaceTokenCounter()
    records: list[object] = []
    records.append(
        TraceHeader(
            experiment_id=config.experiment_id,
            trace_id=trace_id,
            created_at=base,
            source="synthetic",
            content_mode="full",
            seed=config.seed,
            environment=EnvironmentInfo(
                python_version=platform.python_version(),
                platform=platform.platform(),
                agenttrace_version=__version__,
                model="synthetic-not-an-inference-measurement",
                tokenizer=counter.identity,
            ),
            execution_config=config.model_dump(mode="json"),
        )
    )
    for agent_index in range(config.agents):
        agent_id = f"synthetic-agent-{agent_index + 1}"
        agent_start = base + timedelta(milliseconds=agent_index)
        records.append(
            AgentRecord(
                experiment_id=config.experiment_id,
                trace_id=trace_id,
                agent_id=agent_id,
                started_at=agent_start,
                task_hash=f"synthetic:{config.seed}:{agent_index}",
            )
        )
        prior_tokens: int | None = None
        for sequence in range(config.requests_per_agent):
            prompt_tokens = config.initial_prompt_tokens + sequence * config.prompt_growth_tokens
            prompt = counter.construct_text(prompt_tokens, f"agent{agent_index} context")
            submitted = agent_start + timedelta(seconds=sequence * config.inter_request_seconds)
            elapsed = 0.02 + prompt_tokens / 100_000 + config.output_tokens / 50_000
            completed = submitted + timedelta(seconds=elapsed)
            records.append(
                RequestRecord(
                    experiment_id=config.experiment_id,
                    trace_id=trace_id,
                    agent_id=agent_id,
                    request_id=f"req_synthetic_{agent_index}_{sequence}",
                    sequence_number=sequence,
                    model="synthetic-not-an-inference-measurement",
                    created_at=submitted,
                    submitted_at=submitted,
                    completed_at=completed,
                    elapsed_seconds=elapsed,
                    monotonic_started=sequence * config.inter_request_seconds,
                    monotonic_completed=sequence * config.inter_request_seconds + elapsed,
                    time_since_previous_request_seconds=(
                        None if sequence == 0 else config.inter_request_seconds
                    ),
                    input_tokens=prompt_tokens,
                    output_tokens=config.output_tokens,
                    context_growth_tokens=(
                        0 if prior_tokens is None else prompt_tokens - prior_tokens
                    ),
                    prompt_breakdown=PromptTokenBreakdown(user=prompt_tokens),
                    tokenizer=counter.identity,
                    token_count_method=counter.method,
                    status=RequestStatus.SUCCEEDED,
                    sampling_parameters={"synthetic": True},
                    concurrency_at_submission=config.agents,
                    prompt=prompt,
                )
            )
            prior_tokens = prompt_tokens
        records.append(
            OutcomeRecord(
                experiment_id=config.experiment_id,
                trace_id=trace_id,
                agent_id=agent_id,
                status=ExecutionStatus.SUCCEEDED,
                summary="synthetic outcome; no coding agent executed",
                iterations=config.requests_per_agent,
                total_input_tokens=sum(
                    config.initial_prompt_tokens + step * config.prompt_growth_tokens
                    for step in range(config.requests_per_agent)
                ),
                total_output_tokens=config.output_tokens * config.requests_per_agent,
                started_at=agent_start,
                completed_at=agent_start
                + timedelta(seconds=config.requests_per_agent * config.inter_request_seconds),
            )
        )
    with TraceWriter(config.output, mode="w") as writer:
        for record in records:
            writer.write(record)  # type: ignore[arg-type]
    write_manifest(
        config.output.with_suffix(".manifest.json"),
        {
            "kind": "synthetic",
            "not_real_measurements": True,
            "trace": str(config.output),
            "seed": config.seed,
            "parameters": config.model_dump(mode="json"),
        },
    )
