"""Reproducible parameterized workload transformations."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from agenttrace.config import ReplayConfig
from agenttrace.instrumentation.tokens import TokenCounter
from agenttrace.models import ChatMessage, MessageRole
from agenttrace.replay.models import WorkloadRequest
from agenttrace.tracing.storage import write_manifest


def transform_workload(
    workload: list[WorkloadRequest], config: ReplayConfig, counter: TokenCounter
) -> list[WorkloadRequest]:
    """Derive a modified workload while preserving source-request lineage."""

    if not workload:
        return []
    grouped: dict[str, list[WorkloadRequest]] = defaultdict(list)
    for request in workload:
        grouped[request.agent_id].append(request)
    templates = sorted(grouped.values(), key=lambda group: group[0].agent_id)
    target_agents = config.concurrent_agents or len(templates)
    transformed: list[WorkloadRequest] = []
    global_index = 0
    for agent_index in range(target_agents):
        source_stream = templates[agent_index % len(templates)]
        source_stream = sorted(source_stream, key=lambda request: request.sequence_number)
        target_length = config.execution_length or len(source_stream)
        agent_id = f"parameterized-agent-{agent_index + 1}"
        parent_agent_id = _parent_for(agent_index, target_agents, config.active_subagents)
        for sequence in range(target_length):
            source = source_stream[sequence % len(source_stream)]
            messages = source.messages
            if config.prompt_tokens is not None:
                prompt = counter.construct_text(config.prompt_tokens, "agenttrace replay context")
                actual = counter.count(prompt)
                if actual != config.prompt_tokens:
                    raise ValueError(
                        f"tokenizer constructed {actual} tokens, expected {config.prompt_tokens}"
                    )
                messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
            offset = source.recorded_submission_offset_seconds
            if config.arrival_rate is not None:
                offset = global_index / config.arrival_rate
            delay = (
                config.tool_wait_seconds
                if config.tool_wait_seconds is not None
                else source.recorded_inter_request_seconds
            )
            transformation = {
                "seed": config.seed,
                "prompt_tokens": config.prompt_tokens,
                "output_tokens": config.output_tokens,
                "concurrent_agents": target_agents,
                "arrival_rate": config.arrival_rate,
                "tool_wait_seconds": config.tool_wait_seconds,
                "active_subagents": config.active_subagents,
                "execution_length": target_length,
            }
            transformed.append(
                source.model_copy(
                    update={
                        "source_request_id": (
                            f"{source.source_request_id}:variant:{agent_index}:{sequence}"
                        ),
                        "agent_id": agent_id,
                        "parent_agent_id": parent_agent_id,
                        "sequence_number": sequence,
                        "messages": messages,
                        "content_token_count": config.prompt_tokens
                        if config.prompt_tokens is not None
                        else source.content_token_count,
                        "recorded_submission_offset_seconds": offset,
                        "recorded_inter_request_seconds": delay,
                        "expected_output_tokens": config.output_tokens
                        or source.expected_output_tokens,
                        "transformation": transformation,
                    }
                )
            )
            global_index += 1
    return transformed


def _parent_for(agent_index: int, total: int, active_subagents: int | None) -> str | None:
    subagents = min(active_subagents or 0, max(0, total - 1))
    if subagents and agent_index >= total - subagents:
        return "parameterized-agent-1"
    return None


def write_transformation_manifest(
    path: Path,
    *,
    source_trace: Path,
    source_trace_id: str,
    config: ReplayConfig,
    requests: list[WorkloadRequest],
) -> None:
    write_manifest(
        path,
        {
            "manifest_version": "1.0",
            "kind": "parameterized_workload",
            "source_trace": str(source_trace),
            "source_trace_id": source_trace_id,
            "random_seed": config.seed,
            "transformation": config.model_dump(mode="json", exclude={"endpoint"}),
            "resulting_requests": [
                {
                    "source_request_id": request.source_request_id,
                    "agent_id": request.agent_id,
                    "parent_agent_id": request.parent_agent_id,
                    "sequence_number": request.sequence_number,
                    "expected_output_tokens": request.expected_output_tokens,
                    "content_tokens": request.content_token_count,
                }
                for request in requests
            ],
        },
    )
