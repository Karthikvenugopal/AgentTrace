"""Tool execution instrumentation shared by live agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agenttrace.agent.tools import ToolExecution, ToolRegistry
from agenttrace.agent.workspace import RepositoryWorkspace
from agenttrace.instrumentation.timing import snapshot
from agenttrace.instrumentation.tokens import TokenCounter
from agenttrace.models import ToolStatus
from agenttrace.tracing.schema import ToolCallRecord
from agenttrace.tracing.storage import TraceWriter
from agenttrace.telemetry.tracing import trace_span


@dataclass(frozen=True)
class ToolTraceContext:
    experiment_id: str
    trace_id: str
    agent_id: str
    request_id: str
    tool_call_id: str


async def execute_instrumented_tool(
    *,
    registry: ToolRegistry,
    workspace: RepositoryWorkspace,
    name: str,
    arguments: dict[str, Any],
    context: ToolTraceContext,
    counter: TokenCounter,
    writer: TraceWriter,
) -> ToolExecution:
    started = snapshot()
    try:
        with trace_span(
            "tool.call",
            {
                "agenttrace.experiment_id": context.experiment_id,
                "agenttrace.agent_id": context.agent_id,
                "agenttrace.request_id": context.request_id,
                "agenttrace.tool_name": name,
            },
        ):
            execution = await registry.execute(name, workspace, arguments)
    except Exception as exc:  # defensive boundary: failures must remain trace-visible
        execution = ToolExecution(ToolStatus.FAILED, f"{type(exc).__name__}: {exc}")
    completed = snapshot()
    output = execution.output
    writer.write(
        ToolCallRecord(
            experiment_id=context.experiment_id,
            trace_id=context.trace_id,
            agent_id=context.agent_id,
            request_id=context.request_id,
            tool_call_id=context.tool_call_id,
            tool_name=name,
            started_at=started.wall_time,
            completed_at=completed.wall_time,
            monotonic_started=started.monotonic,
            monotonic_completed=completed.monotonic,
            duration_seconds=completed.monotonic - started.monotonic,
            result_size_bytes=len(output.encode()),
            output_tokens=counter.count(output),
            status=execution.status,
            error=output if execution.status != ToolStatus.SUCCEEDED else None,
        )
    )
    return execution
