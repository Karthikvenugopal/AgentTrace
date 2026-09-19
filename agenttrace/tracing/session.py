"""Trace-session construction without leaking credentials or repository content."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from agenttrace import __version__
from agenttrace.config import AgentConfig
from agenttrace.models import new_id, utc_now
from agenttrace.tracing.schema import EnvironmentInfo, TraceHeader
from agenttrace.tracing.storage import TraceWriter


def agent_execution_metadata(config: AgentConfig) -> dict[str, object]:
    """Return reproducibility fields, intentionally excluding secrets and task text."""

    return {
        "agent_id": config.agent_id,
        "parent_agent_id": config.parent_agent_id,
        "model": config.endpoint.model,
        "endpoint_origin": str(config.endpoint.base_url).split("/v1", maxsplit=1)[0],
        "sampling": config.sampling.model_dump(mode="json"),
        "limits": config.limits.model_dump(mode="json"),
        "allowed_commands": sorted(config.workspace.allowed_commands),
        "workspace_name": Path(config.workspace.root).name,
    }


def start_agent_trace(config: AgentConfig) -> tuple[str, TraceWriter]:
    trace_id = new_id("trace")
    writer = TraceWriter(
        config.trace.output,
        flush_each_record=config.trace.flush_each_record,
        mode="w",
    ).open()
    writer.write(
        TraceHeader(
            experiment_id=config.experiment_id,
            trace_id=trace_id,
            created_at=utc_now(),
            source="agent",
            content_mode=config.trace.content_mode,
            environment=EnvironmentInfo(
                python_version=platform.python_version(),
                platform=platform.platform(),
                agenttrace_version=__version__,
                model=config.endpoint.model,
            ),
            execution_config=agent_execution_metadata(config),
        )
    )
    return trace_id, writer
