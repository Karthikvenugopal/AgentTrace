"""Trace import/export utilities and non-performance summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from agenttrace.models import RequestStatus
from agenttrace.tracing.schema import AgentRecord, RequestRecord, ToolCallRecord, TraceHeader
from agenttrace.tracing.storage import iter_trace, write_manifest
from agenttrace.tracing.validation import validate_file


@dataclass(frozen=True)
class TraceSummary:
    experiment_id: str
    trace_id: str
    source: str
    agent_count: int
    request_count: int
    successful_requests: int
    failed_requests: int
    tool_call_count: int
    input_tokens: int
    output_tokens: int
    earliest_submission: str | None
    latest_completion: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_trace(path: Path) -> TraceSummary:
    report = validate_file(path)
    records = report.require_valid()
    header = next(record for record in records if isinstance(record, TraceHeader))
    requests = [record for record in records if isinstance(record, RequestRecord)]
    agents = [record for record in records if isinstance(record, AgentRecord)]
    tools = [record for record in records if isinstance(record, ToolCallRecord)]
    return TraceSummary(
        experiment_id=header.experiment_id,
        trace_id=header.trace_id,
        source=header.source,
        agent_count=len(agents),
        request_count=len(requests),
        successful_requests=sum(req.status == RequestStatus.SUCCEEDED for req in requests),
        failed_requests=sum(req.status != RequestStatus.SUCCEEDED for req in requests),
        tool_call_count=len(tools),
        input_tokens=sum(req.input_tokens for req in requests),
        output_tokens=sum(req.output_tokens for req in requests),
        earliest_submission=(
            min(req.submitted_at for req in requests).isoformat() if requests else None
        ),
        latest_completion=(
            max(req.completed_at for req in requests).isoformat() if requests else None
        ),
    )


def export_validated_trace(source: Path, destination: Path) -> None:
    """Normalize a valid trace to canonical JSONL."""

    report = validate_file(source)
    records = report.require_valid()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(record.model_dump_json(exclude_none=True) + "\n" for record in records),
        encoding="utf-8",
    )


def create_replay_manifest(trace_path: Path, destination: Path) -> None:
    header = next(iter_trace(trace_path))
    if not isinstance(header, TraceHeader):
        raise ValueError("trace does not start with a header")
    summary = summarize_trace(trace_path)
    write_manifest(
        destination,
        {
            "manifest_version": "1.0",
            "source_trace": str(trace_path),
            "source_trace_id": header.trace_id,
            "source_kind": header.source,
            "experiment_id": header.experiment_id,
            "model": header.environment.model,
            "tokenizer": header.environment.tokenizer,
            "request_count": summary.request_count,
            "content_available": header.content_mode != "metadata_only",
        },
    )
