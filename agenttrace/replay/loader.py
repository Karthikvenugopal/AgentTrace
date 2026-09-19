"""Convert recorded trace requests into replayable payloads."""

from __future__ import annotations

from pathlib import Path

from agenttrace.models import ChatMessage, MessageRole
from agenttrace.replay.models import WorkloadRequest
from agenttrace.tracing.schema import RequestRecord, TraceHeader
from agenttrace.tracing.storage import iter_trace


def load_workload(path: Path) -> list[WorkloadRequest]:
    records = list(iter_trace(path))
    header = next((record for record in records if isinstance(record, TraceHeader)), None)
    if header is None:
        raise ValueError("trace has no header")
    requests = [record for record in records if isinstance(record, RequestRecord)]
    if not requests:
        return []
    origin = min(request.submitted_at for request in requests)
    workload: list[WorkloadRequest] = []
    for request in requests:
        messages = _messages_for(request)
        workload.append(
            WorkloadRequest(
                source_trace_id=header.trace_id,
                source_request_id=request.request_id,
                agent_id=request.agent_id,
                sequence_number=request.sequence_number,
                model=request.model,
                messages=messages,
                recorded_submission_offset_seconds=(request.submitted_at - origin).total_seconds(),
                recorded_inter_request_seconds=request.time_since_previous_request_seconds or 0.0,
                expected_output_tokens=max(1, request.output_tokens),
                sampling_parameters=request.sampling_parameters,
            )
        )
    return workload


def _messages_for(request: RequestRecord) -> list[ChatMessage]:
    if request.messages:
        return [ChatMessage.model_validate(message) for message in request.messages]
    if request.prompt and request.prompt != "[REDACTED]":
        return [ChatMessage(role=MessageRole.USER, content=request.prompt)]
    raise ValueError(
        f"request {request.request_id} has no replayable prompt content; "
        "collect full traces or use parameterized replay"
    )
