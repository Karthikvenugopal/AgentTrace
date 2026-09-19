from datetime import UTC, datetime, timedelta
from pathlib import Path

from agenttrace.models import ExecutionStatus, RequestStatus
from agenttrace.tracing.schema import (
    AgentRecord,
    EnvironmentInfo,
    OutcomeRecord,
    PromptTokenBreakdown,
    RequestRecord,
    TraceHeader,
)
from agenttrace.tracing.storage import TraceWriter, iter_trace
from agenttrace.tracing.validation import validate_file, validate_records


def _records() -> list[object]:
    now = datetime.now(UTC)
    common = {"experiment_id": "exp", "trace_id": "trace_1"}
    return [
        TraceHeader(
            **common,
            created_at=now,
            source="agent",
            content_mode="metadata_only",
            environment=EnvironmentInfo(
                python_version="3.11", platform="test", agenttrace_version="0.1", model="mock"
            ),
        ),
        AgentRecord(**common, agent_id="a", started_at=now, task_hash="sha256:test"),
        RequestRecord(
            **common,
            agent_id="a",
            request_id="req_1",
            sequence_number=0,
            model="mock",
            created_at=now,
            submitted_at=now,
            completed_at=now + timedelta(seconds=0.2),
            elapsed_seconds=0.2,
            monotonic_started=1.0,
            monotonic_completed=1.2,
            input_tokens=10,
            output_tokens=2,
            prompt_breakdown=PromptTokenBreakdown(system=4, user=6),
            tokenizer="test",
            token_count_method="fixture",
            status=RequestStatus.SUCCEEDED,
            sampling_parameters={},
            concurrency_at_submission=1,
            stream_chunk_arrivals_seconds=[0.1, 0.2],
            stream_chunk_token_counts=[1, 1],
        ),
        OutcomeRecord(
            **common,
            agent_id="a",
            status=ExecutionStatus.SUCCEEDED,
            summary="done",
            iterations=1,
            total_input_tokens=10,
            total_output_tokens=2,
            started_at=now,
            completed_at=now + timedelta(seconds=0.3),
        ),
    ]


def test_streaming_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    records = _records()
    with TraceWriter(path) as writer:
        for record in records:
            writer.write(record)  # type: ignore[arg-type]
    loaded = list(iter_trace(path))
    assert [type(record) for record in loaded] == [type(record) for record in records]
    assert validate_file(path).valid


def test_missing_parent_is_detected() -> None:
    records = _records()
    agent = records[1]
    assert isinstance(agent, AgentRecord)
    records[1] = agent.model_copy(update={"parent_agent_id": "missing"})
    report = validate_records(records)  # type: ignore[arg-type]
    assert not report.valid
    assert "missing parent" in report.errors[0]


def test_duplicate_request_id_is_detected() -> None:
    records = _records()
    request = records[2]
    assert isinstance(request, RequestRecord)
    records.insert(3, request.model_copy(update={"sequence_number": 1}))
    report = validate_records(records)  # type: ignore[arg-type]
    assert any("duplicate request_id" in error for error in report.errors)
