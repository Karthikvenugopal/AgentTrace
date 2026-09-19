"""Whole-trace validation beyond individual Pydantic record checks."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from agenttrace.tracing.schema import (
    AgentRecord,
    AnyTraceRecord,
    OutcomeRecord,
    RequestRecord,
    ToolCallRecord,
    TraceHeader,
)

RECORD_ADAPTER: TypeAdapter[AnyTraceRecord] = TypeAdapter(AnyTraceRecord)


@dataclass
class ValidationReport:
    records: list[AnyTraceRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def require_valid(self) -> list[AnyTraceRecord]:
        if self.errors:
            raise ValueError("invalid trace:\n" + "\n".join(self.errors))
        return self.records


def parse_record(raw: str) -> AnyTraceRecord:
    """Parse and validate one JSON trace record."""

    return RECORD_ADAPTER.validate_json(raw)


def validate_records(records: Iterable[AnyTraceRecord]) -> ValidationReport:
    report = ValidationReport(records=list(records))
    if not report.records:
        report.errors.append("trace is empty")
        return report
    headers = [r for r in report.records if isinstance(r, TraceHeader)]
    if len(headers) != 1:
        report.errors.append(f"expected exactly one header, found {len(headers)}")
        return report
    if report.records[0] is not headers[0]:
        report.errors.append("trace header must be the first record")
    header = headers[0]
    for index, record in enumerate(report.records, start=1):
        if record.experiment_id != header.experiment_id:
            report.errors.append(f"line {index}: inconsistent experiment_id")
        if getattr(record, "trace_id", header.trace_id) != header.trace_id:
            report.errors.append(f"line {index}: inconsistent trace_id")

    agents = {r.agent_id: r for r in report.records if isinstance(r, AgentRecord)}
    for agent in agents.values():
        if agent.parent_agent_id and agent.parent_agent_id not in agents:
            report.errors.append(
                f"agent {agent.agent_id} references missing parent {agent.parent_agent_id}"
            )
    _detect_parent_cycles(agents, report)

    requests = [r for r in report.records if isinstance(r, RequestRecord)]
    request_ids: set[str] = set()
    sequences: dict[str, list[int]] = {}
    previous_submit: dict[str, datetime] = {}
    for request in requests:
        if request.agent_id not in agents:
            report.errors.append(f"request {request.request_id} references unknown agent")
        if request.request_id in request_ids:
            report.errors.append(f"duplicate request_id {request.request_id}")
        request_ids.add(request.request_id)
        sequences.setdefault(request.agent_id, []).append(request.sequence_number)
        prior = previous_submit.get(request.agent_id)
        if prior is not None and request.submitted_at < prior:
            report.errors.append(f"request order regresses for agent {request.agent_id}")
        previous_submit[request.agent_id] = request.submitted_at
    for agent_id, values in sequences.items():
        if values != list(range(len(values))):
            report.errors.append(f"non-contiguous request sequence for agent {agent_id}: {values}")

    tool_ids: set[str] = set()
    for call in (r for r in report.records if isinstance(r, ToolCallRecord)):
        if call.tool_call_id in tool_ids:
            report.errors.append(f"duplicate tool_call_id {call.tool_call_id}")
        tool_ids.add(call.tool_call_id)
        if call.request_id not in request_ids:
            report.errors.append(f"tool {call.tool_call_id} references unknown request")
    linked_tool_ids = {tool_id for req in requests for tool_id in req.associated_tool_call_ids}
    for missing in sorted(linked_tool_ids - tool_ids):
        report.errors.append(f"request references missing tool_call_id {missing}")

    outcome_agents = {r.agent_id for r in report.records if isinstance(r, OutcomeRecord)}
    for agent_id in agents.keys() - outcome_agents:
        report.warnings.append(f"agent {agent_id} has no outcome record")
    return report


def _detect_parent_cycles(agents: dict[str, AgentRecord], report: ValidationReport) -> None:
    for start in agents:
        visited: set[str] = set()
        current: str | None = start
        while current and current in agents:
            if current in visited:
                report.errors.append(f"parent-child cycle includes agent {current}")
                break
            visited.add(current)
            current = agents[current].parent_agent_id


def validate_file(path: Path) -> ValidationReport:
    report = ValidationReport()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                report.records.append(parse_record(line))
            except (ValidationError, json.JSONDecodeError) as exc:
                report.errors.append(f"line {line_number}: {exc}")
    if report.errors:
        return report
    return validate_records(report.records)
