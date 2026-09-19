"""Versioned, line-oriented trace schema.

Each JSONL line is independently parseable and carries ``schema_version``.  The
first record is a trace header; subsequent records are immutable observations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agenttrace.models import ExecutionStatus, RequestStatus, ToolStatus

SCHEMA_VERSION = "1.0"


class TraceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    experiment_id: str = Field(min_length=1)


class EnvironmentInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    python_version: str
    platform: str
    agenttrace_version: str
    model: str
    tokenizer: str | None = None
    vllm_version: str | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    serving: dict[str, Any] = Field(default_factory=dict)


class TraceHeader(TraceRecord):
    record_type: Literal["header"] = "header"
    trace_id: str
    created_at: datetime
    source: Literal["agent", "synthetic", "transformed", "replay"]
    content_mode: Literal["full", "redacted", "metadata_only"]
    seed: int | None = None
    source_trace_id: str | None = None
    transformation: dict[str, Any] | None = None
    environment: EnvironmentInfo
    execution_config: dict[str, Any] = Field(default_factory=dict)


class AgentRecord(TraceRecord):
    record_type: Literal["agent"] = "agent"
    trace_id: str
    agent_id: str
    parent_agent_id: str | None = None
    started_at: datetime
    task_hash: str

    @model_validator(mode="after")
    def prevent_self_parent(self) -> AgentRecord:
        if self.parent_agent_id == self.agent_id:
            raise ValueError("agent cannot be its own parent")
        return self


class PromptTokenBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: int = Field(default=0, ge=0)
    user: int = Field(default=0, ge=0)
    assistant: int = Field(default=0, ge=0)
    tool: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        return self.system + self.user + self.assistant + self.tool


class RequestRecord(TraceRecord):
    record_type: Literal["request"] = "request"
    trace_id: str
    agent_id: str
    parent_agent_id: str | None = None
    request_id: str
    sequence_number: int = Field(ge=0)
    model: str
    created_at: datetime
    submitted_at: datetime
    completed_at: datetime
    elapsed_seconds: float = Field(ge=0)
    monotonic_started: float = Field(ge=0)
    monotonic_completed: float = Field(ge=0)
    time_since_previous_request_seconds: float | None = Field(default=None, ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    server_input_tokens: int | None = Field(default=None, ge=0)
    server_output_tokens: int | None = Field(default=None, ge=0)
    token_count_discrepancy: int | None = None
    context_window: int | None = Field(default=None, gt=0)
    context_utilization: float | None = Field(default=None, ge=0, le=1)
    prompt_breakdown: PromptTokenBreakdown
    tokenizer: str
    token_count_method: str
    status: RequestStatus
    error_type: str | None = None
    error_message: str | None = None
    sampling_parameters: dict[str, Any]
    concurrency_at_submission: int = Field(ge=1)
    associated_tool_call_ids: list[str] = Field(default_factory=list)
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    first_token_at: datetime | None = None
    ttft_seconds: float | None = Field(default=None, ge=0)
    stream_chunk_arrivals_seconds: list[float] = Field(default_factory=list)
    stream_chunk_token_counts: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_timings(self) -> RequestRecord:
        if self.completed_at < self.submitted_at:
            raise ValueError("request completion precedes submission")
        if self.monotonic_completed < self.monotonic_started:
            raise ValueError("monotonic request completion precedes start")
        if len(self.stream_chunk_arrivals_seconds) != len(self.stream_chunk_token_counts):
            raise ValueError("stream arrival and token-count arrays must have equal lengths")
        if self.status == RequestStatus.SUCCEEDED and self.error_type is not None:
            raise ValueError("successful request cannot carry an error")
        return self

    @field_validator("stream_chunk_arrivals_seconds")
    @classmethod
    def arrivals_are_monotonic(cls, values: list[float]) -> list[float]:
        if any(right < left for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("stream chunk arrivals must be monotonic")
        return values


class ToolCallRecord(TraceRecord):
    record_type: Literal["tool_call"] = "tool_call"
    trace_id: str
    agent_id: str
    request_id: str
    tool_call_id: str
    tool_name: str
    started_at: datetime
    completed_at: datetime
    monotonic_started: float = Field(ge=0)
    monotonic_completed: float = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    result_size_bytes: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    status: ToolStatus
    error: str | None = None

    @model_validator(mode="after")
    def validate_order(self) -> ToolCallRecord:
        if self.completed_at < self.started_at:
            raise ValueError("tool completion precedes start")
        return self


class OutcomeRecord(TraceRecord):
    record_type: Literal["outcome"] = "outcome"
    trace_id: str
    agent_id: str
    status: ExecutionStatus
    summary: str
    iterations: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    started_at: datetime
    completed_at: datetime


AnyTraceRecord = Annotated[
    Union[TraceHeader, AgentRecord, RequestRecord, ToolCallRecord, OutcomeRecord],
    Field(discriminator="record_type"),
]
