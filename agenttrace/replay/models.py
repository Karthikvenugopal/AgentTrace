"""Models for replay inputs and per-attempt observations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agenttrace.models import ChatMessage


class ReplayModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkloadRequest(ReplayModel):
    source_trace_id: str
    source_request_id: str
    agent_id: str
    sequence_number: int = Field(ge=0)
    model: str
    messages: list[ChatMessage]
    recorded_submission_offset_seconds: float = Field(ge=0)
    recorded_inter_request_seconds: float = Field(ge=0)
    expected_output_tokens: int = Field(gt=0)
    sampling_parameters: dict[str, Any] = Field(default_factory=dict)


class ReplayAttempt(ReplayModel):
    replay_session_id: str
    source_trace_id: str
    source_request_id: str
    replay_request_id: str
    agent_id: str
    sequence_number: int
    attempt_number: int = Field(ge=1)
    mode: Literal["open_loop", "closed_loop", "parameterized"]
    scheduled_offset_seconds: float = Field(ge=0)
    submitted_offset_seconds: float = Field(ge=0)
    client_scheduling_delay_seconds: float = Field(ge=0)
    submitted_at: datetime
    completed_at: datetime
    status: Literal["succeeded", "failed", "timeout"]
    latency_seconds: float = Field(ge=0)
    ttft_seconds: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    stream_chunk_arrivals_seconds: list[float] = Field(default_factory=list)
    stream_chunk_token_counts: list[int] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


class ReplaySessionResult(ReplayModel):
    replay_session_id: str
    source_trace_id: str
    mode: Literal["open_loop", "closed_loop", "parameterized"]
    started_at: datetime
    completed_at: datetime
    attempts: list[ReplayAttempt]

    @property
    def successful_attempts(self) -> list[ReplayAttempt]:
        return [attempt for attempt in self.attempts if attempt.status == "succeeded"]
