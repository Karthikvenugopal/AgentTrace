"""Core identifiers and execution models used across subsystem boundaries."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def new_id(prefix: str) -> str:
    """Create a sortable-enough opaque identifier without embedding user data."""

    return f"{prefix}_{uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(UTC)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RequestStatus(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ToolStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    TIMEOUT = "timeout"


class ExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    LIMIT_REACHED = "limit_reached"
    CANCELLED = "cancelled"


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(FrozenModel):
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class ToolRequest(FrozenModel):
    id: str = Field(default_factory=lambda: new_id("tool"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(FrozenModel):
    tool_call_id: str
    name: str
    status: ToolStatus
    output: str
    truncated: bool = False
    started_at: datetime
    completed_at: datetime
    duration_seconds: float = Field(ge=0)


class AgentOutcome(FrozenModel):
    status: ExecutionStatus
    summary: str
    iterations: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    started_at: datetime
    completed_at: datetime
