"""Inference client contracts independent of a serving implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agenttrace.models import ChatMessage


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    seed: int | None = None
    stream: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)

    def openai_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    key: value
                    for key, value in message.model_dump(mode="json").items()
                    if value is not None
                }
                for message in self.messages
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.stream:
            payload["stream_options"] = {"include_usage": True}
        return payload


@dataclass(frozen=True)
class StreamObservation:
    elapsed_seconds: float
    text: str
    estimated_tokens: int


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    content: str
    input_tokens: int | None
    output_tokens: int | None
    elapsed_seconds: float
    ttft_seconds: float | None
    chunks: list[StreamObservation] = field(default_factory=list)
    server_request_id: str | None = None
    finish_reason: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class InferenceError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class InferenceClient(Protocol):
    async def complete(self, request: InferenceRequest) -> InferenceResponse: ...

    async def close(self) -> None: ...
