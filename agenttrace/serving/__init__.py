"""OpenAI-compatible serving integrations."""

from agenttrace.serving.client import (
    InferenceClient,
    InferenceError,
    InferenceRequest,
    InferenceResponse,
    StreamObservation,
)

__all__ = [
    "InferenceClient",
    "InferenceError",
    "InferenceRequest",
    "InferenceResponse",
    "StreamObservation",
]
