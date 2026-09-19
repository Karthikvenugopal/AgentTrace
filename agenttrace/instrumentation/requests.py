"""Per-agent inference request sequence and context-growth state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextObservation:
    sequence_number: int
    prompt_tokens: int
    growth_tokens: int
    time_since_previous_request_seconds: float | None


class RequestSequenceTracker:
    def __init__(self) -> None:
        self._next_sequence = 0
        self._previous_prompt_tokens: int | None = None
        self._previous_submission: float | None = None

    def observe(self, *, prompt_tokens: int, submitted_monotonic: float) -> ContextObservation:
        growth = (
            0
            if self._previous_prompt_tokens is None
            else prompt_tokens - self._previous_prompt_tokens
        )
        delay = (
            None
            if self._previous_submission is None
            else max(0.0, submitted_monotonic - self._previous_submission)
        )
        observation = ContextObservation(
            sequence_number=self._next_sequence,
            prompt_tokens=prompt_tokens,
            growth_tokens=growth,
            time_since_previous_request_seconds=delay,
        )
        self._next_sequence += 1
        self._previous_prompt_tokens = prompt_tokens
        self._previous_submission = submitted_monotonic
        return observation
