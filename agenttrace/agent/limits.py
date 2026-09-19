"""Deterministic agent execution budget checks."""

from __future__ import annotations

from dataclasses import dataclass

from agenttrace.config import AgentLimits


@dataclass
class ExecutionBudget:
    limits: AgentLimits
    started_monotonic: float
    input_tokens: int = 0
    output_tokens: int = 0

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def stop_reason(self, *, now_monotonic: float, next_iteration: int) -> str | None:
        if next_iteration >= self.limits.max_iterations:
            return "iteration limit reached"
        if now_monotonic - self.started_monotonic >= self.limits.max_wall_time_seconds:
            return "wall-time limit reached"
        if self.input_tokens + self.output_tokens >= self.limits.max_total_tokens:
            return "token budget reached"
        return None
