"""Validated configuration shared by AgentTrace components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, model_validator


class StrictModel(BaseModel):
    """Base configuration that rejects misspelled or obsolete keys."""

    model_config = ConfigDict(extra="forbid")


class EndpointConfig(StrictModel):
    base_url: HttpUrl = HttpUrl("http://127.0.0.1:8000/v1")
    model: str = "agenttrace-mock"
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=1, ge=0, le=10)
    stream: bool = True


class SamplingConfig(StrictModel):
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    max_tokens: int = Field(default=256, gt=0)
    seed: int | None = 7


class AgentLimits(StrictModel):
    max_iterations: int = Field(default=12, gt=0, le=500)
    max_wall_time_seconds: float = Field(default=600.0, gt=0)
    max_total_tokens: int = Field(default=100_000, gt=0)
    command_timeout_seconds: float = Field(default=60.0, gt=0)
    max_tool_output_bytes: int = Field(default=65_536, gt=0)


class WorkspaceConfig(StrictModel):
    root: Path
    allowed_commands: list[str] = Field(
        default_factory=lambda: ["pytest", "python", "python3", "ruff", "mypy"]
    )
    writable: bool = True
    isolated_copy: bool = True


class TraceConfig(StrictModel):
    output: Path = Path("traces/agent-run.jsonl")
    content_mode: Literal["full", "redacted", "metadata_only"] = "redacted"
    redact_patterns: list[str] = Field(default_factory=list)
    flush_each_record: bool = True


class AgentConfig(StrictModel):
    experiment_id: str
    agent_id: str = "agent-1"
    parent_agent_id: str | None = None
    task: str
    endpoint: EndpointConfig = Field(default_factory=EndpointConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    limits: AgentLimits = Field(default_factory=AgentLimits)
    workspace: WorkspaceConfig
    trace: TraceConfig = Field(default_factory=TraceConfig)
    system_prompt: str = (
        "You are a bounded coding agent. Inspect the repository, make the requested "
        "change, validate it, and finish using the structured tool protocol."
    )

    @model_validator(mode="after")
    def parent_is_distinct(self) -> AgentConfig:
        if self.parent_agent_id == self.agent_id:
            raise ValueError("an agent cannot be its own parent")
        return self


class ReplayConfig(StrictModel):
    mode: Literal["open_loop", "closed_loop", "parameterized"] = "closed_loop"
    endpoint: EndpointConfig = Field(default_factory=EndpointConfig)
    time_scale: float = Field(default=1.0, gt=0)
    max_concurrency: int = Field(default=16, gt=0)
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    retry_count: int = Field(default=1, ge=0, le=10)
    seed: int = 7
    prompt_tokens: int | None = Field(default=None, gt=0)
    output_tokens: int | None = Field(default=None, gt=0)
    concurrent_agents: int | None = Field(default=None, gt=0)
    arrival_rate: float | None = Field(default=None, gt=0)
    tool_wait_seconds: float | None = Field(default=None, ge=0)
    active_subagents: int | None = Field(default=None, ge=0)
    execution_length: int | None = Field(default=None, gt=0)


class SyntheticConfig(StrictModel):
    experiment_id: str = "synthetic-demo"
    seed: int = 7
    agents: int = Field(default=2, gt=0)
    requests_per_agent: int = Field(default=4, gt=0)
    initial_prompt_tokens: int = Field(default=128, gt=0)
    prompt_growth_tokens: int = Field(default=64, ge=0)
    output_tokens: int = Field(default=32, gt=0)
    inter_request_seconds: float = Field(default=0.1, ge=0)
    output: Path = Path("traces/synthetic.jsonl")


class BenchmarkAxis(StrictModel):
    context_tokens: list[int] = Field(default_factory=lambda: [128, 512, 2048])
    concurrent_agents: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    patterns: list[Literal["sequential", "concurrent", "parent_subagents"]] = Field(
        default_factory=lambda: cast(
            list[Literal["sequential", "concurrent", "parent_subagents"]],
            ["sequential", "concurrent"],
        )
    )
    workload_types: list[Literal["recorded", "synthetic", "parameterized"]] = Field(
        default_factory=lambda: cast(
            list[Literal["recorded", "synthetic", "parameterized"]], ["parameterized"]
        )
    )
    replay_modes: list[Literal["open_loop", "closed_loop"]] = Field(
        default_factory=lambda: cast(
            list[Literal["open_loop", "closed_loop"]], ["open_loop", "closed_loop"]
        )
    )


class BenchmarkConfig(StrictModel):
    experiment_id: str
    source_trace: Path
    measurement_label: Literal["mock", "real_inference"] = "mock"
    output_dir: Path = Path("results")
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    matrix: BenchmarkAxis = Field(default_factory=BenchmarkAxis)
    warmup_requests: int = Field(default=1, ge=0)
    repetitions: int = Field(default=3, gt=0)
    randomize_order: bool = True
    rotate_order_each_repetition: bool = True
    seed: int = 7
    server_metadata: dict[str, Any] = Field(default_factory=dict)


ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_config(path: Path, model: type[ConfigT]) -> ConfigT:
    """Load a YAML document into a strict Pydantic configuration model."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    return model.model_validate(raw)
