"""Deterministic benchmark-matrix expansion and bias-reducing ordering."""

from __future__ import annotations

import hashlib
import itertools
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict

from agenttrace.config import BenchmarkConfig


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    context_tokens: int
    concurrent_agents: int
    pattern: Literal["sequential", "concurrent", "parent_subagents"]
    workload_type: Literal["recorded", "synthetic", "parameterized"]
    replay_mode: Literal["open_loop", "closed_loop"]


def expand_matrix(config: BenchmarkConfig) -> list[BenchmarkCase]:
    cases = []
    combinations = itertools.product(
        config.matrix.context_tokens,
        config.matrix.concurrent_agents,
        config.matrix.patterns,
        config.matrix.workload_types,
        config.matrix.replay_modes,
    )
    for context, agents, pattern, workload_type, replay_mode in combinations:
        identity = (
            f"ctx={context}|agents={agents}|pattern={pattern}|"
            f"workload={workload_type}|mode={replay_mode}"
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:10]
        cases.append(
            BenchmarkCase(
                case_id=f"case-{digest}",
                context_tokens=context,
                concurrent_agents=agents,
                pattern=pattern,
                workload_type=workload_type,
                replay_mode=replay_mode,
            )
        )
    return cases


def ordered_cases(config: BenchmarkConfig, repetition: int) -> list[BenchmarkCase]:
    cases = expand_matrix(config)
    if config.randomize_order:
        random.Random(config.seed + repetition).shuffle(cases)
    if config.rotate_order_each_repetition and cases:
        shift = repetition % len(cases)
        cases = cases[shift:] + cases[:shift]
    return cases
