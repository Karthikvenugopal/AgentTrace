"""Sequential and concurrent execution of independent and related agents."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from agenttrace.agent.runner import CodingAgent
from agenttrace.agent.workspace import IsolatedRepository, RepositoryWorkspace
from agenttrace.config import AgentConfig, EndpointConfig
from agenttrace.instrumentation.timing import ActiveRequestCounter
from agenttrace.models import AgentOutcome
from agenttrace.serving.client import InferenceClient
from agenttrace.tracing.session import start_agent_trace

ClientFactory = Callable[[EndpointConfig], InferenceClient]


async def run_agent_group(
    configs: list[AgentConfig],
    client_factory: ClientFactory,
    *,
    concurrent: bool = True,
) -> list[AgentOutcome]:
    if not configs:
        return []
    _validate_group(configs)
    trace_id, writer = start_agent_trace(configs[0])
    active_requests = ActiveRequestCounter()

    async def run_one(config: AgentConfig) -> AgentOutcome:
        client = client_factory(config.endpoint)
        try:
            if config.workspace.isolated_copy:
                with IsolatedRepository(config.workspace.root) as workspace:
                    return await CodingAgent(
                        config,
                        workspace,
                        client,
                        writer,
                        trace_id,
                        active_requests=active_requests,
                    ).run()
            workspace = RepositoryWorkspace(
                config.workspace.root, writable=config.workspace.writable
            )
            return await CodingAgent(
                config,
                workspace,
                client,
                writer,
                trace_id,
                active_requests=active_requests,
            ).run()
        finally:
            await client.close()

    try:
        if concurrent:
            return list(await asyncio.gather(*(run_one(config) for config in configs)))
        outcomes = []
        for config in configs:
            outcomes.append(await run_one(config))
        return outcomes
    finally:
        writer.close()


def _validate_group(configs: list[AgentConfig]) -> None:
    experiment_ids = {config.experiment_id for config in configs}
    outputs = {config.trace.output.resolve() for config in configs}
    agent_ids = {config.agent_id for config in configs}
    if len(experiment_ids) != 1:
        raise ValueError("all agents in a group must share an experiment_id")
    if len(outputs) != 1:
        raise ValueError("all agents in a group must share one trace output")
    if len(agent_ids) != len(configs):
        raise ValueError("agent IDs must be unique")
    for config in configs:
        if config.parent_agent_id and config.parent_agent_id not in agent_ids:
            raise ValueError(
                f"agent {config.agent_id} references parent outside group: {config.parent_agent_id}"
            )
