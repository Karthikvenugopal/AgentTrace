import asyncio
from collections import defaultdict
from pathlib import Path

import pytest

from agenttrace.agent.orchestration import run_agent_group
from agenttrace.config import AgentConfig
from agenttrace.models import ExecutionStatus
from agenttrace.serving.client import InferenceRequest, InferenceResponse
from agenttrace.tracing.schema import AgentRecord, RequestRecord
from agenttrace.tracing.storage import iter_trace


class ScriptedClient:
    def __init__(self) -> None:
        self.calls: defaultdict[str, int] = defaultdict(int)

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        agent_marker = request.messages[1].content
        self.calls[agent_marker] += 1
        await asyncio.sleep(0.01)
        if self.calls[agent_marker] == 1:
            content = (
                '{"action":"tool","tool":"write_file","arguments":'
                f'{{"path":"{agent_marker}.txt","content":"done"}}}}'
            )
        else:
            content = '{"action":"finish","summary":"task complete"}'
        return InferenceResponse(
            request_id=request.request_id,
            content=content,
            input_tokens=None,
            output_tokens=None,
            elapsed_seconds=0.01,
            ttft_seconds=0.005,
        )

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_concurrent_parent_child_agents_have_independent_streams(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    base = {
        "experiment_id": "multi",
        "task": "parent",
        "workspace": {"root": tmp_path, "isolated_copy": False},
        "trace": {"output": trace, "content_mode": "metadata_only"},
    }
    parent = AgentConfig(**base, agent_id="parent")
    child = AgentConfig(**{**base, "task": "child"}, agent_id="child", parent_agent_id="parent")
    outcomes = await run_agent_group([parent, child], lambda _: ScriptedClient())
    assert all(outcome.status == ExecutionStatus.SUCCEEDED for outcome in outcomes)
    records = list(iter_trace(trace))
    agents = [record for record in records if isinstance(record, AgentRecord)]
    requests = [record for record in records if isinstance(record, RequestRecord)]
    assert {agent.agent_id for agent in agents} == {"parent", "child"}
    assert [request.sequence_number for request in requests if request.agent_id == "child"] == [
        0,
        1,
    ]
    assert max(request.concurrency_at_submission for request in requests) == 2
    assert (tmp_path / "parent.txt").read_text() == "done"
    assert (tmp_path / "child.txt").read_text() == "done"
