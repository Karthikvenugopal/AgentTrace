import shutil
from pathlib import Path

import pytest

from agenttrace.agent.orchestration import run_agent_group
from agenttrace.config import AgentConfig
from agenttrace.models import ExecutionStatus
from agenttrace.serving.client import InferenceRequest, InferenceResponse
from agenttrace.tracing.schema import RequestRecord, ToolCallRecord
from agenttrace.tracing.storage import iter_trace


class CodingScriptClient:
    def __init__(self) -> None:
        self.step = 0

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        actions = [
            '{"action":"tool","tool":"read_file","arguments":{"path":"calc.py"}}',
            (
                '{"action":"tool","tool":"replace_text","arguments":{"path":"calc.py",'
                '"old":"return left - right","new":"return left + right"}}'
            ),
            '{"action":"tool","tool":"run_command","arguments":{"argv":["pytest","-q"]}}',
            '{"action":"finish","summary":"fixed add and pytest passed"}',
        ]
        content = actions[self.step]
        if self.step == 3:
            assert "exit_code=0" in request.messages[-1].content
        self.step += 1
        return InferenceResponse(
            request_id=request.request_id,
            content=content,
            input_tokens=None,
            output_tokens=None,
            elapsed_seconds=0.001,
            ttft_seconds=0.0005,
        )

    async def close(self) -> None:
        return None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_inspects_edits_tests_and_uses_result(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "examples" / "fixtures" / "tiny_math"
    workspace = tmp_path / "repo"
    shutil.copytree(fixture, workspace)
    trace = tmp_path / "agent.jsonl"
    config = AgentConfig(
        experiment_id="coding-integration",
        task="fix add and run tests",
        workspace={
            "root": workspace,
            "isolated_copy": False,
            "allowed_commands": ["pytest"],
        },
        trace={"output": trace, "content_mode": "metadata_only"},
    )
    outcomes = await run_agent_group([config], lambda _: CodingScriptClient())
    assert outcomes[0].status == ExecutionStatus.SUCCEEDED
    assert "left + right" in (workspace / "calc.py").read_text()
    records = list(iter_trace(trace))
    calls = [record for record in records if isinstance(record, ToolCallRecord)]
    requests = [record for record in records if isinstance(record, RequestRecord)]
    assert [call.tool_name for call in calls] == ["read_file", "replace_text", "run_command"]
    assert all(call.request_id in {request.request_id for request in requests} for call in calls)
    assert requests[-1].context_growth_tokens > 0
