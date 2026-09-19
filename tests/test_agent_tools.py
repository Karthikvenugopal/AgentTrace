from pathlib import Path

import pytest

from agenttrace.agent.tools import RunCommandTool, ToolRegistry
from agenttrace.agent.workspace import RepositoryWorkspace
from agenttrace.models import ToolStatus


@pytest.mark.asyncio
async def test_command_tool_enforces_allowlist_and_streaming_output_cap(tmp_path: Path) -> None:
    workspace = RepositoryWorkspace(tmp_path)
    tool = RunCommandTool(
        ["python3"],
        timeout_seconds=5,
        max_output_bytes=32,
        cpu_seconds=2,
        memory_mb=256,
    )
    registry = ToolRegistry([tool])
    denied = await registry.execute("run_command", workspace, {"argv": ["sh", "-c", "echo unsafe"]})
    assert denied.status == ToolStatus.DENIED
    bounded = await registry.execute(
        "run_command",
        workspace,
        {"argv": ["python3", "-c", "print('x' * 10000)"]},
    )
    assert bounded.status == ToolStatus.SUCCEEDED
    assert bounded.truncated
    assert len(bounded.output) < 100
