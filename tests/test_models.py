from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agenttrace.models import ToolResult, ToolStatus, new_id


def test_identifiers_have_requested_prefix_and_are_unique() -> None:
    first = new_id("req")
    second = new_id("req")
    assert first.startswith("req_")
    assert first != second


def test_tool_duration_cannot_be_negative() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError):
        ToolResult(
            tool_call_id="tool_1",
            name="read_file",
            status=ToolStatus.SUCCEEDED,
            output="ok",
            started_at=now,
            completed_at=now,
            duration_seconds=-1,
        )
