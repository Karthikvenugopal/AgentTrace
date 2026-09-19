import pytest
from pydantic import ValidationError

from agenttrace.agent.protocol import FinishAction, ToolAction, parse_action


def test_parse_tool_action_from_json_fence() -> None:
    action = parse_action(
        '```json\n{"action":"tool","tool":"read_file","arguments":{"path":"a.py"}}\n```'
    )
    assert isinstance(action, ToolAction)
    assert action.tool_call_id.startswith("tool_")


def test_protocol_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        parse_action('{"action":"finish","summary":"done","secret":"x"}')


def test_finish_requires_summary() -> None:
    action = parse_action('{"action":"finish","summary":"validated tests"}')
    assert isinstance(action, FinishAction)
