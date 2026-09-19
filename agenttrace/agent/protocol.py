"""Strict JSON action protocol between the coding agent and an LLM."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from agenttrace.models import new_id


class ActionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolAction(ActionBase):
    action: Literal["tool"] = "tool"
    tool_call_id: str = Field(default_factory=lambda: new_id("tool"))
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class FinishAction(ActionBase):
    action: Literal["finish"] = "finish"
    summary: str = Field(min_length=1, max_length=10_000)


AgentAction = Annotated[Union[ToolAction, FinishAction], Field(discriminator="action")]
ACTION_ADAPTER = TypeAdapter(AgentAction)


def parse_action(content: str) -> AgentAction:
    """Parse one action, tolerating only a surrounding Markdown JSON fence."""

    stripped = content.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[len("```json") : -3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
    return ACTION_ADAPTER.validate_python(json.loads(stripped))


def protocol_instruction(tool_specs: list[dict[str, Any]]) -> str:
    tools = json.dumps(tool_specs, sort_keys=True)
    return (
        "Respond with exactly one JSON object. To use a tool: "
        '{"action":"tool","tool":"NAME","arguments":{...}}. '
        'When the task is complete: {"action":"finish","summary":"..."}. '
        f"Available tools: {tools}"
    )
