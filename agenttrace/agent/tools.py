"""Validated, bounded repository tools exposed to the coding agent."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agenttrace.agent.workspace import RepositoryWorkspace
from agenttrace.models import ToolStatus


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class ToolExecution:
    status: ToolStatus
    output: str
    truncated: bool = False


class AgentTool(Protocol):
    name: str
    description: str
    arguments_model: type[ToolArguments]

    async def execute(self, workspace: RepositoryWorkspace, arguments: ToolArguments) -> ToolExecution:
        ...

    def specification(self) -> dict[str, Any]: ...


class ToolBase:
    name: str
    description: str
    arguments_model: type[ToolArguments]

    def specification(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.arguments_model.model_json_schema(),
        }


class ReadFileArguments(ToolArguments):
    path: str
    start_line: int = Field(default=1, gt=0)
    max_lines: int = Field(default=400, gt=0, le=2000)


class ReadFileTool(ToolBase):
    name = "read_file"
    description = "Read a bounded line range from a UTF-8 repository file."
    arguments_model = ReadFileArguments

    async def execute(self, workspace: RepositoryWorkspace, arguments: ToolArguments) -> ToolExecution:
        args = ReadFileArguments.model_validate(arguments)
        path = workspace.resolve(args.path)
        if not path.is_file():
            return ToolExecution(ToolStatus.FAILED, f"not a file: {args.path}")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            return ToolExecution(ToolStatus.FAILED, f"binary or non-UTF-8 file: {args.path}")
        selected = lines[args.start_line - 1 : args.start_line - 1 + args.max_lines]
        numbered = "\n".join(
            f"{number}: {line}" for number, line in enumerate(selected, start=args.start_line)
        )
        truncated = args.start_line - 1 + args.max_lines < len(lines)
        return ToolExecution(ToolStatus.SUCCEEDED, numbered, truncated)


class ListDirectoryArguments(ToolArguments):
    path: str = "."
    max_entries: int = Field(default=500, gt=0, le=5000)


class ListDirectoryTool(ToolBase):
    name = "list_directory"
    description = "List repository-relative directory entries without traversing .git."
    arguments_model = ListDirectoryArguments

    async def execute(self, workspace: RepositoryWorkspace, arguments: ToolArguments) -> ToolExecution:
        args = ListDirectoryArguments.model_validate(arguments)
        path = workspace.resolve(args.path)
        if not path.is_dir():
            return ToolExecution(ToolStatus.FAILED, f"not a directory: {args.path}")
        entries = sorted(
            item.relative_to(workspace.root).as_posix()
            + ("/" if item.is_dir() else "")
            for item in path.iterdir()
            if item.name != ".git"
        )
        return ToolExecution(
            ToolStatus.SUCCEEDED,
            "\n".join(entries[: args.max_entries]),
            len(entries) > args.max_entries,
        )


class SearchArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=500)
    path: str = "."
    glob: str = "*"
    max_matches: int = Field(default=100, gt=0, le=1000)


class SearchTool(ToolBase):
    name = "search"
    description = "Search UTF-8 repository files for a literal string."
    arguments_model = SearchArguments

    async def execute(self, workspace: RepositoryWorkspace, arguments: ToolArguments) -> ToolExecution:
        args = SearchArguments.model_validate(arguments)
        root = workspace.resolve(args.path)
        matches: list[str] = []
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or ".git" in path.parts or not fnmatch.fnmatch(path.name, args.glob):
                continue
            try:
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if args.query in line:
                        matches.append(
                            f"{path.relative_to(workspace.root).as_posix()}:{line_number}:{line}"
                        )
                        if len(matches) >= args.max_matches:
                            return ToolExecution(ToolStatus.SUCCEEDED, "\n".join(matches), True)
            except (UnicodeDecodeError, OSError):
                continue
        return ToolExecution(ToolStatus.SUCCEEDED, "\n".join(matches))


class ToolRegistry:
    def __init__(self, tools: list[AgentTool]) -> None:
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("tool names must be unique")

    @property
    def specifications(self) -> list[dict[str, Any]]:
        return [tool.specification() for tool in self._tools.values()]

    async def execute(
        self, name: str, workspace: RepositoryWorkspace, raw_arguments: dict[str, Any]
    ) -> ToolExecution:
        tool = self._tools.get(name)
        if tool is None:
            return ToolExecution(ToolStatus.DENIED, f"unknown tool: {name}")
        try:
            arguments = tool.arguments_model.model_validate(raw_arguments)
            return await tool.execute(workspace, arguments)
        except (ValidationError, ValueError, OSError) as exc:
            return ToolExecution(ToolStatus.FAILED, str(exc))


def inspection_tools() -> list[AgentTool]:
    return [ReadFileTool(), ListDirectoryTool(), SearchTool()]
