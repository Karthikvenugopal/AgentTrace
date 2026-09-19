"""Validated, bounded repository tools exposed to the coding agent."""

from __future__ import annotations

import asyncio
import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

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

    async def execute(
        self, workspace: RepositoryWorkspace, arguments: ToolArguments
    ) -> ToolExecution: ...

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

    async def execute(
        self, workspace: RepositoryWorkspace, arguments: ToolArguments
    ) -> ToolExecution:
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

    async def execute(
        self, workspace: RepositoryWorkspace, arguments: ToolArguments
    ) -> ToolExecution:
        args = ListDirectoryArguments.model_validate(arguments)
        path = workspace.resolve(args.path)
        if not path.is_dir():
            return ToolExecution(ToolStatus.FAILED, f"not a directory: {args.path}")
        entries = sorted(
            item.relative_to(workspace.root).as_posix() + ("/" if item.is_dir() else "")
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

    async def execute(
        self, workspace: RepositoryWorkspace, arguments: ToolArguments
    ) -> ToolExecution:
        args = SearchArguments.model_validate(arguments)
        root = workspace.resolve(args.path)
        matches: list[str] = []
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if (
                not path.is_file()
                or ".git" in path.parts
                or not fnmatch.fnmatch(path.name, args.glob)
            ):
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
    return cast(list[AgentTool], [ReadFileTool(), ListDirectoryTool(), SearchTool()])


class WriteFileArguments(ToolArguments):
    path: str
    content: str


class WriteFileTool(ToolBase):
    name = "write_file"
    description = "Create or replace a UTF-8 file inside the repository workspace."
    arguments_model = WriteFileArguments

    async def execute(
        self, workspace: RepositoryWorkspace, arguments: ToolArguments
    ) -> ToolExecution:
        args = WriteFileArguments.model_validate(arguments)
        workspace.require_writable()
        path = workspace.resolve(args.path, must_exist=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.content, encoding="utf-8")
        return ToolExecution(ToolStatus.SUCCEEDED, f"wrote {len(args.content.encode())} bytes")


class ReplaceTextArguments(ToolArguments):
    path: str
    old: str = Field(min_length=1)
    new: str
    expected_replacements: int = Field(default=1, gt=0, le=1000)


class ReplaceTextTool(ToolBase):
    name = "replace_text"
    description = "Replace an exact text fragment with an expected occurrence count."
    arguments_model = ReplaceTextArguments

    async def execute(
        self, workspace: RepositoryWorkspace, arguments: ToolArguments
    ) -> ToolExecution:
        args = ReplaceTextArguments.model_validate(arguments)
        workspace.require_writable()
        path = workspace.resolve(args.path)
        content = path.read_text(encoding="utf-8")
        actual = content.count(args.old)
        if actual != args.expected_replacements:
            return ToolExecution(
                ToolStatus.FAILED,
                f"expected {args.expected_replacements} occurrences, found {actual}",
            )
        path.write_text(content.replace(args.old, args.new), encoding="utf-8")
        return ToolExecution(ToolStatus.SUCCEEDED, f"replaced {actual} occurrence(s)")


class RunCommandArguments(ToolArguments):
    argv: list[str] = Field(min_length=1, max_length=64)
    cwd: str = "."


class RunCommandTool(ToolBase):
    name = "run_command"
    description = "Run one allowlisted command without a shell, with timeout and output limits."
    arguments_model = RunCommandArguments

    def __init__(
        self,
        allowed_commands: list[str],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> None:
        self.allowed_commands = frozenset(allowed_commands)
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    async def execute(
        self, workspace: RepositoryWorkspace, arguments: ToolArguments
    ) -> ToolExecution:
        args = RunCommandArguments.model_validate(arguments)
        executable = Path(args.argv[0]).name
        if executable not in self.allowed_commands or args.argv[0] != executable:
            return ToolExecution(ToolStatus.DENIED, f"command is not allowlisted: {args.argv[0]}")
        cwd = workspace.resolve(args.cwd)
        if not cwd.is_dir():
            return ToolExecution(ToolStatus.FAILED, f"command cwd is not a directory: {args.cwd}")
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "LANG", "LC_ALL", "PYTHONPATH", "VIRTUAL_ENV"}
        }
        process = await asyncio.create_subprocess_exec(
            *args.argv,
            cwd=cwd,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=self.timeout_seconds)
        except TimeoutError:
            process.kill()
            await process.wait()
            return ToolExecution(ToolStatus.TIMEOUT, f"timed out after {self.timeout_seconds}s")
        truncated = len(stdout) > self.max_output_bytes
        output = stdout[: self.max_output_bytes].decode("utf-8", errors="replace")
        status = ToolStatus.SUCCEEDED if process.returncode == 0 else ToolStatus.FAILED
        return ToolExecution(status, f"exit_code={process.returncode}\n{output}", truncated)


def default_tools(
    allowed_commands: list[str], *, timeout_seconds: float, max_output_bytes: int
) -> list[AgentTool]:
    return cast(
        list[AgentTool],
        [
            *inspection_tools(),
            WriteFileTool(),
            ReplaceTextTool(),
            RunCommandTool(
                allowed_commands,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
            ),
        ],
    )
