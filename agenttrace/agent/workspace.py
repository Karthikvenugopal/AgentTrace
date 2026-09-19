"""Filesystem boundary for coding-agent tools."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import TracebackType


class WorkspaceViolation(ValueError):
    pass


class RepositoryWorkspace:
    """Resolve all model-supplied paths beneath one repository root."""

    def __init__(self, root: Path, *, writable: bool = True) -> None:
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            raise WorkspaceViolation(f"workspace root is not a directory: {resolved}")
        self.root = resolved
        self.writable = writable

    def resolve(self, supplied: str | Path, *, must_exist: bool = True) -> Path:
        relative = Path(supplied)
        if relative.is_absolute():
            raise WorkspaceViolation("absolute paths are not allowed")
        candidate = (self.root / relative).resolve(strict=False)
        if not candidate.is_relative_to(self.root):
            raise WorkspaceViolation(f"path escapes workspace: {supplied}")
        if must_exist and not candidate.exists():
            raise WorkspaceViolation(f"path does not exist: {supplied}")
        if ".git" in candidate.relative_to(self.root).parts:
            raise WorkspaceViolation("direct access to .git is not allowed")
        return candidate

    def require_writable(self) -> None:
        if not self.writable:
            raise WorkspaceViolation("workspace is read-only")


class IsolatedRepository:
    """Temporary copy of a repository, excluding credentials and Git internals."""

    def __init__(self, source: Path) -> None:
        self.source = source.expanduser().resolve()
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self.path: Path | None = None

    def __enter__(self) -> RepositoryWorkspace:
        self._temporary = tempfile.TemporaryDirectory(prefix="agenttrace-workspace-")
        self.path = Path(self._temporary.name) / "repo"
        shutil.copytree(
            self.source,
            self.path,
            symlinks=False,
            ignore=shutil.ignore_patterns(
                ".git", ".env", ".venv", "__pycache__", "*.pem", "*.key", "node_modules"
            ),
        )
        return RepositoryWorkspace(self.path)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
