from pathlib import Path

import pytest

from agenttrace.agent.workspace import IsolatedRepository, RepositoryWorkspace, WorkspaceViolation


def test_workspace_rejects_traversal_and_absolute_paths(tmp_path: Path) -> None:
    workspace = RepositoryWorkspace(tmp_path)
    with pytest.raises(WorkspaceViolation, match="escapes"):
        workspace.resolve("../secret", must_exist=False)
    with pytest.raises(WorkspaceViolation, match="absolute"):
        workspace.resolve("/etc/passwd")


def test_isolated_repository_excludes_git_and_secret_files(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_text("answer = 42\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=x\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    with IsolatedRepository(tmp_path) as workspace:
        assert (workspace.root / "source.py").exists()
        assert not (workspace.root / ".env").exists()
        assert not (workspace.root / ".git").exists()
