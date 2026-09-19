from pathlib import Path

import pytest
from pydantic import ValidationError

from agenttrace.config import AgentConfig, ReplayConfig, load_config


def test_agent_config_rejects_self_parent(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="own parent"):
        AgentConfig(
            experiment_id="exp",
            agent_id="same",
            parent_agent_id="same",
            task="fix it",
            workspace={"root": tmp_path},
        )


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "replay.yaml"
    path.write_text("mode: closed_loop\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path, ReplayConfig)
