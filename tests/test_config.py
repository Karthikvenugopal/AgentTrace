from pathlib import Path

import pytest
from pydantic import ValidationError

from agenttrace.config import AgentConfig, BenchmarkConfig, ReplayConfig, load_config


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


def test_real_benchmark_rejects_incomplete_server_metadata() -> None:
    with pytest.raises(ValidationError, match="exact server_metadata"):
        BenchmarkConfig(
            experiment_id="real",
            source_trace=Path("trace.jsonl"),
            measurement_label="real_inference",
            replay={"tokenizer": "Qwen/tokenizer"},
            server_metadata={"backend": "vllm"},
        )


def test_real_benchmark_rejects_uncaptured_hardware_placeholders() -> None:
    metadata = {
        "backend": "vllm",
        "vllm_version": "0.6.3",
        "model": "model",
        "model_revision": "revision",
        "gpu": "REQUIRED_REPLACE_GPU",
        "nvidia_driver": "driver",
        "cuda_version": "cuda",
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "gpu_memory_utilization": 0.85,
        "prefix_caching": False,
        "vllm_image_digest": "sha256:example",
    }
    with pytest.raises(ValidationError, match="placeholders"):
        BenchmarkConfig(
            experiment_id="real",
            source_trace=Path("trace.jsonl"),
            measurement_label="real_inference",
            replay={"tokenizer": "Qwen/tokenizer"},
            server_metadata=metadata,
        )
