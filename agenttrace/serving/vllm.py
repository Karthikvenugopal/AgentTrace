"""Version-tolerant configuration for vLLM's OpenAI-compatible server."""

from __future__ import annotations

import importlib.metadata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VLLMServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    dtype: Literal["auto", "half", "float16", "bfloat16", "float"] = "auto"
    tensor_parallel_size: int = Field(default=1, gt=0)
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    max_model_len: int | None = Field(default=None, gt=0)
    max_num_seqs: int | None = Field(default=None, gt=0)
    enable_prefix_caching: bool = False
    disable_log_requests: bool = False
    served_model_name: str | None = None
    tokenizer: str | None = None
    trust_remote_code: bool = False


def build_vllm_command(config: VLLMServerConfig) -> list[str]:
    command = [
        "vllm",
        "serve",
        config.model,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--dtype",
        config.dtype,
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
    ]
    if config.max_model_len is not None:
        command.extend(["--max-model-len", str(config.max_model_len)])
    if config.max_num_seqs is not None:
        command.extend(["--max-num-seqs", str(config.max_num_seqs)])
    if config.enable_prefix_caching:
        command.append("--enable-prefix-caching")
    if config.disable_log_requests:
        command.append("--disable-log-requests")
    if config.served_model_name:
        command.extend(["--served-model-name", config.served_model_name])
    if config.tokenizer:
        command.extend(["--tokenizer", config.tokenizer])
    if config.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def installed_vllm_version() -> str | None:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return None


def serving_metadata(config: VLLMServerConfig) -> dict[str, object]:
    """Persist actual cache/batching settings alongside experiment data."""

    return {
        "backend": "vllm",
        "vllm_version": installed_vllm_version(),
        "model": config.model,
        "tokenizer": config.tokenizer or config.model,
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.max_num_seqs,
        "prefix_caching": config.enable_prefix_caching,
        "dtype": config.dtype,
    }
