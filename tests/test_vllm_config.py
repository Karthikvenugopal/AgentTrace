from agenttrace.serving.vllm import VLLMServerConfig, build_vllm_command, serving_metadata


def test_vllm_command_records_cache_and_capacity_settings() -> None:
    config = VLLMServerConfig(
        model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        revision="revision-123",
        max_model_len=4096,
        max_num_seqs=8,
        enable_prefix_caching=True,
    )
    command = build_vllm_command(config)
    assert command[:3] == ["vllm", "serve", config.model]
    assert "--enable-prefix-caching" in command
    assert command[command.index("--max-model-len") + 1] == "4096"
    assert command[command.index("--revision") + 1] == "revision-123"
    assert serving_metadata(config)["prefix_caching"] is True
    assert serving_metadata(config)["model_revision"] == "revision-123"
