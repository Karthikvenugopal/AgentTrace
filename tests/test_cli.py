from pathlib import Path

from typer.testing import CliRunner

from agenttrace.cli.app import app

runner = CliRunner()


def test_trace_generate_validate_and_summarize_commands(tmp_path: Path) -> None:
    trace = tmp_path / "synthetic.jsonl"
    config = tmp_path / "synthetic.yaml"
    config.write_text(
        f"""experiment_id: cli-test
seed: 11
agents: 1
requests_per_agent: 2
initial_prompt_tokens: 8
prompt_growth_tokens: 4
output_tokens: 2
inter_request_seconds: 0.01
output: {trace}
""",
        encoding="utf-8",
    )
    generated = runner.invoke(app, ["trace", "generate", "--config", str(config)])
    assert generated.exit_code == 0, generated.output
    validated = runner.invoke(app, ["trace", "validate", "--input", str(trace)])
    assert validated.exit_code == 0, validated.output
    assert '"valid": true' in validated.output
    summarized = runner.invoke(app, ["trace", "summarize", "--input", str(trace)])
    assert summarized.exit_code == 0, summarized.output
    assert '"request_count": 2' in summarized.output
    assert '"source": "synthetic"' in summarized.output


def test_doctor_reports_optional_components_without_failing() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert '"name": "cuda"' in result.output
    assert '"name": "vllm_version"' in result.output
