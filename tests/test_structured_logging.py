import json
import logging

from agenttrace.telemetry.logging import JsonFormatter, bind_correlation


def test_json_logs_include_correlation_without_prompt_content() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="agenttrace.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request completed",
        args=(),
        exc_info=None,
    )
    with bind_correlation(experiment_id="exp", request_id="req"):
        payload = json.loads(formatter.format(record))
    assert payload["experiment_id"] == "exp"
    assert payload["request_id"] == "req"
    assert "prompt" not in payload
