import asyncio

import pytest

from agenttrace.config import ReplayConfig
from agenttrace.instrumentation.tokens import WhitespaceTokenCounter
from agenttrace.models import ChatMessage, MessageRole
from agenttrace.replay.engine import ReplayEngine
from agenttrace.replay.models import WorkloadRequest
from agenttrace.replay.transform import transform_workload
from agenttrace.serving.client import InferenceError, InferenceRequest, InferenceResponse


def workload_request(sequence: int, *, agent: str = "a") -> WorkloadRequest:
    return WorkloadRequest(
        source_trace_id="trace",
        source_request_id=f"source-{agent}-{sequence}",
        agent_id=agent,
        sequence_number=sequence,
        model="mock",
        messages=[ChatMessage(role=MessageRole.USER, content="one two")],
        content_token_count=2,
        recorded_submission_offset_seconds=0,
        recorded_inter_request_seconds=0,
        expected_output_tokens=2,
    )


class ConcurrencyClient:
    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return InferenceResponse(
            request_id=request.request_id,
            content="ok",
            input_tokens=2,
            output_tokens=1,
            elapsed_seconds=0.02,
            ttft_seconds=0.01,
        )

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_open_loop_does_not_wait_for_same_agent_request() -> None:
    client = ConcurrencyClient()
    engine = ReplayEngine(ReplayConfig(mode="open_loop"), client)
    result = await engine.run_open_loop([workload_request(0), workload_request(1)])
    assert client.maximum == 2
    assert len(result.successful_attempts) == 2


@pytest.mark.asyncio
async def test_closed_loop_serializes_each_agent_stream() -> None:
    client = ConcurrencyClient()
    engine = ReplayEngine(ReplayConfig(mode="closed_loop"), client)
    result = await engine.run_closed_loop([workload_request(0), workload_request(1)])
    assert client.maximum == 1
    assert len(result.successful_attempts) == 2


class RetryClient(ConcurrencyClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.calls += 1
        if self.calls == 1:
            raise InferenceError("overloaded", status_code=503, retryable=True)
        return await super().complete(request)


@pytest.mark.asyncio
async def test_retry_attempts_are_recorded_separately() -> None:
    client = RetryClient()
    engine = ReplayEngine(ReplayConfig(retry_count=1), client)
    result = await engine.run_open_loop([workload_request(0)])
    assert [attempt.status for attempt in result.attempts] == ["failed", "succeeded"]
    assert [attempt.attempt_number for attempt in result.attempts] == [1, 2]


def test_parameterized_prompt_is_actually_resized() -> None:
    config = ReplayConfig(
        mode="parameterized",
        prompt_tokens=113,
        output_tokens=17,
        concurrent_agents=3,
        active_subagents=2,
        execution_length=2,
    )
    counter = WhitespaceTokenCounter()
    result = transform_workload([workload_request(0)], config, counter)
    assert len(result) == 6
    assert all(counter.count(request.messages[0].content) == 113 for request in result)
    assert all(request.expected_output_tokens == 17 for request in result)
    assert sum(request.parent_agent_id is not None for request in result) == 4
