"""Open-loop workload replay with attempt-level observations."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

from agenttrace.config import ReplayConfig
from agenttrace.models import new_id
from agenttrace.replay.models import ReplayAttempt, ReplaySessionResult, WorkloadRequest
from agenttrace.serving.client import InferenceClient, InferenceError, InferenceRequest


class ReplayEngine:
    def __init__(self, config: ReplayConfig, client: InferenceClient) -> None:
        self.config = config
        self.client = client

    async def run_open_loop(self, workload: list[WorkloadRequest]) -> ReplaySessionResult:
        session_id = new_id("replay")
        wall_started = datetime.now(UTC)
        monotonic_started = time.monotonic()
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def scheduled(request: WorkloadRequest) -> list[ReplayAttempt]:
            target = request.recorded_submission_offset_seconds * self.config.time_scale
            await asyncio.sleep(max(0.0, target - (time.monotonic() - monotonic_started)))
            async with semaphore:
                return await self._issue_with_retries(
                    request,
                    session_id=session_id,
                    mode="open_loop",
                    scheduled_offset=target,
                    session_started=monotonic_started,
                )

        nested = await asyncio.gather(*(scheduled(request) for request in workload))
        attempts = [attempt for group in nested for attempt in group]
        return ReplaySessionResult(
            replay_session_id=session_id,
            source_trace_id=workload[0].source_trace_id if workload else "empty",
            mode="open_loop",
            started_at=wall_started,
            completed_at=datetime.now(UTC),
            attempts=attempts,
        )

    async def _issue_with_retries(
        self,
        request: WorkloadRequest,
        *,
        session_id: str,
        mode: str,
        scheduled_offset: float,
        session_started: float,
    ) -> list[ReplayAttempt]:
        attempts: list[ReplayAttempt] = []
        replay_request_id = new_id("replayreq")
        for number in range(1, self.config.retry_count + 2):
            submitted_offset = time.monotonic() - session_started
            submitted_at = datetime.now(UTC)
            started = time.monotonic()
            try:
                response = await self.client.complete(
                    InferenceRequest(
                        request_id=replay_request_id,
                        model=self.config.endpoint.model or request.model,
                        messages=request.messages,
                        temperature=float(request.sampling_parameters.get("temperature", 0.0)),
                        top_p=float(request.sampling_parameters.get("top_p", 1.0)),
                        max_tokens=self.config.output_tokens or request.expected_output_tokens,
                        seed=self.config.seed,
                        stream=True,
                    )
                )
                completed_at = datetime.now(UTC)
                attempts.append(
                    ReplayAttempt(
                        replay_session_id=session_id,
                        source_trace_id=request.source_trace_id,
                        source_request_id=request.source_request_id,
                        replay_request_id=replay_request_id,
                        agent_id=request.agent_id,
                        sequence_number=request.sequence_number,
                        attempt_number=number,
                        mode=mode,  # type: ignore[arg-type]
                        scheduled_offset_seconds=scheduled_offset,
                        submitted_offset_seconds=submitted_offset,
                        client_scheduling_delay_seconds=max(0, submitted_offset - scheduled_offset),
                        submitted_at=submitted_at,
                        completed_at=completed_at,
                        status="succeeded",
                        latency_seconds=time.monotonic() - started,
                        ttft_seconds=response.ttft_seconds,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        stream_chunk_arrivals_seconds=[c.elapsed_seconds for c in response.chunks],
                        stream_chunk_token_counts=[c.estimated_tokens for c in response.chunks],
                    )
                )
                break
            except InferenceError as exc:
                status = "timeout" if "timeout" in str(exc).lower() else "failed"
                attempts.append(
                    ReplayAttempt(
                        replay_session_id=session_id,
                        source_trace_id=request.source_trace_id,
                        source_request_id=request.source_request_id,
                        replay_request_id=replay_request_id,
                        agent_id=request.agent_id,
                        sequence_number=request.sequence_number,
                        attempt_number=number,
                        mode=mode,  # type: ignore[arg-type]
                        scheduled_offset_seconds=scheduled_offset,
                        submitted_offset_seconds=submitted_offset,
                        client_scheduling_delay_seconds=max(0, submitted_offset - scheduled_offset),
                        submitted_at=submitted_at,
                        completed_at=datetime.now(UTC),
                        status=status,
                        latency_seconds=time.monotonic() - started,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                if not exc.retryable:
                    break
        return attempts
