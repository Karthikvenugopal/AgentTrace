"""Bounded coding-agent execution loop with request and tool tracing."""

from __future__ import annotations

import hashlib
from datetime import timedelta

from pydantic import ValidationError

from agenttrace.agent.limits import ExecutionBudget
from agenttrace.agent.protocol import FinishAction, ToolAction, parse_action, protocol_instruction
from agenttrace.agent.tools import ToolRegistry, default_tools
from agenttrace.agent.workspace import RepositoryWorkspace
from agenttrace.config import AgentConfig
from agenttrace.instrumentation.requests import RequestSequenceTracker
from agenttrace.instrumentation.timing import ActiveRequestCounter, snapshot
from agenttrace.instrumentation.tokens import TokenCounter, WhitespaceTokenCounter, count_messages
from agenttrace.instrumentation.tools import ToolTraceContext, execute_instrumented_tool
from agenttrace.models import (
    AgentOutcome,
    ChatMessage,
    ExecutionStatus,
    MessageRole,
    RequestStatus,
    new_id,
)
from agenttrace.serving.client import InferenceClient, InferenceError, InferenceRequest
from agenttrace.telemetry.logging import bind_correlation
from agenttrace.telemetry.metrics import METRICS
from agenttrace.telemetry.tracing import trace_span
from agenttrace.tracing.schema import AgentRecord, OutcomeRecord, RequestRecord
from agenttrace.tracing.storage import TraceWriter


class CodingAgent:
    def __init__(
        self,
        config: AgentConfig,
        workspace: RepositoryWorkspace,
        client: InferenceClient,
        writer: TraceWriter,
        trace_id: str,
        *,
        token_counter: TokenCounter | None = None,
        active_requests: ActiveRequestCounter | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.client = client
        self.writer = writer
        self.trace_id = trace_id
        self.counter = token_counter or WhitespaceTokenCounter()
        self.active_requests = active_requests or ActiveRequestCounter()
        self.tools = ToolRegistry(
            default_tools(
                config.workspace.allowed_commands,
                timeout_seconds=config.limits.command_timeout_seconds,
                max_output_bytes=config.limits.max_tool_output_bytes,
            )
        )

    async def run(self) -> AgentOutcome:
        with (
            bind_correlation(
                experiment_id=self.config.experiment_id,
                trace_id=self.trace_id,
                agent_id=self.config.agent_id,
            ),
            trace_span(
                "agent.execution",
                {
                    "agenttrace.experiment_id": self.config.experiment_id,
                    "agenttrace.trace_id": self.trace_id,
                    "agenttrace.agent_id": self.config.agent_id,
                    "agenttrace.parent_agent_id": self.config.parent_agent_id,
                },
            ),
        ):
            return await self._run()

    async def _run(self) -> AgentOutcome:
        started = snapshot()
        self.writer.write(
            AgentRecord(
                experiment_id=self.config.experiment_id,
                trace_id=self.trace_id,
                agent_id=self.config.agent_id,
                parent_agent_id=self.config.parent_agent_id,
                started_at=started.wall_time,
                task_hash="sha256:" + hashlib.sha256(self.config.task.encode()).hexdigest(),
            )
        )
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=self.config.system_prompt
                + "\n\n"
                + protocol_instruction(self.tools.specifications),
            ),
            ChatMessage(role=MessageRole.USER, content=self.config.task),
        ]
        total_input = 0
        total_output = 0
        budget = ExecutionBudget(self.config.limits, started.monotonic)
        sequence_tracker = RequestSequenceTracker()
        summary = "execution limit reached"
        status = ExecutionStatus.LIMIT_REACHED
        iterations = 0

        for sequence in range(self.config.limits.max_iterations):
            stop_reason = budget.stop_reason(
                now_monotonic=snapshot().monotonic, next_iteration=sequence
            )
            if stop_reason is not None:
                summary = stop_reason
                break
            iterations = sequence + 1
            request_id = new_id("req")
            accounting = count_messages(messages, self.counter)
            created = snapshot()
            async with self.active_requests.track() as concurrency:
                submitted = snapshot()
                try:
                    with (
                        bind_correlation(request_id=request_id),
                        METRICS.track_request("agent", self.config.endpoint.model),
                        trace_span(
                            "agent.step",
                            {
                                "agenttrace.agent_id": self.config.agent_id,
                                "agenttrace.sequence": sequence,
                            },
                        ),
                        trace_span(
                            "llm.request",
                            {
                                "agenttrace.request_id": request_id,
                                "agenttrace.agent_id": self.config.agent_id,
                                "llm.model": self.config.endpoint.model,
                            },
                        ),
                    ):
                        response = await self.client.complete(
                            InferenceRequest(
                                request_id=request_id,
                                model=self.config.endpoint.model,
                                messages=list(messages),
                                temperature=self.config.sampling.temperature,
                                top_p=self.config.sampling.top_p,
                                max_tokens=self.config.sampling.max_tokens,
                                seed=self.config.sampling.seed,
                                stream=self.config.endpoint.stream,
                            )
                        )
                    completed = snapshot()
                except InferenceError as exc:
                    completed = snapshot()
                    self._write_failed_request(
                        request_id,
                        sequence,
                        messages,
                        accounting.total,
                        accounting.breakdown,
                        created,
                        submitted,
                        completed,
                        concurrency,
                        sequence_tracker,
                        exc,
                    )
                    summary = f"inference failed: {exc}"
                    METRICS.observe_request(
                        component="agent",
                        model=self.config.endpoint.model,
                        status="failed",
                        latency_seconds=completed.monotonic - submitted.monotonic,
                        ttft_seconds=None,
                        input_tokens=accounting.total,
                        output_tokens=0,
                    )
                    status = ExecutionStatus.FAILED
                    break
            output_tokens = response.output_tokens
            if output_tokens is None:
                output_tokens = self.counter.count(response.content)
            total_input += response.input_tokens or accounting.total
            total_output += output_tokens
            METRICS.observe_request(
                component="agent",
                model=self.config.endpoint.model,
                status="succeeded",
                latency_seconds=completed.monotonic - submitted.monotonic,
                ttft_seconds=response.ttft_seconds,
                input_tokens=response.input_tokens or accounting.total,
                output_tokens=output_tokens,
            )
            budget.add_usage(response.input_tokens or accounting.total, output_tokens)
            try:
                action = parse_action(response.content)
            except (ValidationError, ValueError) as exc:
                action = None
                messages.extend(
                    [
                        ChatMessage(role=MessageRole.ASSISTANT, content=response.content),
                        ChatMessage(
                            role=MessageRole.USER,
                            content=f"Protocol error: {exc}. Return one valid JSON action.",
                        ),
                    ]
                )
            tool_ids = [action.tool_call_id] if isinstance(action, ToolAction) else []
            self._write_successful_request(
                request_id,
                sequence,
                messages,
                accounting.total,
                accounting.breakdown,
                created,
                submitted,
                completed,
                concurrency,
                sequence_tracker,
                response,
                output_tokens,
                tool_ids,
            )
            if action is None:
                continue
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response.content))
            if isinstance(action, FinishAction):
                status = ExecutionStatus.SUCCEEDED
                summary = action.summary
                break
            execution = await execute_instrumented_tool(
                registry=self.tools,
                workspace=self.workspace,
                name=action.tool,
                arguments=action.arguments,
                context=ToolTraceContext(
                    experiment_id=self.config.experiment_id,
                    trace_id=self.trace_id,
                    agent_id=self.config.agent_id,
                    request_id=request_id,
                    tool_call_id=action.tool_call_id,
                ),
                counter=self.counter,
                writer=self.writer,
            )
            tool_output = execution.output
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL,
                    name=action.tool,
                    tool_call_id=action.tool_call_id,
                    content=tool_output,
                )
            )

        finished = snapshot()
        outcome = AgentOutcome(
            status=status,
            summary=summary,
            iterations=iterations,
            input_tokens=total_input,
            output_tokens=total_output,
            started_at=started.wall_time,
            completed_at=finished.wall_time,
        )
        self.writer.write(
            OutcomeRecord(
                experiment_id=self.config.experiment_id,
                trace_id=self.trace_id,
                agent_id=self.config.agent_id,
                status=status,
                summary=summary,
                iterations=iterations,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                started_at=started.wall_time,
                completed_at=finished.wall_time,
            )
        )
        return outcome

    def _content_fields(self, messages: list[ChatMessage]) -> dict[str, object]:
        mode = self.config.trace.content_mode
        if mode == "metadata_only":
            return {"prompt": None, "messages": None}
        if mode == "redacted":
            return {
                "prompt": "[REDACTED]",
                "messages": [
                    {"role": message.role.value, "content": "[REDACTED]"} for message in messages
                ],
            }
        return {
            "prompt": "\n".join(message.content for message in messages),
            "messages": [
                message.model_dump(mode="json", exclude_none=True) for message in messages
            ],
        }

    def _request_base(  # type: ignore[no-untyped-def]
        self,
        request_id,
        sequence,
        messages,
        input_tokens,
        breakdown,
        created,
        submitted,
        completed,
        concurrency,
        sequence_tracker,
    ) -> dict[str, object]:
        observation = sequence_tracker.observe(
            prompt_tokens=input_tokens, submitted_monotonic=submitted.monotonic
        )
        return {
            "experiment_id": self.config.experiment_id,
            "trace_id": self.trace_id,
            "agent_id": self.config.agent_id,
            "parent_agent_id": self.config.parent_agent_id,
            "request_id": request_id,
            "sequence_number": sequence,
            "model": self.config.endpoint.model,
            "created_at": created.wall_time,
            "submitted_at": submitted.wall_time,
            "completed_at": completed.wall_time,
            "elapsed_seconds": completed.monotonic - submitted.monotonic,
            "monotonic_started": submitted.monotonic,
            "monotonic_completed": completed.monotonic,
            "time_since_previous_request_seconds": observation.time_since_previous_request_seconds,
            "input_tokens": input_tokens,
            "context_growth_tokens": observation.growth_tokens,
            "prompt_breakdown": breakdown,
            "tokenizer": self.counter.identity,
            "token_count_method": self.counter.method,
            "sampling_parameters": self.config.sampling.model_dump(mode="json"),
            "concurrency_at_submission": concurrency,
            **self._content_fields(messages),
        }

    def _write_successful_request(  # type: ignore[no-untyped-def]
        self,
        request_id,
        sequence,
        messages,
        input_tokens,
        breakdown,
        created,
        submitted,
        completed,
        concurrency,
        sequence_tracker,
        response,
        output_tokens,
        tool_ids,
    ) -> None:
        self.writer.write(
            RequestRecord(
                **self._request_base(
                    request_id,
                    sequence,
                    messages,
                    input_tokens,
                    breakdown,
                    created,
                    submitted,
                    completed,
                    concurrency,
                    sequence_tracker,
                ),
                output_tokens=output_tokens,
                server_input_tokens=response.input_tokens,
                server_output_tokens=response.output_tokens,
                token_count_discrepancy=(
                    None if response.input_tokens is None else response.input_tokens - input_tokens
                ),
                status=RequestStatus.SUCCEEDED,
                associated_tool_call_ids=tool_ids,
                first_token_at=(
                    None
                    if response.ttft_seconds is None
                    else submitted.wall_time + timedelta(seconds=response.ttft_seconds)
                ),
                ttft_seconds=response.ttft_seconds,
                stream_chunk_arrivals_seconds=[chunk.elapsed_seconds for chunk in response.chunks],
                stream_chunk_token_counts=[chunk.estimated_tokens for chunk in response.chunks],
            )
        )

    def _write_failed_request(  # type: ignore[no-untyped-def]
        self,
        request_id,
        sequence,
        messages,
        input_tokens,
        breakdown,
        created,
        submitted,
        completed,
        concurrency,
        sequence_tracker,
        error,
    ) -> None:
        status = RequestStatus.TIMEOUT if "timeout" in str(error).lower() else RequestStatus.FAILED
        self.writer.write(
            RequestRecord(
                **self._request_base(
                    request_id,
                    sequence,
                    messages,
                    input_tokens,
                    breakdown,
                    created,
                    submitted,
                    completed,
                    concurrency,
                    sequence_tracker,
                ),
                output_tokens=0,
                status=status,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        )
