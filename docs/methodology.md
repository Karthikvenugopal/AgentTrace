# Measurement methodology

AgentTrace treats an inference workload as a time-ordered set of client requests that
belong to agent streams. It separates what the client directly observes, what the
server exports in aggregate, and what can only be calculated across a complete coding
task. This distinction is part of the data model and reporting policy.

## Clocks and correlation

Wall-clock UTC timestamps correlate events across processes. Durations use a monotonic
clock so NTP adjustments and wall-clock changes cannot produce negative latency.
Experiment, trace, agent, parent-agent, request, and tool-call IDs join observations.
OpenTelemetry spans carry the same identifiers. Unique IDs are never Prometheus labels.

## Request-level metrics

| Metric | Definition | Source |
|---|---|---|
| End-to-end latency | Client submission until the stream closes or JSON response completes | Client monotonic clock |
| TTFT | Client submission until the first non-empty streamed content chunk arrives | Client monotonic clock |
| Chunk interval | Difference between consecutive non-empty chunk arrival times | Client stream |
| Input tokens | Model-tokenizer count when configured; labeled whitespace estimate otherwise | Client tokenizer |
| Output tokens | Server usage where present, otherwise client counter | Response/client tokenizer |
| Context growth | Current client prompt count minus the preceding count for that agent | Ordered trace |
| Client scheduling delay | Actual replay submission offset minus requested schedule offset | Replay scheduler |
| Request concurrency | Active client requests, including the request at submission | Client process |

A streamed chunk may contain multiple tokens, and network/proxy buffering changes its
arrival. AgentTrace therefore calls this an inter-*chunk* arrival interval. The first
token represented by a chunk can be assigned the observed interval; timing for extra
tokens in the same chunk is unknown. It is never presented as per-token GPU decode
latency.

TTFT contains some combination of network, server queue, prefill, first-token decode,
and buffering time. AgentTrace does not subtract or relabel TTFT as queueing time.

## Server-level metrics

The vLLM adapter discovers known metric names rather than assuming one release. It can
capture snapshots of running/waiting requests, GPU KV-cache usage, cumulative prompt
and generation tokens, and queue/prefill/decode histograms when the connected release
exports them. Missing families are recorded as unavailable.

Prometheus server metrics are usually worker/model aggregates. Unless the backend
exposes a correlated request identifier, a histogram observation cannot be assigned to
one client request. Reports therefore show aggregate server queue metrics separately.
Likewise, PyTorch CUDA values describe the AgentTrace client process and visible devices;
they do not reveal memory allocated by a separate vLLM process.

## Agent-task metrics

- Tool duration is monotonic completion minus start for each call.
- Tool-output volume is counted after output truncation, matching what enters context.
- Agent tool-wait idle time is the sum of tool durations for the agent. It does not
  imply the inference server was idle during the same interval.
- Execution length is the number of inference steps in an agent stream.
- Cumulative prompt consumption is the sum of prompt tokens across steps, rather than
  only the final context length.
- Total task tokens sum all input and output tokens for the task.
- Active-agent concurrency is derived from overlapping agent lifetimes; inference
  concurrency is derived from overlapping request lifetimes. They are not equivalent.
- Subagent arrival analysis uses parent IDs and each child's independent request stream.

## Replay semantics

Open-loop replay computes a target offset from the earliest recorded submission and
submits at that time without waiting for prior requests. A semaphore is an explicit
client resource limit; time spent waiting for it is client scheduling delay.

Closed-loop replay partitions requests by agent and sequence. Each stream waits for its
prior response, then applies the scaled recorded inter-request delay. Other agent
streams run concurrently. Consequently, closed-loop arrival times respond to server
latency while open-loop arrival times do not. Reports must not compare them as if their
offered loads were identical.

Parameterized replay is a generated derivative. Prompt text is constructed and counted
with the selected tokenizer, output limits are sent to the server, agents are replicated
with source-request lineage, and a manifest records the source, parameters, and seed.
Changing a stored token-count field without changing the sent content is prohibited.

## Trial protocol

Warmups run before each measured case and are not written into measured aggregates.
The configured matrix expands context tokens, concurrent agents, execution pattern,
workload type, and replay mode. A seeded shuffle plus per-repetition rotation reduces
fixed-order bias. Metadata records model, tokenizer, dependency versions, hardware
visibility, sampling, serving/batch/cache settings, seed, and timestamps.

For comparative studies, keep the model revision, tokenizer, sampling, output limit,
vLLM configuration, GPU clock/thermal conditions, and background load constant. Change
one controlled axis where feasible. Prefix cache and GPU cache settings are experimental
conditions, not implementation details.

## Aggregation

Every attempt remains in `observations.jsonl`. Successful-request latency and throughput
exclude failures, timeouts, and failed retry attempts. Reports state that policy and
separately count failures, timeouts, and attempts with retry number greater than one.

- Median is reported for any non-empty sample.
- p95 is reported only with at least 20 samples.
- p99 is reported only with at least 100 samples.
- Request throughput is successful requests divided by measured trial wall duration.
- Input/output token throughput uses successful reported/estimated tokens divided by
  that duration.
- Tail percentiles are linearly interpolated from ordered observations.

Repeated trials are pooled within an identical case for the current report. Researchers
who need uncertainty estimates should use the raw per-trial observations for bootstrap
confidence intervals or a pre-registered statistical model.

## Required analyses

The chart pipeline directly supports context versus TTFT, configured concurrency versus
end-to-end latency, and concurrency versus aggregate generation throughput. Trace
analysis emits prompt tokens, context growth, cumulative prompt consumption, tool wait,
tool-output volume, active request concurrency, and subagent counts for execution-length
and arrival studies.

Tool wait versus server utilization requires aligned time-series server samples. A
before/after Prometheus snapshot is insufficient, so the standard report marks that
analysis unavailable rather than manufacturing a curve. The same rule applies to exact
request-level server queueing without backend correlation.

## Interpretation

Generated charts are observations for the named experiment and configuration. They do
not automatically establish causality, performance improvements, or generality. Mock
reports validate orchestration and timing calculations only. Real inference claims
require a report explicitly configured as `real_inference` and evidence that it ran
against the documented vLLM/GPU environment.
