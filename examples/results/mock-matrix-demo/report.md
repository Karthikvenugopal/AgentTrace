# AgentTrace benchmark report: mock-matrix-demo

> **MOCK-SERVER DEMONSTRATION — NOT REAL INFERENCE PERFORMANCE**

Started: `2026-09-19T19:21:01.358761+00:00`  
Tokenizer: `agenttrace/whitespace-v1` (whitespace_estimate)  
Repeated trials: `2`

## Measurement semantics

Open-loop replay schedules arrivals by trace time without waiting for prior responses. Closed-loop replay advances each agent stream only after its preceding request completes and the recorded delay elapses. Client TTFT ends at the first non-empty streamed chunk; it is not treated as exact server queueing time. Failed and timeout attempts are shown but excluded from successful-request latency and throughput aggregates.

## Aggregate observations

| Case | Context tokens | Agents | Mode | Pattern | n | Median TTFT (s) | Median latency (s) | Output tok/s | Failed/timeouts |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| case-1d271f28df | 256 | 2 | closed_loop | concurrent | 8 | 0.0139 | 0.0330 | 813.7007 | 0/0 |
| case-1de954e9c5 | 256 | 2 | open_loop | parent_subagents | 8 | 0.0172 | 0.0366 | 1368.8668 | 0/0 |
| case-3424bba683 | 64 | 1 | open_loop | concurrent | 4 | 0.0139 | 0.0331 | 737.2423 | 0/0 |
| case-57406a3d0f | 64 | 1 | closed_loop | concurrent | 4 | 0.0135 | 0.0326 | 413.1004 | 0/0 |
| case-62e829a596 | 64 | 2 | closed_loop | parent_subagents | 8 | 0.0140 | 0.0330 | 813.5197 | 0/0 |
| case-650546bcf1 | 64 | 2 | open_loop | parent_subagents | 8 | 0.0143 | 0.0335 | 1393.5310 | 0/0 |
| case-6deac7e7bc | 256 | 1 | closed_loop | parent_subagents | 4 | 0.0137 | 0.0328 | 406.2125 | 0/0 |
| case-7648dfd4bb | 64 | 1 | closed_loop | parent_subagents | 4 | 0.0135 | 0.0326 | 416.1465 | 0/0 |
| case-7daf8c7899 | 256 | 1 | open_loop | concurrent | 4 | 0.0139 | 0.0326 | 747.1224 | 0/0 |
| case-8f9dbcb7b4 | 256 | 1 | open_loop | parent_subagents | 4 | 0.0160 | 0.0349 | 709.1648 | 0/0 |
| case-a7c858a6b0 | 64 | 1 | open_loop | parent_subagents | 4 | 0.0139 | 0.0331 | 741.9258 | 0/0 |
| case-a943e8f8d2 | 256 | 2 | closed_loop | parent_subagents | 8 | 0.0139 | 0.0331 | 787.8669 | 0/0 |
| case-b28bdd04c5 | 64 | 2 | open_loop | concurrent | 8 | 0.0171 | 0.0362 | 1358.6380 | 0/0 |
| case-c2b2666bca | 256 | 2 | open_loop | concurrent | 8 | 0.0147 | 0.0341 | 1368.9547 | 0/0 |
| case-c81c36450f | 64 | 2 | closed_loop | concurrent | 8 | 0.0139 | 0.0326 | 821.7138 | 0/0 |
| case-d00b8f2fde | 256 | 1 | closed_loop | concurrent | 4 | 0.0135 | 0.0326 | 410.8305 | 0/0 |

Tail percentiles are omitted when fewer than 20 observations are available for p95 or fewer than 100 for p99.

## Charts

- [concurrency_vs_generation_throughput](charts/concurrency-vs-output-throughput.png)
- [concurrency_vs_latency](charts/concurrency-vs-latency.png)
- [context_vs_ttft](charts/context-vs-ttft.png)
- [execution_length_vs_cumulative_prompt_tokens](charts/execution-length-vs-cumulative-prompt-tokens.png)
- [subagent_activity_vs_latency](charts/subagent-activity-vs-latency.png)
- `subagent_arrivals_vs_latency`: unavailable: source trace has no parent-child agent activity
- `tool_wait_vs_server_utilization`: unavailable: requires two tool-wait settings and sampled GPU KV-cache usage

## Unavailable measurements

- `request_level_server_queueing`: vLLM Prometheus queue histograms are aggregate and cannot be assigned exactly to individual requests without backend-supported correlation
- `tool_wait_vs_server_utilization`: requires at least two tool-wait settings and aligned vLLM GPU KV-cache samples; the current experiment did not provide both

## Interpretation boundary

This file reports observations only. It does not claim causal improvements or generalize beyond the recorded configuration.

Machine-readable inputs: `metadata.json`, `observations.jsonl`, `aggregates.json`, and `analysis.json`.
