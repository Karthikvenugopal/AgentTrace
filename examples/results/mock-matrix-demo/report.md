# AgentTrace benchmark report: mock-matrix-demo

> **MOCK-SERVER DEMONSTRATION — NOT REAL INFERENCE PERFORMANCE**

Started: `2026-09-19T19:13:36.695975+00:00`  
Tokenizer: `agenttrace/whitespace-v1` (whitespace_estimate)  
Repeated trials: `2`

## Measurement semantics

Open-loop replay schedules arrivals by trace time without waiting for prior responses. Closed-loop replay advances each agent stream only after its preceding request completes and the recorded delay elapses. Client TTFT ends at the first non-empty streamed chunk; it is not treated as exact server queueing time. Failed and timeout attempts are shown but excluded from successful-request latency and throughput aggregates.

## Aggregate observations

| Case | Context tokens | Agents | Mode | Pattern | n | Median TTFT (s) | Median latency (s) | Output tok/s | Failed/timeouts |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| case-1d271f28df | 256 | 2 | closed_loop | concurrent | 8 | 0.0138 | 0.0329 | 926.2542 | 0/0 |
| case-1de954e9c5 | 256 | 2 | open_loop | parent_subagents | 8 | 0.0126 | 0.0320 | 1903.6571 | 0/0 |
| case-3424bba683 | 64 | 1 | open_loop | concurrent | 4 | 0.0151 | 0.0342 | 791.6677 | 0/0 |
| case-57406a3d0f | 64 | 1 | closed_loop | concurrent | 4 | 0.0134 | 0.0325 | 471.9834 | 0/0 |
| case-62e829a596 | 64 | 2 | closed_loop | parent_subagents | 8 | 0.0140 | 0.0332 | 896.3837 | 0/0 |
| case-650546bcf1 | 64 | 2 | open_loop | parent_subagents | 8 | 0.0136 | 0.0327 | 1839.7413 | 0/0 |
| case-6deac7e7bc | 256 | 1 | closed_loop | parent_subagents | 4 | 0.0135 | 0.0326 | 470.6540 | 0/0 |
| case-7648dfd4bb | 64 | 1 | closed_loop | parent_subagents | 4 | 0.0134 | 0.0325 | 471.6564 | 0/0 |
| case-7daf8c7899 | 256 | 1 | open_loop | concurrent | 4 | 0.0133 | 0.0324 | 916.3672 | 0/0 |
| case-8f9dbcb7b4 | 256 | 1 | open_loop | parent_subagents | 4 | 0.0134 | 0.0325 | 855.0206 | 0/0 |
| case-a7c858a6b0 | 64 | 1 | open_loop | parent_subagents | 4 | 0.0134 | 0.0324 | 914.5862 | 0/0 |
| case-a943e8f8d2 | 256 | 2 | closed_loop | parent_subagents | 8 | 0.0136 | 0.0324 | 949.4563 | 0/0 |
| case-b28bdd04c5 | 64 | 2 | open_loop | concurrent | 8 | 0.0138 | 0.0328 | 1757.1797 | 0/0 |
| case-c2b2666bca | 256 | 2 | open_loop | concurrent | 8 | 0.0137 | 0.0330 | 1792.1147 | 0/0 |
| case-c81c36450f | 64 | 2 | closed_loop | concurrent | 8 | 0.0136 | 0.0326 | 940.6577 | 0/0 |
| case-d00b8f2fde | 256 | 1 | closed_loop | concurrent | 4 | 0.0129 | 0.0320 | 483.2778 | 0/0 |

Tail percentiles are omitted when fewer than 20 observations are available for p95 or fewer than 100 for p99.

## Charts

- [concurrency_vs_generation_throughput](charts/concurrency-vs-output-throughput.png)
- [concurrency_vs_latency](charts/concurrency-vs-latency.png)
- [context_vs_ttft](charts/context-vs-ttft.png)

## Unavailable measurements

- `request_level_server_queueing`: vLLM Prometheus queue histograms are aggregate and cannot be assigned exactly to individual requests without backend-supported correlation
- `tool_wait_vs_server_utilization`: requires aligned time-series vLLM utilization samples; before/after counters are not sufficient for a utilization curve

## Interpretation boundary

This file reports observations only. It does not claim causal improvements or generalize beyond the recorded configuration.

Machine-readable inputs: `metadata.json`, `observations.jsonl`, `aggregates.json`, and `analysis.json`.
