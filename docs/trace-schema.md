# Trace schema

The current schema version is `1.0`. A trace is UTF-8 JSON Lines so a collector can
flush each event without retaining a long experiment in memory. Every line includes
`schema_version`, `record_type`, and `experiment_id`. The first line must be one header.

## Record types

### `header`

Identifies the trace, source (`agent`, `synthetic`, `transformed`, or `replay`), content
privacy mode, creation time, environment, serving configuration, optional source trace,
transformation, and seed. Synthetic traces are unambiguously labeled.

### `agent`

Records agent ID, optional parent ID, start time, and a hash of the task. The validator
rejects self-parenting, missing parents, duplicate agents, and lineage cycles. Children
retain separate request histories.

### `request`

Contains:

- request ID and per-agent contiguous sequence number;
- model, creation/submission/completion wall times and monotonic duration anchors;
- time since that agent's previous request;
- input/output counts, server-reported usage, discrepancy, tokenizer, and method;
- role-level system/user/assistant/tool counts and prompt-growth delta;
- context window/utilization when known;
- sampling parameters and concurrency at submission;
- success/failure/timeout status and error details;
- associated tool-call IDs;
- optional privacy-controlled prompt/messages;
- TTFT, first-token wall time, chunk arrival offsets, and tokens per chunk.

Failed requests use the same record and are never silently dropped.

### `tool_call`

Joins a validated tool call to the LLM request that selected it. It records agent,
tool-call and request IDs, tool name, wall/monotonic timing, duration, bounded result
size, result token count, status, and error. The trace contains no command shell string;
the execution layer accepts only an argument vector for allowlisted executables.

### `outcome`

Records success, failure, cancellation, or limit exhaustion, along with summary,
iterations, total tokens, and agent lifetime.

## Example (abbreviated)

```json
{"schema_version":"1.0","record_type":"header","experiment_id":"exp","trace_id":"trace_1","source":"agent","content_mode":"metadata_only","created_at":"2026-01-01T00:00:00Z","environment":{"python_version":"3.11","platform":"linux","agenttrace_version":"0.1.0","model":"served-model"},"execution_config":{}}
{"schema_version":"1.0","record_type":"agent","experiment_id":"exp","trace_id":"trace_1","agent_id":"agent-1","parent_agent_id":null,"started_at":"2026-01-01T00:00:00Z","task_hash":"sha256:..."}
```

The canonical model definitions live in `agenttrace/tracing/schema.py`. Use the library
instead of depending on this abbreviated example.

## Validation

```bash
agenttrace trace validate --input path/to/trace.jsonl
```

Validation covers Pydantic field types and bounds, explicit supported schema versions,
one first-position header, consistent experiment/trace IDs, parent existence/cycles,
unique request/tool IDs, contiguous per-agent sequences, monotonic timestamps, aligned
stream arrays, tool-request references, and outcome completeness warnings.

Unknown schema versions fail rather than being guessed. Future migrations should parse
the source version, perform an explicit conversion, and preserve the original trace.

## Privacy and replayability

Metadata-only and fully redacted requests do not contain enough content for exact
recorded replay. Parameterized replay remains possible because it constructs new input
and labels the workload transformed. Full traces should only contain public/reviewed
content and should configure redaction patterns for tokens, credentials, and sensitive
identifiers. Authentication headers and `SecretStr` configuration values are never
serialized into the schema.
