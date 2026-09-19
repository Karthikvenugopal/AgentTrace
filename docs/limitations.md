# Threats to validity and current limitations

AgentTrace makes workload mechanics inspectable; it does not eliminate experimental
confounders. Reports should address the following threats explicitly.

## Workload realism

Synthetic requests isolate context, arrival, and output dimensions but omit model-driven
branching, repository semantics, and the heavy-tailed think/tool waits of live agents.
The CPU mock server validates protocol and timing code only. Neither source can support
claims about real LLM quality or GPU performance.

The included coding-agent fixture is deliberately small and deterministic. Larger
repositories change search volume, tool output, prompt growth, test duration, and task
success rates. Public benchmark tasks may also be memorized by a model.

## Replay fidelity

Inference replay preserves requests, lineage, and timing policy, not complete autonomous
behavior. A live model response can change the next tool call, prompt, timing, or task
outcome. Closed-loop replay captures response-dependent pacing but still follows a
recorded request sequence. Parameterized replay is a generated workload, not a claim
that a real agent would produce those prompts.

Metadata-only or fully redacted traces are not exactly replayable. Constructing replacement
content with the same token count does not reproduce vocabulary, attention patterns,
prefix reuse, or model behavior.

## Tokenizers and context

The fallback whitespace counter is intentionally labeled an estimate. Chat templates,
special tokens, tool-call serialization, Unicode, and tokenizer revisions can produce a
different server prompt count. Client/server discrepancies are retained. Context-length
experiments should use the exact served tokenizer and document whether the target counts
content tokens or the complete templated request.

## Cache effects

Prefix cache, model weight cache, CUDA graph capture, allocator state, compilation,
filesystem cache, and previous prompts can affect measurements. Warmups address only
some cold-start effects. Reusing repeated prompt text can overstate prefix-cache benefit.
Record actual cache flags and separate cold/warm-cache studies.

## Hardware and system variability

GPU SKU, memory capacity/bandwidth, driver, CUDA, vLLM kernels, tensor parallel layout,
power/clock policy, temperature, CPU scheduling, NUMA placement, network path, and other
tenants all affect results. A client process's PyTorch allocator is not the vLLM server's
GPU utilization. Container tags should be resolved to immutable digests for publication.

## Agent nondeterminism

Temperature zero does not guarantee bitwise reproducibility across backend releases or
parallel schedules. Tool output, filesystem order, dependency versions, network access,
and concurrent subagents add nondeterminism. Agent success and serving performance need
separate outcome reporting.

## Timing and queueing

Client TTFT includes transport and server work; it is not queueing delay. Stream chunks
may contain multiple tokens and can be buffered. Prometheus scraping is sampled and
aggregate. Request-level server queue/prefill/decode values remain unavailable unless a
backend supplies correlated observations.

## Sample size and ordering

Small samples make p95/p99 unstable; AgentTrace suppresses them below 20 and 100 samples.
Randomization and rotation reduce fixed-order bias but cannot remove time trends. Report
trial counts, failures, timeouts, retry policy, and execution order. Use more repetitions
and uncertainty estimates for research claims.

## Security boundary

Path resolution, isolation copies, allowlists, timeouts, and no-shell subprocesses reduce
risk but do not constitute a hardened multi-tenant sandbox. Tests or interpreters can
execute arbitrary repository code when allowlisted. Use an OS/container/VM security
boundary for untrusted repositories and restrict network, credentials, and mounts.

Full traces can contain source code and tool output. Redaction is regex-based and may
miss secrets. Review artifacts before sharing; prefer metadata-only collection when
exact replay is unnecessary.
