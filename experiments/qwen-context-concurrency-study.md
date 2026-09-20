# Qwen context × concurrency study

## Execution status

**No GPU/vLLM performance results were produced on the available host.** This is a
preflight record, not a benchmark report and not a source of resume metrics.

Preflight performed 2026-09-19 (America/Los_Angeles):

- Host: Apple M3 (10-core integrated Apple GPU), Darwin arm64.
- NVIDIA CUDA device: unavailable; `nvidia-smi` is not installed.
- Local Python environment: neither PyTorch nor vLLM is installed.
- Configured vLLM endpoint: no service accepted connections at `127.0.0.1:8000`.
- Docker Desktop was restarted and verified as Linux/arm64 (`29.2.1`). It exposes only
  `runc`, with no NVIDIA container runtime and no GPU generic resources.
- Existing repository results are labeled `mock`; they validate mechanics only.

Creating request/summary CSVs, charts, comparisons, or resume statements from the mock
timings would misrepresent simulated delays as model inference performance. Those result
artifacts are intentionally absent until the study runs on a CUDA host.

## Registered matrix

The executable configuration is [`configs/benchmark-gpu.yaml`](../configs/benchmark-gpu.yaml):

| Controlled dimension | Values |
|---|---|
| Model | Qwen2.5-Coder-0.5B-Instruct, pinned revision |
| Content-token target | 2,048; 8,192; 16,384; 32,000 |
| Configured concurrent agents | 1; 2; 4; 8 |
| Output target | 128 tokens with vLLM `ignore_eos` |
| Requests per agent per trial | 2, sequential within each agent stream |
| Measured repetitions | 10 |
| Warmups | 1 before the first measured trial of each case, excluded |
| Scheduling | Closed loop with zero think delay; seeded shuffle plus rotation |
| Prefix caching | Disabled |

At concurrency 1, each case has 20 measured requests, meeting AgentTrace's minimum p95
sample size. The full matrix schedules 1,200 measured requests. Closed-loop streams keep
at most one in-flight request per agent, so configured agent counts bound request
concurrency; measured maximum and time-weighted effective concurrency remain in the
artifacts. The 32,000-token bin is
used instead of 32,768 content tokens to leave room for the chat template and 128 output
tokens inside the model's 32,768-position limit.

## Required CUDA-host execution

1. Start the pinned vLLM container from `docker-compose.gpu.yml`.
2. Capture GPU name/memory, driver, CUDA version, vLLM version, and image digest using the
   commands in [`docs/reproducibility.md`](../docs/reproducibility.md).
3. Copy the benchmark config and replace every `REQUIRED_*` value. AgentTrace rejects a
   real run while placeholders remain.
4. Set `AGENTTRACE_VLLM_METRICS_URL` and run `agenttrace benchmark run`.
5. Inspect failures, context-limit errors, actual server token counts, and trial-level
   anomalies. Add disclosed repetitions for anomalies; do not delete raw attempts.
6. Run the explicit GPU integration suite and archive the config, server log, image
   digest, raw observations, CSVs, charts, and report together.

The generated report will state that ITL is unavailable, keep aggregate vLLM queue/KV
metrics separate from request observations, and expose all percentage-comparison source
values and case identifiers.
