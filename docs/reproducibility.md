# Reproducibility guide

This guide recreates the CPU demonstration and describes the additional controls for a
GPU-backed vLLM study. Preserve the raw artifacts before interpreting aggregates.

## Record the source revision

```bash
git clone https://github.com/Karthikvenugopal/AgentTrace.git
cd AgentTrace
git rev-parse HEAD
python3 --version
```

Use Python 3.11 or 3.12 for the closest match to CI. Create an isolated environment and
install the project:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
ruff format --check .
ruff check .
mypy agenttrace
pytest -m 'not gpu' -q
```

Dependency major versions are constrained in `pyproject.toml`. For an archival study,
also capture `pip freeze`, the OS/kernel, NVIDIA driver, CUDA runtime, GPU SKU, and the
resolved vLLM container digest. AgentTrace stores relevant runtime versions and visible
client device metadata in `metadata.json`.

## Recreate the CPU demonstration

Start the mock server:

```bash
agenttrace mock-server \
  --port 8010 \
  --first-token-delay 0.01 \
  --inter-chunk-delay 0.005
```

In another terminal, execute and validate the real bounded agent workflow:

```bash
agenttrace agent run --config configs/agent.yaml
agenttrace trace validate --input examples/traces/mock-agent.jsonl
agenttrace trace summarize --input examples/traces/mock-agent.jsonl
```

The model behavior is scripted for the public fixture, but the repository tools,
temporary workspace, edits, subprocess test, model/tool feedback loop, streaming HTTP,
and trace collector are the production code paths.

Replay and benchmark it:

```bash
agenttrace replay run \
  --trace examples/traces/mock-agent.jsonl \
  --config configs/replay.yaml \
  --output results/mock-replay.json

AGENTTRACE_VLLM_METRICS_URL=http://127.0.0.1:8010/metrics \
  agenttrace benchmark run --config configs/benchmark.yaml
```

Inspect, do not merely assume, the artifacts:

```bash
agenttrace trace validate --input examples/traces/mock-agent.jsonl
sed -n '1,80p' results/mock-matrix-demo/report.md
python -m json.tool results/mock-matrix-demo/aggregates.json >/dev/null
wc -l results/mock-matrix-demo/observations.jsonl
```

The report banner must say `MOCK-SERVER DEMONSTRATION — NOT REAL INFERENCE
PERFORMANCE`. A checked-in example is under `examples/results/mock-matrix-demo/`.
Timing will vary with the host scheduler even though workload construction is seeded.

## Recreate a GPU experiment

Prerequisites are a Linux host supported by the selected vLLM/CUDA release, NVIDIA
Container Toolkit, a visible GPU, enough local storage for the model, and network access
or a populated model cache. The supplied 0.5B model/profile is intended as an accessible
starting point; verify actual allocation in vLLM logs and `nvidia-smi`.

```bash
docker compose -f docker-compose.gpu.yml config
docker compose -f docker-compose.gpu.yml up --build vllm
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models | python -m json.tool
curl -fsS http://127.0.0.1:8000/metrics | head
```

Install an exact tokenizer in the client environment for model studies:

```bash
pip install -e '.[dev,tokenizers,telemetry]'
```

Copy `configs/benchmark-gpu.yaml`. It intentionally contains `REQUIRED_*` hardware and
image-digest placeholders and fails validation until they are replaced. Capture exact
runtime values on the serving host:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
docker image inspect vllm/vllm-openai:v0.6.3 --format '{{index .RepoDigests 0}}'
docker compose -f docker-compose.gpu.yml exec vllm python -c \
  'import torch, vllm; print(vllm.__version__, torch.version.cuda, torch.cuda.get_device_name())'
```

Record those values in the copied configuration. The controlled profile pins the Qwen
model and tokenizer revision, sets a 32,768-token model limit, disables prefix caching,
requests 128 output tokens with EOS ignored, and tests 2,048/8,192/16,384/32,000 content
tokens at concurrency 1/2/4/8. Two closed-loop requests per agent with zero think delay
and 10 repetitions keep at most one request active per agent while providing 20 samples
at concurrency=1. The 32,000 target leaves capacity for chat-template tokens and
generation. Keep all serving flags fixed across cases.

```bash
export AGENTTRACE_VLLM_METRICS_URL=http://127.0.0.1:8000/metrics
agenttrace benchmark run --config configs/benchmark-gpu.yaml
agenttrace benchmark report --results results/qwen-context-concurrency-gpu
```

The study writes `requests.csv`, `summary.csv`, three p95/throughput charts, JSON audit
artifacts, and a Markdown report. Ten repetitions and two requests per agent produce at
least 20 observations for concurrency=1, meeting the reporter's minimum p95 sample size.
Inspect actual prompt usage and server logs for context-limit rejection before accepting
the 32K cases.

Run explicit integration checks. They are designed to fail, not skip, after the GPU
suite is requested and CUDA or the selected model is absent:

```bash
AGENTTRACE_RUN_GPU_TESTS=1 \
AGENTTRACE_VLLM_URL=http://127.0.0.1:8000/v1 \
AGENTTRACE_VLLM_MODEL=qwen-coder-small \
pytest -m gpu -vv
```

## Experimental controls

1. Let the GPU reach a stable idle temperature; record clocks/power controls.
2. Stop unrelated GPU jobs and record all co-tenants if that is impossible.
3. Preserve the model and tokenizer revision, container digest, and serving flags.
4. Decide prefix-cache policy before running; do not silently change it between cases.
5. Keep sampling and requested output lengths constant across comparisons.
6. Use configured warmups and exclude them from aggregates.
7. Randomize or rotate case order with a recorded seed.
8. Use repeated trials and retain failures/timeouts.
9. Archive `metadata.json`, `observations.jsonl`, server metrics, logs, aggregates,
   analysis, charts, and the report together.
10. Report unavailable measurements and limited sample sizes explicitly.

## Artifact integrity

For publication, hash the source trace, configuration, server image, and results:

```bash
shasum -a 256 \
  examples/traces/mock-agent.jsonl \
  configs/benchmark-gpu.yaml \
  results/vllm-single-gpu/metadata.json \
  results/vllm-single-gpu/observations.jsonl
```

Do not publish full traces until repository content has been reviewed. Metadata-only
traces are safer but cannot be exact recorded replays; record that fidelity limitation.
