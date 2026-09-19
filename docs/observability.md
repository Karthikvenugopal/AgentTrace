# Observability and export verification

AgentTrace exposes client metrics and traces separately from vLLM's server metrics. The
Compose stack provides an OpenTelemetry Collector, Prometheus, and Grafana; it does not
start a GPU server or download a model.

## Start the stack

```bash
docker compose config --quiet
docker compose up -d otel-collector prometheus grafana
docker compose ps
```

Verify each component rather than treating container startup as proof of export:

```bash
curl -fsS http://127.0.0.1:13133/
curl -fsS http://127.0.0.1:9090/-/ready
curl -fsS http://127.0.0.1:9090/api/v1/targets | python -m json.tool
```

Prometheus initially reports the AgentTrace and vLLM scrape targets down if neither
process is running. That is expected and visible, not a successful data export.

## Export client telemetry

Install the OTLP exporter and launch any CLI workflow with environment configuration:

```bash
pip install -e '.[telemetry]'
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
export AGENTTRACE_METRICS_PORT=9464
export AGENTTRACE_METRICS_HOST=0.0.0.0
export AGENTTRACE_JSON_LOGS=1
agenttrace mock-server --port 8010
```

The CLI appends `/v1/traces` to the OTLP base endpoint. Check the actual client metrics:

```bash
curl -fsS http://127.0.0.1:9464/metrics | grep '^agenttrace_'
docker compose logs --since=2m otel-collector
```

The collector's debug exporter prints completed spans. Expected span names include
`agent.execution`, `agent.step`, `llm.request`, `tool.call`, `replay.session`,
`replay.request`, and `benchmark.experiment`.

## Verify server metrics

For vLLM:

```bash
curl -fsS http://127.0.0.1:8000/metrics | grep -E 'vllm[:_]'
export AGENTTRACE_VLLM_METRICS_URL=http://127.0.0.1:8000/metrics
```

The adapter recognizes colon and underscore naming used by different releases. Its raw
name list and per-semantic availability are captured around benchmark trials. Absence
is recorded with a reason. No fallback fabricates queue/prefill/decode observations.

The bundled mock server exports only running and waiting request gauges. This exercises
discovery but is not native vLLM telemetry.

## Prometheus labels and dashboards

Client metric names begin `agenttrace_client_`; server series remain `vllm...`.
Labels are limited to component, configured model, status, token direction, and a fixed
tool-name vocabulary. Request/trace IDs, prompts, and repository paths appear in neither
label names nor values.

Grafana is available at `http://127.0.0.1:3000`. The provisioned overview shows request
rate, active client requests, and vLLM running/waiting requests. Prometheus remains the
source of truth; use its target page to diagnose missing panels.

## Correlation scope

OpenTelemetry spans and structured JSON logs carry experiment, trace, agent, and
request identifiers. Prometheus does not. The `X-Request-ID` header is sent to compatible
servers, but exact server-span linkage depends on backend support. Aggregate vLLM
histograms cannot be retrospectively attributed to an individual request.
