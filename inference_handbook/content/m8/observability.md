<div class="chapter-meta">Module 8 · Chapter 4</div>

# LLM Observability and Monitoring

<p class="lede">You cannot operate what you cannot see. Observability is the practice of collecting <strong>metrics</strong>, <strong>logs</strong>, and <strong>traces</strong> from every layer of the stack so that when something breaks, you know what, when, and why. For LLM serving, two metric classes matter: engine metrics (TTFT, TPOT, goodput) and GPU metrics (SM utilization, HBM bandwidth, power). This chapter covers the dashboards every production team should have.</p>

## The three pillars

| Pillar | Tool | What it answers |
|---|---|---|
| Metrics | Prometheus + Grafana | How is the system behaving now and over time? |
| Logs | Loki / ELK / CloudWatch | What did a specific request do? |
| Traces | OpenTelemetry + Tempo/Jaeger | Where did time go for a slow request? |

All three are needed. Metrics alone can't debug one slow request; logs alone can't show trends; traces alone are noisy without metrics for context.

## Metrics every LLM service should expose

### Engine-level (per request)

- `ttft_seconds` histogram, labeled by model.
- `tpot_seconds` histogram.
- `e2e_seconds` histogram.
- `prompt_tokens`, `completion_tokens` counters.
- `num_requests_waiting` gauge — queue depth.
- `gpu_cache_usage_perc` gauge.
- `requests_total`, `requests_errors_total` counters, by status.

### GPU-level (from DCGM)

- `DCGM_FI_DEV_GPU_UTIL` — SM utilization %.
- `DCGM_FI_DEV_MEM_COPY_UTIL` — HBM bandwidth %.
- `DCGM_FI_DEV_POWER_USAGE` — current draw (W).
- `DCGM_FI_DEV_FB_USED` — HBM allocation.
- `DCGM_FI_DEV_SM_CLOCK` — SM clock frequency.

### System-level

- `pod_restarts_total` — stability.
- `cpu_usage`, `memory_usage` — rarely the bottleneck but useful.
- Network: ingress/egress bandwidth.

## Four dashboards every team should build

### 1. Service health dashboard

- P50 / P95 / P99 TTFT and TPOT by model.
- Goodput (requests within SLO).
- Error rate by status code.
- Queue depth.

Target audience: on-call.

### 2. GPU utilization dashboard

- SM utilization per GPU.
- HBM bandwidth utilization.
- Power draw.
- GPU memory allocation.

Target audience: infra engineers.

### 3. Capacity dashboard

- RPS per replica over time.
- Replica count.
- Autoscaler events.
- Headroom (current RPS vs. measured ceiling).

Target audience: capacity planners.

### 4. Cost dashboard

- GPU-hours consumed per hour.
- &#36;/million tokens based on current pricing.
- Utilization-adjusted cost.
- Comparison vs. API baselines.

Target audience: finance + eng leads.

## Setting up the pipeline

### vLLM metrics

vLLM exposes Prometheus metrics at `/metrics` by default. Scrape config:

```yaml
scrape_configs:
- job_name: vllm
  kubernetes_sd_configs:
  - role: pod
    namespaces: {names: [serving]}
  relabel_configs:
  - source_labels: [__meta_kubernetes_pod_label_app]
    regex: vllm
    action: keep
```

### DCGM Exporter

NVIDIA ships DCGM Exporter — a DaemonSet that scrapes every GPU and exposes metrics. Install via the NVIDIA GPU Operator (M8.2) which bundles it.

### Grafana dashboards

Ready-made dashboards:

- [vLLM Grafana dashboard](https://github.com/vllm-project/vllm/tree/main/examples) — importable JSON.
- NVIDIA DCGM dashboard (Grafana Marketplace ID 12239) — GPU vitals.

Import both; combine into one view per cluster.

## Tracing a slow request

A request flows: gateway → worker queue → scheduler → model forward → sampler → stream. OpenTelemetry spans let you see time per hop:

```python
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("llm.request") as span:
    span.set_attribute("model", "llama-3-70b")
    span.set_attribute("prompt_tokens", 850)
    with tracer.start_as_current_span("llm.prefill"):
        prefill(...)
    with tracer.start_as_current_span("llm.decode"):
        decode(...)
```

In production, OTel SDK auto-instruments HTTP, gRPC, and most Python frameworks. You add LLM-specific spans.

## Alerting

Alert on *symptoms*, not causes. Good alerts:

- P99 TTFT &gt; 1.5× SLO for 5 minutes.
- Error rate &gt; 2 % for 5 minutes.
- GPU replica count < desired for 10 minutes.
- Any pod OOM in the last hour.

Bad alerts:

- CPU usage &gt; 80 % (doesn't matter for GPU workloads).
- Memory usage &gt; 90 % (vLLM pre-allocates).
- Specific NCCL error codes (too noisy).

## Logging best practices

- **Structured JSON logs.** Grep and jq are easier than parsing text.
- **Request ID in every log line.** Correlate across services.
- **Never log full prompts by default.** User PII leaks. Log token counts + request metadata.
- **Sample at high volume.** 1 % sampling of successful requests + 100 % of errors is a common pattern.

## Pitfalls

- **Scraping too frequently.** Prometheus 1-second scrape per worker causes problems at scale. 15 s is fine for LLM metrics.
- **Logging prompts and completions.** Privacy incident. Hash or truncate.
- **Dashboards without labels.** A P99 TTFT chart without per-model breakdown misses the one slow model.
- **Alerting on raw GPU utilization.** Useful for capacity planning, noisy for alerts.
- **No traces.** When a request is unexpectedly slow, metrics show the symptom; only traces show where the time went.

<section class="faq">
<h2>FAQ</h2>
<h3>Can I use Datadog / New Relic instead?</h3>
<p>Yes — they have LLM integrations now. Trade cost for turnkey.</p>
<h3>How do I get application traces into my OTel pipeline?</h3>
<p>vLLM supports OTel export via <code>--otlp-traces-endpoint</code>. Point it at your collector.</p>
<h3>What's the DCGM Exporter overhead?</h3>
<p>Negligible. ~0.1 % GPU overhead per scrape.</p>
<h3>Should I log GPU metrics per request?</h3>
<p>No — they're aggregate, not per-request. Log request metrics per request; sample GPU metrics on a schedule.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://docs.vllm.ai/en/stable/serving/metrics.html" target="_blank" rel="noreferrer">vLLM — Metrics reference</a></li>
<li><a href="https://github.com/NVIDIA/dcgm-exporter" target="_blank" rel="noreferrer">NVIDIA DCGM Exporter</a></li>
<li><a href="https://opentelemetry.io/" target="_blank" rel="noreferrer">OpenTelemetry</a></li>
<li><a href="https://grafana.com/oss/tempo/" target="_blank" rel="noreferrer">Grafana Tempo — distributed tracing</a></li>
</ul>
</section>

## What's next

[Chapter 5](?module=m8&chapter=load-testing-slo) is the companion: how to generate synthetic load, measure the metrics in this chapter under real traffic, and set the SLOs the alerts fire on.
