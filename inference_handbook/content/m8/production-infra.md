<div class="chapter-meta">Module 8 · Chapter 1</div>

# Understanding Production LLM Inference Infrastructure

<p class="lede">An inference engine is not a serving stack. A production LLM service has four layers above the engine: a <strong>gateway</strong> for auth and routing, <strong>inference workers</strong> running the engine, <strong>model storage</strong> for weights, and a <strong>control plane</strong> that orchestrates everything. This chapter is the reference architecture every serious team ends up with.</p>

## The four-layer reference architecture

```
                 Client
                   │
            ┌──────┼──────┐
            │   Gateway   │  ← auth, rate limit, routing, logging
            └──────┬──────┘
                   │
       ┌───────────┼───────────┐
       │           │           │
   ┌───┴───┐  ┌────┴───┐  ┌────┴───┐
   │Worker │  │Worker  │  │Worker  │  ← vLLM/SGLang/TRT-LLM
   │(GPU)  │  │(GPU)   │  │(GPU)   │
   └───┬───┘  └────┬───┘  └────┬───┘
       │           │           │
       └───────────┼───────────┘
                   │
         ┌─────────┴──────────┐
         │   Model storage    │  ← S3/GCS/Ceph; model registry
         │   (safetensors)    │
         └────────────────────┘

         ┌────────────────────┐
         │   Control plane    │  ← K8s, deploys, autoscaling, observability
         └────────────────────┘
```

Each layer has one job. Keep them decoupled.

## Layer 1 — the gateway

Sits in front of everything. Responsibilities:

- **Authentication** — API keys, OAuth, mTLS between services.
- **Rate limiting** — RPM/TPM per key, concurrent-request caps.
- **Request routing** — pick the right worker pool (model, version, tenant).
- **Observability** — structured logging, request IDs, trace propagation.
- **Protocol translation** — OpenAI ↔ Anthropic ↔ internal.
- **TLS termination** — HTTPS edge.

Common implementations: Envoy, Traefik, Kong, nginx. For LLM-specific features (model routing, cache-aware routing): LiteLLM, Portkey, Kong Gateway with LLM plugin.

## Layer 2 — inference workers

The engines themselves. A worker is one process (or container) running vLLM / SGLang / TensorRT-LLM / TGI. Responsibilities:

- Load the model into GPU memory.
- Process incoming requests via the engine.
- Emit telemetry to Prometheus.
- Fail gracefully (health probes).

Workers are fungible. A replica set behind the gateway scales horizontally. Each worker is a pod on Kubernetes; GPU nodes host them.

## Layer 3 — model storage

Model weights (10–200 GB) live in object storage, not in container images. Containers pull weights at startup. Storage layers:

- **Blob store** — S3, GCS, Azure Blob, or self-hosted Ceph/MinIO.
- **Model registry** — metadata index (model → version → location → config). Hugging Face Hub, MLflow, or internal.
- **Local cache** — fast NVMe on each GPU node, warmed on pod startup.

Cold start cost scales with model size and storage bandwidth. A 140 GB model from S3 at 10 Gbps takes ~2 minutes. Pre-warming pods keeps TTFT low under autoscale.

## Layer 4 — the control plane

Everything that manages the other three layers:

- **Kubernetes** — the de facto orchestration.
- **Helm** — templated deployments.
- **Autoscalers** — HPA, KEDA (M8.3).
- **Observability** — Prometheus, Grafana, Loki, Tempo (M8.4).
- **Secret management** — Vault, cloud KMS.
- **CI/CD** — ArgoCD, Flux, GitHub Actions for model/engine releases.

The control plane is where InferenceOps (M8.7) lives.

## A minimal production stack

For a mid-scale deployment (10–100 GPUs):

| Layer | Component |
|---|---|
| Gateway | Envoy or LiteLLM behind an HTTPS LB |
| Worker | vLLM, 1 pod per GPU replica |
| Storage | S3 for weights; EBS NVMe local cache |
| Orchestration | Kubernetes (EKS, GKE, or on-prem) |
| Metrics | Prometheus + Grafana + DCGM Exporter |
| Logs | Loki or CloudWatch |
| Traces | OpenTelemetry → Tempo or Jaeger |
| CI/CD | ArgoCD + Helm |

This is boring infrastructure. That is the point. The LLM-specific bits (vLLM, DCGM, LiteLLM) fit into the standard Kubernetes world.

## What makes LLM serving different from "normal" serving

- **GPU scheduling.** Pods request GPU resources, not just CPU/memory.
- **Huge model binaries.** Multi-GB weights need pre-warming and fast local disk.
- **Long request lifetimes.** Streaming connections hold open for seconds; load balancers must tolerate it.
- **Expensive capacity.** A GPU replica is &#36;5–15/hr. Idle costs real money. Autoscaling policy is tighter.
- **Stateful by prefix.** Prefix caching means request routing affects throughput — round-robin can halve throughput.

Everything else is standard web-service engineering.

## Pitfalls

- **Treating the gateway as optional.** vLLM has no auth. Exposing it directly is a security incident waiting.
- **Baking weights into the container image.** Image becomes 100+ GB. Builds break, registries groan, cold starts spike.
- **Running one giant replica instead of many.** Rolling updates, failure isolation, canary deploys all need multiple replicas.
- **No observability from day one.** The day you need to debug a P99 TTFT regression is not the day to start collecting metrics.

<section class="faq">
<h2>FAQ</h2>
<h3>Do I need Kubernetes?</h3>
<p>For production at scale, effectively yes. Docker Compose works for single-node dev; Nomad/ECS work for smaller teams. K8s is the dominant option for GPU workloads in 2026.</p>
<h3>Is it OK to put LiteLLM in front of vLLM?</h3>
<p>Yes — this is the common multi-provider pattern. LiteLLM handles auth, rate limiting, and provider translation; vLLM does the inference.</p>
<h3>How large should my model cache be?</h3>
<p>Big enough to hold your active model set, not every model that exists. For a one-model deploy: just the current model + the previous version (for rollback).</p>
<h3>What about edge deployment?</h3>
<p>Different architecture — single device, single model, no orchestration. See M3.1 and M2.1.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/" target="_blank" rel="noreferrer">Kubernetes — Scheduling GPUs</a></li>
<li><a href="https://envoyproxy.io/" target="_blank" rel="noreferrer">Envoy Proxy</a></li>
<li><a href="https://github.com/BerriAI/litellm" target="_blank" rel="noreferrer">LiteLLM — LLM gateway</a></li>
</ul>
</section>

## What's next

[Chapter 2](?module=m8&chapter=containerization-k8s) covers containerization and Kubernetes specifics, including the vLLM production-stack tutorials. [Chapter 5](?module=m8&chapter=load-testing-slo) ties capacity math back to real load tests.
