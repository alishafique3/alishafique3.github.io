<div class="chapter-meta">Module 8 · Chapter 3</div>

# Load Balancing, Request Routing, and Autoscaling

<p class="lede">Once you have multiple replicas, two questions arise: how are incoming requests distributed, and how does the replica count follow traffic? Default web-service answers (round-robin + CPU-based HPA) are wrong for LLM serving. This chapter covers what actually works.</p>

## Three routing strategies

### Round-robin — the wrong default

Fair, simple, bad for LLM. A request's warm KV cache (from turn 1 of a chat) lives on *one* replica; round-robin sends turn 2 to a different replica that must re-prefill. Prefix cache hit rate drops to ~0.

Use round-robin only for stateless fresh requests.

### Least-connections — reasonable for mixed workloads

Route to the replica with the fewest in-flight requests. Avoids piling up on a slow replica. Doesn't help prefix caching.

### KV-aware / session-affinity — the LLM default

Route a continuation to the replica that prefilled the earlier turn. Hash by session ID, user ID, or system-prompt-prefix hash. Needs a stateful gateway (LiteLLM, SGLang router, Portkey).

Impact: on multi-turn chat, prefix cache hit rate goes from &lt; 5 % to 60–80 %. TPOT drops ~30 % on turns 2+.

## The autoscaling problem

LLM workloads are expensive and bursty. You want to scale up fast on traffic spikes, scale down slowly to avoid thrashing. Kubernetes gives you two primary tools:

### HPA (Horizontal Pod Autoscaler)

Scales based on a metric crossing a threshold. The default CPU metric is useless for GPU serving — the CPU is barely used.

Useful HPA metrics for LLM:

- `vllm:num_requests_waiting` (queue depth)
- `vllm:gpu_cache_usage_perc` (KV cache utilization)
- Custom: tokens per second per replica

Example — HPA on queue depth:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric: {name: vllm_num_requests_waiting}
      target: {type: AverageValue, averageValue: "5"}
```

### KEDA (Kubernetes Event-Driven Autoscaling)

More flexible than HPA. Supports Prometheus queries, SQS queue depth, Pub/Sub backlog, cron schedules, and most importantly — scale-to-zero (HPA minimum is 1).

```yaml
triggers:
- type: prometheus
  metadata:
    serverAddress: http://prometheus:9090
    metricName: vllm_waiting
    threshold: "5"
    query: avg(vllm:num_requests_waiting)
```

KEDA + scale-to-zero is how you cut GPU bill for intermittent workloads.

## Cold starts

Scale-up has a real cost:

| Phase | Time |
|---|---|
| K8s scheduling | 10–30 s |
| Image pull (if not cached) | 30 s – 5 min |
| Weight download (140 GB from S3 @ 10 Gbps) | ~2 min |
| Engine init + model load | 30–90 s |
| CUDA graph capture, warm-up | 20–60 s |
| **Total cold start** | **2–10 min** |

Mitigations:

- **Warm pool** — keep N replicas always running, even at low traffic.
- **Image caching** — pre-pull images on all GPU nodes (DaemonSet).
- **Local NVMe weight cache** — hostPath volume shared across pods.
- **Faster weight format** — safetensors with mmap is faster than loading via HF transformers naively.

## Horizontal vs. vertical scaling

Kubernetes autoscales *horizontally* (more replicas). LLM replicas are GPU-coupled — you can't add half a GPU to an existing pod. So the knob is number of replicas, not size of replicas.

Vertical scaling (bigger GPU per replica) is an engine config change, not an autoscaler concern.

## A realistic autoscaler policy

For a chat workload on Llama 3.1 70B:

- Min replicas: 2 (resilience).
- Max replicas: 20.
- Target metric: `vllm:gpu_cache_usage_perc` average, threshold 70 %.
- Scale-up cooldown: 30 s (fast up).
- Scale-down cooldown: 300 s (slow down to avoid thrashing).
- Scale-down policy: 10 % per minute max.

The slow scale-down is crucial. LLM traffic has brief lulls that would flap-scale otherwise, causing cold-start pain when the next burst arrives.

## Pitfalls

- **Scaling on CPU.** The CPU is idle. The metric is meaningless. Always use GPU or application-level metrics.
- **Round-robin with prefix caching.** Cache hit rate drops to zero. Use a stateful router.
- **HPA with min=1.** Loses resilience during restarts. Use min=2 even for small deployments.
- **Scale-to-zero without warming.** First request after idle gets a 5-minute cold start. Users hate it.
- **No PDB (PodDisruptionBudget).** Node drain evicts all replicas simultaneously, causing an outage window.

<section class="faq">
<h2>FAQ</h2>
<h3>Can I use HPA + KEDA together?</h3>
<p>KEDA is built on HPA. You configure KEDA; it manages the HPA for you.</p>
<h3>What triggers scale-up fastest?</h3>
<p>Queue-depth metrics. Latency is slow to respond; queue depth reacts in the first affected request.</p>
<h3>Does session affinity break during rollouts?</h3>
<p>Yes — when a replica is replaced, its sessions lose affinity. Mitigation: graceful drain + short rolling window.</p>
<h3>How do I handle sudden traffic spikes?</h3>
<p>Combine: large warm pool (expensive) + fast scale-up (reactive) + queue tolerance at the gateway. There is no magic.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/" target="_blank" rel="noreferrer">Kubernetes HPA documentation</a></li>
<li><a href="https://keda.sh/" target="_blank" rel="noreferrer">KEDA — Kubernetes Event-Driven Autoscaling</a></li>
<li><a href="https://docs.vllm.ai/en/stable/serving/metrics.html" target="_blank" rel="noreferrer">vLLM — Metrics exposed for autoscaling</a></li>
</ul>
</section>

## What's next

[Chapter 4](?module=m8&chapter=observability) covers the metrics that feed the autoscaler and the dashboards that keep you honest.
