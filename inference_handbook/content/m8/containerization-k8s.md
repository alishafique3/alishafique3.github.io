<div class="chapter-meta">Module 8 · Chapter 2</div>

# Containerization and Kubernetes Deployment

<p class="lede">LLM serving in production runs in containers on Kubernetes. Three things make it different from a plain web service: GPU resource scheduling, giant model binaries fetched at runtime, and the NCCL / IB plumbing needed for multi-GPU inference. This chapter covers the Docker + K8s patterns that work.</p>

## Why containers (and why not)

Containers give you reproducibility, versioning, and fast rollback. For LLM workers the trade is:

- **Pro** — same engine version, driver stack, Python deps everywhere.
- **Pro** — Kubernetes handles health checks, rolling updates, resource limits.
- **Con** — GPU drivers must match on host. Container shares kernel; CUDA toolkit inside only works if the host NVIDIA driver supports that CUDA version.

Alternatives (bare metal, systemd units) are faster for single-GPU dev work. Containers win at &gt; 1 GPU, &gt; 1 team member.

## A production Dockerfile — the shape

```dockerfile
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NOT the weights — those come from object storage at startup
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

Image size: ~3–5 GB (CUDA + Python + vLLM). Model weights are fetched at pod startup.

## Why not bake weights into the image

A 70 B model is 140 GB. Baking into the image means:

- Image build takes 20+ minutes every change.
- Container registry bills pile up.
- Rolling updates must re-pull 140 GB per node.
- Cold starts are indistinguishable from image pulls.

Instead: pull weights from S3 / GCS / object storage into an EmptyDir volume at container start. Cache on fast local NVMe.

## The GPU scheduling bit

Pods declare GPU requests:

```yaml
resources:
  limits:
    nvidia.com/gpu: 2   # request 2 GPUs
```

For this to work the cluster needs the NVIDIA device plugin installed. The plugin exposes `nvidia.com/gpu` as a schedulable resource and injects the right device mounts into the container. Standard on EKS, GKE, AKS with GPU node pools.

## A minimal Helm chart — values.yaml

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct
  source: s3://my-weights/qwen-2.5-7b-instruct/
replicas: 4
gpu:
  type: nvidia.com/gpu
  count: 1
vllm:
  tensor_parallel_size: 1
  dtype: bfloat16
  max_model_len: 8192
  gpu_memory_utilization: 0.90
resources:
  requests: {memory: "48Gi", cpu: "8"}
  limits:   {memory: "64Gi", cpu: "16"}
service:
  type: ClusterIP
  port: 8000
```

The Helm chart renders a Deployment (replicas), a Service (ClusterIP), and optionally an HPA or KEDA scaler (M8.3).

## Multi-GPU and multi-node

**Tensor parallelism within a node** — one pod requests $N$ GPUs on the same host, `--tensor-parallel-size=N`. All happens inside one container.

**Pipeline parallelism across nodes** — pods on different hosts. Need NCCL to find each other; use the NVIDIA GPU Operator which auto-configures network fabric (InfiniBand or RoCE) in the pods.

**Kubernetes Jobs vs. Deployments** — long-lived serving uses Deployments. One-shot training/batch inference uses Jobs.

## The production-stack tutorials

The main [LLM-infra-from-scratch](https://github.com/alishafique3/LLM-infra-from-scratch) repository ships **17 production-stack tutorials** covering end-to-end Kubernetes deployment of vLLM-based LLM serving. Topics include:

- Install Kubernetes environment
- Minimal Helm installation
- Basic vLLM config
- Load model from persistent volume
- Launch multiple models
- Offload KV cache
- Remote shared KV cache
- Benchmarks (single and multi-GPU)
- LoRA-enabled installation
- Horizontal autoscaling
- Basic pipeline parallel
- Disaggregated prefill
- KV-aware and prefix-aware routing
- Engine sleep / wakeup mode
- KEDA autoscaling

Each tutorial is a reproducible set of manifests. Treat them as the canonical "how to run this in Kubernetes" reference; this chapter is the mental model.

## Pitfalls

- **Driver / CUDA version mismatches.** Container CUDA 12.6 on a host driver that only supports CUDA 12.2 → broken. Always match.
- **No liveness probe.** When vLLM hangs (rare but happens), K8s doesn't restart it. Add a probe hitting `/health`.
- **Pulling weights in every container restart.** Cache on local NVMe or a hostPath volume.
- **Ignoring the pod disruption budget.** Default is unlimited disruption — K8s can evict every replica during a node drain, causing brief total outage. Set a PDB.

<section class="faq">
<h2>FAQ</h2>
<h3>Do I need the NVIDIA GPU Operator?</h3>
<p>For production clusters, yes. Handles driver installation, device plugin, DCGM exporter, MIG configuration — a lot that is painful to install manually.</p>
<h3>Can I run Kubernetes on-prem for LLMs?</h3>
<p>Yes — many teams do. Rancher, OpenShift, and vanilla kubeadm all work with the GPU Operator.</p>
<h3>What is a sidecar for LLM pods?</h3>
<p>A second container in the same pod that runs alongside the engine. Common for metrics exporters, log forwarders, or specialized proxies.</p>
<h3>How do I debug a pod that won't start?</h3>
<p><code>kubectl describe pod &lt;name&gt;</code> for events, <code>kubectl logs</code> for engine output. Most failures are OOM (GPU or memory) or driver mismatches.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://github.com/vllm-project/production-stack" target="_blank" rel="noreferrer">vLLM Production Stack — K8s reference deployments</a></li>
<li><a href="https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/overview.html" target="_blank" rel="noreferrer">NVIDIA GPU Operator documentation</a></li>
<li><a href="https://github.com/alishafique3/LLM-infra-from-scratch" target="_blank" rel="noreferrer">LLM-infra-from-scratch — 17 K8s tutorials</a></li>
</ul>
</section>

## What's next

[Chapter 3](?module=m8&chapter=load-balancing-autoscaling) covers the gateway + autoscaler side: routing, HPA, KEDA, cold starts.
