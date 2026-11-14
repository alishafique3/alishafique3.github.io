<div class="chapter-meta">Module 6 · Chapter 6</div>

# Distributed Parallelism

<p class="lede">M2.6 introduced the parallelism families — tensor, pipeline, expert, data. This chapter is the practitioner's guide: how to configure each in vLLM and SGLang, the honest cost of each, and how to pick a hybrid layout for a 70 B or MoE model that fits your fleet.</p>

## Tensor parallelism (TP)

Weights split across $N$ GPUs by column (Q, K, V, up-proj) or row (out-proj, down-proj). One AllReduce per transformer block. Latency-critical, requires NVLink.

Configure in vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --dtype bfloat16 --port 8000
```

Constraint: TP must divide `num_attention_heads` evenly. Llama 3 70B has 64 heads → TP of 1, 2, 4, 8, 16, 32, 64.

Cost: one AllReduce per attention block and one per FFN block. On NVLink 4 (450 GB/s), ~1 ms per AllReduce for Llama 3 70B at batch 16 — about 20 % of per-layer time. Negligible if your fabric is NVLink; dominant if it's PCIe.

## Pipeline parallelism (PP)

Give GPU 0 layers 1–20, GPU 1 layers 21–40, etc. Microbatches flow sequentially. Good for crossing nodes.

Configure in vLLM:

```bash
--tensor-parallel-size 4 \
--pipeline-parallel-size 2       # 2 pipeline stages, TP=4 within each
```

Cost: pipeline bubble (fill + drain) of roughly $(N - 1) / M$ of total time, where $M$ is microbatch count. For interactive serving with $M = 1$, the bubble is huge. For throughput workloads with $M = 32$, negligible.

## Expert parallelism (EP) — MoE only

Distribute experts across GPUs; route each token to the GPU owning its chosen expert via all-to-all. Essential for MoE models (DeepSeek V3, Mixtral) because replicating all experts on every GPU is memory-prohibitive.

Configure in vLLM for Mixtral:

```bash
--tensor-parallel-size 8      # Includes expert sharding across 8 GPUs
```

vLLM auto-detects MoE and shards experts. SGLang exposes `--expert-parallel-size` explicitly. Cost: one all-to-all per MoE layer — the fabric-hungriest collective. NVLink Switch System matters for 32+ GPU EP.

## Data parallelism (DP)

Replicate the entire model, distribute requests across replicas. Pure throughput. Zero coordination between replicas.

Configure at the orchestration layer — launch $N$ copies of the engine behind a load balancer. Each replica is independent.

## Hybrid layouts

Real fleets compose multiple axes. Common shapes:

- **TP = 8 within a node, DP across nodes.** Each 8-GPU node is one replica. The simplest pattern for 70 B models.
- **TP = 4 + PP = 2 across 2 nodes.** Weights span 8 GPUs, pipeline across the 2 nodes. Used when a model doesn't fit in a single-node TP but doesn't need all 8 GPUs of a second node.
- **TP = 8 + EP = 8 for MoE.** Weights tensor-parallel, experts also distributed.

Rule of thumb: keep TP within NVLink-connected GPUs, use PP (or DP) across nodes.

## Code-level view — what a TP engine does per block

For attention block with hidden $d$, head count $a$, TP degree $N$:

1. Each GPU holds weights for $a / N$ attention heads and $d / N$ FFN columns.
2. QKV projection: each GPU computes its shard of Q, K, V.
3. Attention: each GPU computes attention for its heads (its slice of KV cache).
4. Output projection: each GPU computes partial output; an **AllReduce** sums across GPUs.
5. FFN up-proj: each GPU computes its $d/N$ slice.
6. Activation: elementwise.
7. FFN down-proj: partial output; another **AllReduce**.

Two AllReduces per block × 80 blocks = 160 per forward pass. This is why NVLink matters.

## Picking a configuration — a worked example

Llama 3.1 70B, 4 × H100 (one node), interactive serving target:

- TP = 4 within the node.
- PP = 1 (one stage).
- DP replicas: 1 per node.
- Rationale: 4 × H100 (320 GB) fits 70 B bf16 (140 GB) + 128 GB headroom for KV. TP = 4 means two AllReduces per block at ~0.5 ms each over NVLink.

Same hardware, long-prompt RAG throughput target:

- TP = 2 within the node, DP = 2 — two independent serving replicas.
- Rationale: each 2 × H100 pair has enough memory for quantized 70 B. Two replicas doubles RPS at some latency cost.

## Pitfalls

- **TP > 8 within one chassis.** NVLink Switch System only exists for 8-way; above that you leave NVLink per-link and performance drops.
- **PP for interactive serving.** The pipeline bubble is a real latency hit. Use PP for throughput workloads or cross-node sizing, not for lowest TTFT.
- **Mixing precisions across TP shards.** All shards must be the same dtype. Cross-GPU communication assumes uniform precision.
- **EP without noticing load imbalance.** MoE routing can send disproportionate traffic to some experts. Watch the balance in your monitoring.

<section class="faq">
<h2>FAQ</h2>
<h3>Should I use TP or PP for a 70 B model?</h3>
<p>If it fits in one NVLink node, TP. If you need multiple nodes, TP within node + PP across nodes.</p>
<h3>Does continuous batching work with PP?</h3>
<p>Yes, but each microbatch moves through the pipeline. Scheduler complexity is higher.</p>
<h3>Can I combine TP and EP?</h3>
<p>Yes — standard for MoE at scale. vLLM and SGLang handle the combination automatically when configured.</p>
<h3>Does DP need special engine config?</h3>
<p>No — DP is orchestration. Launch N engine instances; put a load balancer in front. Each engine is oblivious to the others.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://arxiv.org/abs/1909.08053" target="_blank" rel="noreferrer">Shoeybi et al. — Megatron-LM (TP origins)</a></li>
<li><a href="https://arxiv.org/abs/1811.06965" target="_blank" rel="noreferrer">Huang et al. — GPipe (PP origins)</a></li>
<li><a href="https://docs.vllm.ai/en/stable/serving/distributed_serving.html" target="_blank" rel="noreferrer">vLLM — Distributed serving documentation</a></li>
<li><a href="https://docs.sglang.ai/backend/server_arguments.html" target="_blank" rel="noreferrer">SGLang — Server arguments documentation</a></li>
</ul>
</section>

## What's next

[Chapter 7 — Advanced Serving](?module=m6&chapter=advanced-serving) covers disaggregated prefill/decode, KV-aware routing, and offline batch workloads.
