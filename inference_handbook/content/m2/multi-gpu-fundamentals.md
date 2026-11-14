<div class="chapter-meta">Module 2 · Chapter 6</div>

# Multi-GPU Inference Fundamentals

<p class="lede">Once a model exceeds a single GPU's memory, or you want more throughput than one GPU can produce, you shard. The three most important axes are <strong>tensor parallelism</strong>, <strong>pipeline parallelism</strong>, and <strong>expert parallelism</strong> (for MoE). Each slices the problem differently, with different communication costs. This chapter is the overview — Chapter M6.6 handles the engine-level mechanics.</p>

## Three sharding strategies

- **Tensor parallelism (TP)** — Split each weight matrix across GPUs. Every forward pass does an AllReduce across GPUs after attention and FFN blocks. Best for latency, assumes NVLink.
- **Pipeline parallelism (PP)** — Give each GPU a contiguous subset of layers. Microbatch flows sequentially across the pipeline. Best for crossing nodes, costs latency to fill the pipeline.
- **Expert parallelism (EP)** — Only for MoE. Distribute experts across GPUs; route each token to the GPU owning its chosen expert via all-to-all.
- **Data parallelism (DP)** — Replicate the entire model, distribute requests. Pure throughput; no help for a model that doesn't fit on one GPU.

These compose: 2 nodes × 8 GPUs × hybrid PP=2 + TP=8 for a frontier model is routine.

## Tensor parallelism in one picture

A matmul $Y = X W$ where $W$ has columns split across $N$ GPUs produces partial outputs $Y_i = X W_i$ on GPU $i$. An AllReduce reconstructs the full $Y$. For attention, Q, K, V projections are column-split (independent per head), attention output is row-split, output projection reduces — one AllReduce per attention block. Same pattern for FFN with up/down projections.

Memory: weights divide by $N$, KV cache divides by $N$ (each shard holds its heads' KV). Latency: adds one AllReduce per block, as computed in M2.5.

## Pipeline parallelism in one picture

Layers 1–20 live on GPU 0, layers 21–40 on GPU 1, etc. A request's activations flow through each stage. To keep all stages busy, microbatches are pipelined — GPU 0 starts microbatch 2 while GPU 1 processes microbatch 1. The pipeline bubble (fill + drain time) is roughly $(N - 1) / M$ of total time, where $M$ is microbatch count.

Memory: weights divide by $N$. Latency: for an interactive request with few microbatches, the bubble dominates and you lose latency compared to TP.

## When to use which

| Situation | Preferred shape |
|---|---|
| Model fits on 1 GPU, need throughput | DP (one worker per GPU) |
| Model needs 2–8 GPUs, within one NVLink node | Pure TP |
| Model needs 16+ GPUs, 2+ nodes | TP within node + PP across nodes |
| MoE model | TP + EP (experts split across GPUs) |
| Very long context, tight TTFT | TP within node + disaggregated prefill/decode |

## Collectives in 30 seconds

- **AllReduce** — Every GPU sends its partial, every GPU gets the sum. Used in TP.
- **AllGather** — Every GPU sends its slice, every GPU gets the full concatenated result. Used in some TP variants, SP (sequence parallel).
- **ReduceScatter** — Like AllReduce but each GPU only keeps its slice. Memory-efficient pair with AllGather for ZeRO-style partitioning.
- **Send/Recv** — Point-to-point. Used in PP.
- **AllToAll** — Every GPU sends a distinct slice to every other GPU. Used in EP for MoE routing.

NCCL is the standard implementation; it chooses an efficient algorithm (ring, tree, or NVLS) per operation and topology.

## A realistic example — Llama 3 70B serving

Model: 140 GB bf16, 80 layers.

- 2 × H200 (141 GB each): TP = 2. Fits comfortably; one AllReduce per block, ~1 ms each over NVLink. Simple, low-latency.
- 4 × H100 (80 GB each): TP = 4. Two AllReduces per block, higher communication cost, more KV capacity available. Good for high-concurrency.
- 8 × H100 across 2 nodes: TP = 4 within each node + PP = 2 across nodes. Weights live comfortably; pipeline bubble is the new cost.
- 1 × B200 (180 GB): no sharding needed. Highest per-GPU throughput, highest per-GPU cost.

## Pitfalls

- **Mixing TP and PP on the same node.** Wastes NVLink on PP sends that could have gone over PCIe. Keep TP within NVLink-connected GPUs.
- **Picking TP = N without checking.** N must divide the number of attention heads evenly; many models (e.g., 32 heads) do not accept TP = 3 or TP = 5.
- **Assuming PP is latency-neutral.** It is not — pipeline fill adds $(N - 1) \times \text{stage latency}$ to the first-token time.

<section class="faq">
<h2>FAQ</h2>
<h3>Should I use TP or PP for a 70B model?</h3>
<p>If it fits within one NVLink-connected node, use TP — simpler and lower latency. If you cross nodes, use TP within each node plus PP across nodes.</p>
<h3>Can I shard across different GPU models?</h3>
<p>Not cleanly for TP — all shards should be identical to balance. Mixing H100 and H200 in a TP group means the H100 is your bottleneck.</p>
<h3>Does pipeline parallelism work with continuous batching?</h3>
<p>Yes, but the pipeline-bubble overhead adds latency to interactive requests. Works best when microbatches are plentiful.</p>
<h3>What is the TP / head divisibility constraint?</h3>
<p>TP degree must evenly divide <code>num_attention_heads</code>. Llama-3 70B has 64 heads → TP of 1, 2, 4, 8, 16, 32, or 64 works; TP = 3 or 5 does not.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://arxiv.org/abs/1909.08053" target="_blank" rel="noreferrer">Shoeybi et al. — Megatron-LM: Training Multi-Billion Parameter Models (TP origins)</a></li>
<li><a href="https://arxiv.org/abs/1811.06965" target="_blank" rel="noreferrer">Huang et al. — GPipe: Efficient Training of Giant Neural Networks (PP)</a></li>
<li><a href="https://docs.vllm.ai/en/stable/serving/distributed_serving.html" target="_blank" rel="noreferrer">vLLM — Distributed inference documentation</a></li>
</ul>
</section>

## What's next

The deep dive on each parallelism, including code and vLLM configuration, lives in [M6 Ch. 6](?module=m6&chapter=distributed-parallelism). The communication cost math is in [M2 Ch. 5](?module=m2&chapter=gpu-interconnects).
