<div class="chapter-meta">Module 1 · Chapter 4</div>

# Training vs. Inference

<p class="lede">A model is trained once and served millions of times. The two workloads look superficially similar — same architecture, same weights, same tokenizer — but their performance characteristics are opposites. Confusing the two is the single most common source of bad inference decisions, including the perennial "just make the batch bigger" that works in training and fails in inference.</p>

## What each phase actually does

**Training** runs the forward pass, computes a loss, runs a backward pass, and updates weights:

1. Forward pass: inputs → predictions. Same as inference.
2. Backward pass: compute gradient of the loss with respect to every weight.
3. Optimizer step: update weights using the gradient and the optimizer state (Adam's first and second moments).

**Inference** runs only the forward pass. Weights are read-only; no gradients, no optimizer state, no backward pass.

## Memory — the biggest difference

For a model with $P$ parameters at bf16 (2 bytes), rough memory budget:

| Component | Training (Adam) | Inference |
|---|---|---|
| Weights (fp16/bf16 working) | $2P$ | $2P$ |
| Weights (fp32 master copy) | $4P$ | — |
| Gradients (fp32) | $4P$ | — |
| Adam moments (fp32, m and v) | $8P$ | — |
| Activations (all layers, mixed) | $O(b \cdot s \cdot d \cdot L)$ | $O(b \cdot s \cdot d)$ (one layer) |
| KV cache | — | $2 \cdot L \cdot b \cdot s \cdot n_{kv} \cdot d_h$ |

Total: training a 70 B model with Adam needs ~18 × 70 B = ~1.3 TB of GPU memory just for weights + optimizer state, before activations. Inference of the same model needs 140 GB for weights plus the KV cache. The 10 × ratio is why training fleets shard aggressively and inference fleets don't need to.

## Compute — opposite batch regimes

Training is batch-greedy. Big batches amortize the backward pass, keep gradients statistically stable, and saturate tensor cores. Batches of 4M tokens per step are routine for frontier training runs.

Inference is latency-constrained. A single user does not want to wait for 4M tokens to accumulate. Continuous batching (Chapter M6.2) reclaims some batching benefit by sharing the forward pass across concurrent requests, but the batch size is capped by what fits in KV memory and what SLO allows. Production batches are 8–128 — three orders of magnitude smaller than training.

## Precision — training needs more headroom

Training usually keeps a fp32 master copy of weights and runs matmuls in bf16 or fp16 — mixed precision. The fp32 copy exists because small gradient updates accumulate and would be lost in bf16. Inference happily runs weights in bf16, fp8, or even int4 end-to-end because there is no accumulation of small updates.

This is why inference can quantize aggressively (int4 weights are now routine) while training mostly stops at fp8.

## Scheduling — one job vs. thousands

A training run is one long job. Scheduling it is static: lay out the parallelism, run until done. An inference serving system is thousands of concurrent short jobs, each with its own prompt length and output length. Scheduling is the hard problem — it is what the inference engine (Module 6) is.

## Why training intuitions mislead inference engineers

- **"Bigger batch is always better"** — true in training (helps gradients), false in inference (hurts latency once compute-bound).
- **"Checkpointing saves memory"** — gradient checkpointing (recomputing activations in backward) is a training trick; there is no backward in inference.
- **"ZeRO stage 3 lets us run 70 B on 8 × A100"** — true in training (sharding optimizer state); irrelevant in inference (there is no optimizer state).
- **"FSDP makes large models fit"** — training construct; the inference analog is tensor parallelism (Chapter M6.6).

## Pitfalls

- **Sizing inference hardware with training rules of thumb.** "We trained 70 B on 128 H100s, so we need 128 H100s to serve" — usually wildly over-provisioned.
- **Importing a model with its training config.** Training configs often enable gradient checkpointing, dropout, and loss computation; make sure the inference loader disables them.
- **Assuming bf16 everywhere.** Many kernels favor fp8 or int8 for inference; keeping bf16 may leave 2–4× on the table.

<section class="faq">
<h2>FAQ</h2>
<h3>Can I serve a model while I'm still training it?</h3>
<p>Weights aren't thread-safe during optimizer updates. The standard production pattern is blue-green: serve from checkpoint N while training produces checkpoint N+1, then swap.</p>
<h3>Why doesn't ZeRO help inference?</h3>
<p>ZeRO shards optimizer state, gradients, and (at stage 3) weights across data-parallel replicas. Inference has no optimizer state or gradients, so ZeRO stages 1 and 2 reduce to nothing. For weight sharding, use tensor parallelism instead.</p>
<h3>Why does inference OOM at a batch that trained fine?</h3>
<p>Different memory profile. Training retains activations for all layers for the backward pass; inference does not. Inference's dominant non-weight memory is the KV cache — which has no training analog.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://arxiv.org/abs/1910.02054" target="_blank" rel="noreferrer">Rajbhandari et al. — ZeRO: Memory Optimizations for Training Trillion-Parameter Models</a></li>
<li><a href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noreferrer">Hoffmann et al. — Training Compute-Optimal LLMs (Chinchilla)</a></li>
<li><a href="https://blog.eleuther.ai/transformer-math/" target="_blank" rel="noreferrer">EleutherAI — Transformer Math 101</a></li>
</ul>
</section>

## What's next

With the contrast established, [Chapter 5](?module=m1&chapter=prefill-and-decode) drills into the inference-specific split of prefill and decode, which has no training analog and dictates nearly every serving optimization downstream.
