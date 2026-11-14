<div class="chapter-meta">Module 3 · Chapter 3</div>

# GPU Selection and Memory Sizing

<p class="lede">The first production decision is picking the GPU and sizing its memory for your target model + concurrency. Pick wrong and you'll be OOM at 10 RPS or paying for compute you don't use. The math is not complicated — but it has to add up before you order the hardware.</p>

## The data-center GPU landscape (2026)

| GPU | HBM | HBM BW | BF16 TFLOPs | FP8 TFLOPs | TDP |
|---|---|---|---|---|---|
| A100 80 GB | 80 GB | 2.0 TB/s | 312 | — | 400 W |
| L40S | 48 GB | 0.9 TB/s | 362 | 733 | 350 W |
| H100 SXM | 80 GB | 3.35 TB/s | 1,979 | 3,958 | 700 W |
| H200 SXM | 141 GB | 4.8 TB/s | 1,979 | 3,958 | 700 W |
| B200 | 180 GB | 8.0 TB/s | 4,500 | 9,000 | 1,000 W |

Two observations: (1) H200 is a **memory refresh** of H100 — same compute, +76% capacity, +43% bandwidth. It's a near drop-in replacement that immediately relieves memory pressure on 70B-class models. (2) B200 is a generational leap — 2.3× FP8 throughput and 2.4× bandwidth over H100 — and it introduces FP4 as a first-class dtype.

## What lives in GPU memory during inference

Four buckets. Sum them before you pick the card.

1. **Model weights** — fixed per model + dtype.
2. **KV cache** — scales with batch × context × model config.
3. **Activations** — scales with batch × hidden × intermediate; typically small during inference since only one layer's activation is live.
4. **CUDA kernel workspace** — ~1–2 GB fixed overhead for the allocator, kernels, and runtime.

### The weights formula

$$ \text{weights bytes} = P \times \text{bpp} $$

where $P$ is parameter count and bpp (bytes per parameter) is 2 for BF16/FP16, 1 for FP8/INT8, 0.5 for INT4/FP4.

### The KV cache formula

For a model with $L$ layers, $n_{kv}$ KV heads, $d_h$ head dimension, batch $B$, sequence length $s$, and precision $b_{\text{elem}}$ bytes per element:

$$ \text{KV bytes} = 2 \cdot L \cdot B \cdot s \cdot n_{kv} \cdot d_h \cdot b_{\text{elem}} $$

The leading 2 accounts for keys *and* values. Note it depends on *KV* heads (not attention heads) — GQA models have far fewer KV heads than Q heads, which is the single most important KV-cache optimization in the architecture era.

## Worked example — Llama-3 70B at 8k context

Config: $L = 80$, $n_{kv} = 8$ (GQA), $d_h = 128$, BF16.

- Weights: &#36;70 \times 10^9 \times 2 = 140 \text{ GB}$ — does *not* fit on one H100 (80 GB).
- KV per token per sequence: &#36;2 \cdot 80 \cdot 8 \cdot 128 \cdot 2 = 327{,}680$ bytes ≈ 0.31 MB.
- KV at 8k context, batch = 16: $0.31 \times 8192 \times 16 \approx 40.5 \text{ GB}$.
- Workspace + activations: ≈ 4 GB.
- **Total: ~184 GB.**

Options:

- 2 × H200 (282 GB total) — comfortable. TP = 2.
- 2 × H100 (160 GB) — doesn't fit.
- 4 × H100 (320 GB) — fits, TP = 4, more NVLink traffic.
- 1 × B200 (180 GB) — just barely; no headroom for traffic spikes.
- 2 × H100 + FP8 quantization (70 GB weights + ~40 GB KV + 4 GB) = 114 GB — fits with headroom; see [M4 Ch. 2](?module=m4&chapter=quantization-fundamentals).

## Pitfalls

- **Forgetting KV cache scales with concurrency.** A memory budget that works for one user explodes at 32 concurrent users.
- **Reading num_attention_heads instead of num_key_value_heads.** GQA saves ~4–8× on KV — using the wrong number inflates your estimate.
- **Picking H100 to save money, then paying 2× in NVLink shards.** H200's extra 60 GB often eliminates a TP shard entirely and lowers TPOT.
- **Reserving zero headroom.** Leave 10–15% for allocator fragmentation and temporary spikes.

<section class="faq">
<h2>FAQ</h2>
<h3>H100 or H200 for Llama-3 70B?</h3>
<p>H200 fits the model unquantized on 2 GPUs with comfortable KV headroom, usually at a lower effective &#36;/token. H100 forces more sharding or quantization.</p>
<h3>Is B200 worth it for inference today?</h3>
<p>If your model fits once (≤ 180 GB) and you can use FP4, yes — biggest per-GPU throughput on the market. Supply is tight in 2026.</p>
<h3>Why pick L40S?</h3>
<p>Cheapest NVIDIA SKU with FP8. Great for ≤ 30 B models, high-throughput RAG workloads, or cost-sensitive inference where latency is relaxed.</p>
<h3>How do I size KV cache without doing arithmetic every time?</h3>
<p>Use the <a href="https://alishafique3.github.io/kv-cache-calculator/" target="_blank" rel="noreferrer">KV Cache Calculator</a> — it handles MHA, GQA, MLA, and hybrid variants across a 55-model catalog.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://www.nvidia.com/en-us/data-center/h200/" target="_blank" rel="noreferrer">NVIDIA H200 product page</a></li>
<li><a href="https://www.cloudzero.com/blog/h100-gpu-cost/" target="_blank" rel="noreferrer">CloudZero — H100 cloud pricing comparison 2026</a></li>
<li><a href="https://alishafique3.github.io/memory-calculator/" target="_blank" rel="noreferrer">Memory Calculator</a> and <a href="https://alishafique3.github.io/kv-cache-calculator/" target="_blank" rel="noreferrer">KV Cache Calculator</a> — interactive sizing tools</li>
</ul>
</section>

## What's next

Use the [Memory Calculator](https://alishafique3.github.io/memory-calculator/) to iterate configurations, and the [KV Cache Calculator](https://alishafique3.github.io/kv-cache-calculator/) for per-attention-variant KV math. Continue to [Chapter 4 — Capacity Planning](?module=m3&chapter=capacity-planning) for the RPS/TPS side.
