<div class="chapter-meta">Module 2 · Chapter 3</div>

# GPU Compute and Memory Hierarchy

<p class="lede">A GPU's compute throughput and its memory bandwidth are specified in different units and quoted on different graphs, but they are the <em>same budget</em> for most LLM work. Every byte that moves from HBM to the SMs is a byte you are paying for. This chapter maps the memory tiers and the compute throughput so you can tell at a glance whether your bottleneck is one or the other.</p>

## The memory tiers

| Tier | Size (per GPU) | Latency | Bandwidth |
|---|---|---|---|
| Registers (per thread) | ~0.25 KB each · 20 MB total on H100 | ~1 cycle | > 100 TB/s |
| Shared memory / L1 (per SM) | ~228 KB (H100) | ~20 cycles | ~20 TB/s aggregate |
| L2 cache (chip-wide) | 50 MB (H100) / 60 MB (H200) / 128 MB (B200) | ~200 cycles | ~10 TB/s |
| HBM (device memory) | 80 GB (H100) / 141 GB (H200) / 180 GB (B200) | ~400 cycles | 3.35 / 4.8 / 8.0 TB/s |
| CPU DRAM (via PCIe) | any | microseconds | 64 GB/s (PCIe 5.0 x16) |

Each step down the tier list is roughly an order of magnitude more capacity and roughly an order of magnitude less bandwidth. All optimization work is about moving the *minimum* bytes up the hierarchy — ideally bringing weights to the SMs once and reusing them many times.

## The compute throughput

Tensor-core throughput is quoted per precision. On an H100 SXM:

- FP64: 67 TFLOPs
- FP32 / TF32: 494 TFLOPs
- BF16 / FP16: 1,979 TFLOPs
- FP8: 3,958 TFLOPs

Each halving of precision roughly doubles throughput on the same silicon. B200 adds FP4 at ~9,000 TFLOPs — another doubling.

## The ratio that decides your kernel's fate

Define **arithmetic intensity** as flops per byte moved from HBM:

$$ \text{AI} = \frac{\text{FLOPs}}{\text{HBM bytes moved}} $$

And the hardware's **ridge point** as peak compute divided by peak bandwidth:

$$ \text{Ridge} = \frac{\text{peak FLOPs}}{\text{peak bandwidth}} $$

For H100 at bf16: &#36;1{,}979 \times 10^{12} / 3.35 \times 10^{12} \approx 590$ flops/byte. At fp8: ~1,180 flops/byte.

If AI > ridge, the kernel is **compute-bound** — the GPU runs flat-out on tensor cores, HBM has slack. If AI < ridge, the kernel is **memory-bound** — tensor cores sit idle waiting for bytes.

## LLM kernels on the roofline

- **Prefill matmul** at sequence length $s$: AI ≈ $s$. Above ~600 tokens on H100, compute-bound.
- **Decode matmul** at batch $B$: AI ≈ $B$. At $B = 1$, hopelessly memory-bound. At $B = 256$, approaching the ridge.
- **Attention**: AI depends on sequence length and head dim; FlashAttention is the IO-aware rewrite that raises AI by tiling.
- **Element-wise ops** (residual, layer norm, activation): AI ≈ 1. Always memory-bound. Fusion (M7.5) pays its rent here.

## Why this matters more than any other number

Every serving decision — whether to quantize, whether to batch bigger, whether to switch to H200, whether to deploy speculative decoding — maps to "where does this put me on the roofline?" An engineer who has internalized the ridge point can predict optimization outcomes without running them.

- Quantization (bf16 → int4) cuts bytes/weight by 4×, raises decode AI by 4×. Helps if memory-bound.
- Bigger batch raises decode AI linearly. Helps if memory-bound; stops helping at the ridge.
- FP8 raises peak compute 2×, moves the ridge point further right. Helps only if compute-bound — mostly prefill.
- FlashAttention raises attention AI by tiling. Helps regardless, but especially for long sequences.

## Pitfalls

- **Chasing peak compute when memory-bound.** FP8 kernels will not speed up a decode step at $B = 1$ — the GPU was already idle on memory.
- **Assuming L2 is free.** L2 is shared chip-wide and contested. Big kernels can thrash it; keeping block working sets below ~1 MB helps.
- **Ignoring HBM capacity for bandwidth.** H200 wins over H100 on decode more because of bandwidth (4.8 vs 3.35 TB/s) than capacity (141 vs 80 GB).

<section class="faq">
<h2>FAQ</h2>
<h3>Is more memory bandwidth always better for inference?</h3>
<p>For decode, almost always yes — decode is memory-bound. For prefill, only until compute saturates.</p>
<h3>How do I tell if I'm memory-bound or compute-bound?</h3>
<p>Look at SM % vs. HBM % during the phase of interest. SM high + HBM low → compute-bound. HBM high + SM middling → memory-bound. Use <code>dcgmi dmon</code> or Nsight.</p>
<h3>Does L2 cache help LLM inference?</h3>
<p>Marginally for weights reuse within a layer; not for KV cache reads which exceed L2 by orders of magnitude. H200 and B200's larger L2 helps at the margin, not dramatically.</p>
<h3>Why does H200 beat H100 at decode despite identical FLOPs?</h3>
<p>Decode is memory-bound. H200 ships 43% more HBM bandwidth. The roofline ridge shifts left; the same memory-bound kernel runs proportionally faster.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://horace.io/brrr_intro.html" target="_blank" rel="noreferrer">Horace He — Making Deep Learning Go Brrrr From First Principles</a></li>
<li><a href="https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf" target="_blank" rel="noreferrer">Williams, Waterman, Patterson — The Roofline Model (CACM 2009)</a></li>
<li><a href="https://docs.nvidia.com/cuda/hopper-tuning-guide/" target="_blank" rel="noreferrer">NVIDIA Hopper Tuning Guide</a></li>
</ul>
</section>

## What's next

[Chapter 4](?module=m2&chapter=gpu-memory-for-inference) decomposes what actually fills those bytes during inference, and [Chapter 7](?module=m2&chapter=performance-bottlenecks) turns the roofline into a workflow for diagnosing slow serving.
