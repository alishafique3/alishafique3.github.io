<div class="chapter-meta">Module 7 · Chapter 4</div>

# FlashAttention and Efficient Attention Kernels

<p class="lede">Standard attention materializes the full <code>Q × K^T</code> matrix in HBM — <strong>O(s²)</strong> memory. At 32k context, that's 4 GB per attention head per sample. <strong>FlashAttention</strong> rewrites the kernel to never materialize that matrix: it processes attention block-by-block in shared memory, with memory from O(s²) down to O(s). Every serious LLM engine in 2026 runs FlashAttention.</p>

## The problem FlashAttention solves

Standard (naive) attention computes:

1. $S = Q K^\top$ — produces an $s \times s$ matrix.
2. $P = \text{softmax}(S / \sqrt{d_k})$ — same shape.
3. $O = P V$ — produces the output.

The problem: step 1 and 2 require writing and reading $s^2$ values to HBM. For $s = 32{,}768$ at bf16, that's ~2 GB per head per sample. For Llama 3 with 64 heads, this dominates memory — and HBM traffic dominates wall time.

## The FlashAttention insight

Compute attention in **tiles**: load blocks of Q, K, V into shared memory, compute the block's contribution to the output, update a running softmax normalizer. Never materialize the full $s \times s$ matrix.

Pseudocode:

```
for each query block Q_i:
    initialize m_i = -inf, l_i = 0, O_i = 0     (running max, sum, output)
    for each key/value block K_j, V_j:
        S_ij = Q_i @ K_j.T / sqrt(d_k)
        m_new = max(m_i, rowmax(S_ij))
        P_ij  = exp(S_ij - m_new)
        l_i = exp(m_i - m_new) * l_i + rowsum(P_ij)
        O_i = exp(m_i - m_new) * O_i + P_ij @ V_j
        m_i = m_new
    O_i = O_i / l_i                             (final normalize)
    write O_i to HBM
```

The entire attention computation for a query block stays in SRAM (shared memory). HBM reads/writes drop by $O(s)$. Memory is $O(s)$, not $O(s^2)$.

## The three generations

- **FlashAttention-1 (2022)** — Original. Memory-aware tiling. 2–4× faster than naive on A100.
- **FlashAttention-2 (2023)** — Better work partitioning across warps. Another 2× speedup. Now the baseline in PyTorch, vLLM, SGLang.
- **FlashAttention-3 (2024)** — Hopper-tuned. Uses Tensor Memory Accelerator (TMA) and warpgroup matmul. FP8 attention native. ~2× over FA-2 on H100.

If you're on A100, FA-2. If H100 or H200, FA-3. If B200, FA-3 or specialized Blackwell variants are emerging.

## FlashAttention variants

- **FlashAttention (forward + backward)** — training. Most people doing inference only need forward.
- **FlashAttention-prefill** — handles variable-length sequences packed into one big tensor. Used by continuous-batching engines.
- **FlashAttention-decode** — optimized for single new query against a long KV cache. Different access pattern than prefill.
- **FlashInfer** — a family of attention kernels specialized for various LLM serving scenarios. Used by SGLang.
- **PageAttention** (vLLM's custom kernel) — handles paged KV cache; combines FA-2 ideas with block-table gather.

## Memory savings in practice

For Llama 3 70B at 32k context, prefill:

| Attention impl | HBM reads/writes | Wall time |
|---|---|---|
| Naive attention | ~40 GB | 180 ms |
| FlashAttention-2 | ~4 GB | 45 ms |
| FlashAttention-3 (FP8) | ~2 GB | 22 ms |

The memory and time wins compound at longer sequence. At 4k it's a 2× win; at 128k it's an 8–10× win.

## What you need to know as an infra engineer

- **Which kernel does my engine use?** vLLM: PagedAttention kernel (FA-2-based). SGLang: FlashInfer. TensorRT-LLM: NVIDIA's own. All use a FlashAttention-family variant.
- **How do I enable FP8 attention?** On H100/B200 with FA-3, set `--kv-cache-dtype=fp8` in vLLM.
- **Does it work on AMD?** ROCm FlashAttention exists but trails CUDA by 1–2 releases.

You don't write the kernel. You pick the engine that ships the right variant for your hardware.

## When attention is NOT the bottleneck

- **Short sequences (&lt; 1k).** Attention is tiny; FFN dominates.
- **Decode at tiny batch.** The whole model is memory-bound; attention is a small fraction.
- **MoE models.** Expert routing + MoE FFN can dominate attention.

Profile before you blame or praise attention.

## Pitfalls

- **Enabling FA-3 on a non-Hopper GPU.** FA-3 requires Hopper (H100/H200) or Blackwell (B200). On Ampere it silently falls back to FA-2 — that's fine, just know.
- **Mixing attention backends across TP ranks.** All ranks must use the same backend. Debugging a mismatched backend is a nightmare.
- **Treating "attention kernel" as one thing.** Prefill, decode, long-context, and streaming decode all have different optimal kernel choices.

<section class="faq">
<h2>FAQ</h2>
<h3>Do I need to install FlashAttention separately?</h3>
<p>Usually no — modern vLLM and SGLang bundle it. For PyTorch-level use, <code>pip install flash-attn</code> installs the standalone library.</p>
<h3>Does FlashAttention change the output?</h3>
<p>Mathematically identical (up to floating-point ordering). Outputs from FA and naive attention differ by ~1e-5 on typical inputs — invisible at inference.</p>
<h3>What about sliding window attention?</h3>
<p>FA-2/3 support sliding window natively. Used by Gemma, Phi-3 for long-context efficiency.</p>
<h3>FlashInfer vs. FlashAttention — which is better?</h3>
<p>FlashInfer has more variants (page-aware, bulk prefill) and often wins in serving benchmarks. FlashAttention is simpler and more universally available.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://arxiv.org/abs/2205.14135" target="_blank" rel="noreferrer">Dao et al. — FlashAttention (NeurIPS 2022)</a></li>
<li><a href="https://arxiv.org/abs/2307.08691" target="_blank" rel="noreferrer">Dao — FlashAttention-2</a></li>
<li><a href="https://tridao.me/publications/flash3/flash3.pdf" target="_blank" rel="noreferrer">Shah et al. — FlashAttention-3</a></li>
<li><a href="https://github.com/flashinfer-ai/flashinfer" target="_blank" rel="noreferrer">FlashInfer — GitHub</a></li>
</ul>
</section>

## What's next

[Chapter 5](?module=m7&chapter=kernel-fusion) covers fusion and CUDA Graphs — the other side of making kernel launches cheap.
