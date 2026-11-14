<div class="chapter-meta">Module 7 · Chapter 2</div>

# Understanding LLM Compute Kernels

<p class="lede">Five kernel families consume nearly all the GPU time in LLM inference: <strong>GEMM</strong>, <strong>attention</strong>, <strong>normalization</strong>, <strong>activation</strong>, and <strong>sampling</strong>. This chapter is a quick tour of what each does, where it bottlenecks, and which one your diagnostics should look at first.</p>

## 1. GEMM — the general matrix multiply

The workhorse. $Y = X W$. Every linear layer — Q/K/V projection, attention output, FFN up, FFN down, LM head — is a GEMM.

For Llama 3 70B, a single FFN up-projection on a sequence of 2k tokens is a GEMM of shape $(2048, 8192) \times (8192, 28672)$ = ~960 GFLOPs. There are 80 layers × 3 FFN projections + 4 attention projections = 560 GEMMs per forward pass.

Implementations:

- **cuBLAS** — NVIDIA's closed-source standard. Highly optimized, dynamically selects best algorithm.
- **CUTLASS** — NVIDIA's open-source templates. Used to build custom GEMMs with specific tile sizes.
- **Triton matmul** — community implementations; fast enough for most cases.

GEMM tuning: pick the right tile shape, maximize tensor-core utilization, overlap HBM loads with compute.

## 2. Attention — specialized matmul-sandwich

$\text{softmax}(Q K^\top / \sqrt{d_k}) V$. Looks like two GEMMs with a softmax in between. In naive implementation, the intermediate $Q K^\top$ attention matrix is huge ($O(s^2)$ bytes) and dominates memory.

**FlashAttention** (M7.4) reformulates this to never materialize the full attention matrix — it tiles the computation so each tile fits in shared memory. Memory from $O(s^2)$ to $O(s)$; wall time from 2–4× faster at long sequences.

FlashAttention-3 (Hopper-tuned, 2024) adds FP8 attention natively.

## 3. Normalization — RMSNorm / LayerNorm

Normalize across the hidden dimension per token:

$$ \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot g $$

Elementwise with a reduction. Memory-bound. Fast on modern GPUs ($\sim 1$ % of forward pass time per layer) unless unfused — fusing the preceding residual add and the following projection matters (M7.5).

RMSNorm (LLaMA, Qwen) is slightly cheaper than LayerNorm (GPT-2 era) because it skips the mean subtraction.

## 4. Activation — SiLU / GELU / SwiGLU

Elementwise non-linearity in the FFN:

- **GELU** — $x \cdot \Phi(x)$. GPT-2-era.
- **SiLU** (aka Swish) — $x \cdot \sigma(x)$. Simpler, equally expressive.
- **SwiGLU** — gated variant: $\sigma(W_1 x) \odot (W_2 x)$. Used in Llama, Qwen, Gemma. Requires three FFN projections instead of two.

Pure elementwise ops are memory-bound and fast. Fused into the surrounding matmul when possible.

## 5. Sampling — softmax + draw

Final step per token. The logit vector (vocab size, e.g., 128k) gets softmax'd, filtered by top-p / top-k / min-p, and a single token is sampled.

Looks tiny but adds up: 128k softmax + CDF for 16 sequences in a batch is ~2 MB of data per decode step. Fused samplers (in TensorRT-LLM, vLLM) keep this under ~5 % of decode time.

## 6. AllReduce and collectives (TP serving)

Not a compute kernel but a communication kernel. Covered in M2.5. In profiling it shows up as NCCL kernels running in parallel with — or sometimes blocking — the compute kernels.

## A quick per-op decision guide

| Op type | Bottleneck class | What fixes it |
|---|---|---|
| Prefill QKV GEMM | Compute | Lower precision (FP8), better GEMM kernel |
| Prefill attention | Memory + compute mix | FlashAttention, tiled softmax |
| Decode attention (bs=1) | Memory | Bigger batch, KV quantization |
| FFN up/down | Compute (prefill) / Memory (decode) | Precision, bigger batch |
| RMSNorm | Memory + overhead | Fuse with surrounding ops |
| Elementwise | Overhead | Fuse (torch.compile, CUDA Graphs) |
| Sampler | Memory + overhead | Fused sampler kernel |

## Pitfalls

- **Optimizing the small kernels.** Tweaking RMSNorm when attention is 40 % of your time is theater.
- **Assuming GEMM is "always fast".** Shape and precision matter enormously. A `(1, 8192) × (8192, 4096)` decode GEMM is dramatically different than a `(2048, 8192) × (8192, 4096)` prefill GEMM.
- **Treating attention as a single kernel.** FlashAttention is one kernel; naive attention is 3–4 kernels. Profilers can show either.

<section class="faq">
<h2>FAQ</h2>
<h3>Which kernel should I profile first?</h3>
<p>The one taking the most wall time. Nsight's top-time view is the right first click.</p>
<h3>Is cuBLAS always faster than CUTLASS?</h3>
<p>Not always. CUTLASS lets you tune for a specific shape; cuBLAS picks an algorithm but may not be optimal. For production kernels in serving engines, hand-tuned CUTLASS often wins by 10–30 %.</p>
<h3>Does a bigger batch always make GEMMs faster per token?</h3>
<p>Yes up to the ridge (M2.3). Beyond that, you saturate tensor cores and gains flatten.</p>
<h3>What is "attention v2" or "Flash-Attention-2"?</h3>
<p>The second-generation FlashAttention paper (2023). FA-3 (2024) adds Hopper + FP8. See M7.4.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://developer.nvidia.com/cublas" target="_blank" rel="noreferrer">NVIDIA cuBLAS</a></li>
<li><a href="https://github.com/Dao-AILab/flash-attention" target="_blank" rel="noreferrer">Dao-AILab — FlashAttention repository</a></li>
<li><a href="https://arxiv.org/abs/2108.12409" target="_blank" rel="noreferrer">Press et al. — GLU Variants Improve Transformer (SwiGLU)</a></li>
</ul>
</section>

## What's next

[Chapter 3](?module=m7&chapter=cuda-triton) covers when to write CUDA vs. Triton vs. stay in PyTorch, and [Chapter 4](?module=m7&chapter=flashattention) deep-dives FlashAttention.
