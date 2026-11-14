<div class="chapter-meta">Module 7 · Chapter 1</div>

# Introduction to GPU Kernel Optimization

<p class="lede">A <strong>kernel</strong> is a function that runs on the GPU — the primitive operation underneath every PyTorch op. Modern LLM serving engines rely on a small set of highly-tuned kernels (attention, matmul, norms). You rarely write new ones, but understanding what they do and when they are the bottleneck is essential infrastructure literacy.</p>

## What a kernel is

When you call `torch.matmul(A, B)`, Python dispatches to a C++ backend which launches a CUDA kernel — a function that runs on thousands of GPU threads simultaneously. The kernel reads A and B from HBM, computes partial products, and writes the result back.

A "kernel" is not one giant function. A single forward pass through a 70 B model may launch 2,000–20,000 kernels — one or more per operation per layer.

## Three kinds of kernels in LLM inference

- **Matmul (GEMM)** — the heavy lifter. Dense matrix multiplication. Lives in Q/K/V projection, attention output, FFN. Comes from cuBLAS, CUTLASS, or hand-written kernels (FlashAttention).
- **Attention** — a specialized matmul-sandwich around softmax. Historically computed as discrete matmuls; FlashAttention (M7.4) fuses it into one IO-aware kernel.
- **Elementwise + norm** — RMSNorm, LayerNorm, SiLU, GELU, residual adds. Memory-bound; fusion (M7.5) is the game.

## When to care about kernels

As an infra engineer, you care about kernels when:

1. **You're diagnosing a slow forward pass.** Nsight (M7.6) shows which kernels take the most time.
2. **You're evaluating a new framework.** TensorRT-LLM vs. vLLM can differ by 20 % on identical hardware mostly because of kernel choices.
3. **You're on non-NVIDIA silicon.** Kernel coverage is the main gap vs. CUDA. ROCm has fewer well-tuned kernels than cuBLAS.

You rarely care about kernels when:

1. **You're using a modern engine on a Hopper GPU.** The vendors have done the work.
2. **Your bottleneck is the scheduler, memory, or network.** Kernels are not the problem.

## What the main kernels look like

For Llama 3 70B forward pass at batch 16, seq 2048 (prefill), representative kernel breakdown:

| Kernel | ~% of GPU time |
|---|---|
| FlashAttention-3 forward | 35 % |
| Matmul (FFN down_proj + up_proj) | 30 % |
| Matmul (QKV + output_proj) | 20 % |
| AllReduce (if TP > 1) | 10 % |
| Norm + elementwise + residual (fused) | 3 % |
| Sampler | 2 % |

Attention and matmul together are &gt; 85 %. Everything else is noise. Optimization effort follows that distribution.

## Who writes kernels

- **NVIDIA** — cuBLAS, cuDNN, CUTLASS, TensorRT-LLM core kernels. Closed or semi-closed source but foundational.
- **Dao AI Lab (Tri Dao)** — FlashAttention series. Open source. Now the baseline attention kernel for the industry.
- **OpenAI** — Triton. A Python-like DSL for writing kernels. Several vLLM/SGLang custom kernels use it.
- **Meta / Google / open-source community** — specialized kernels (RMSNorm fusion, fused sampling) for specific models.

Most infra engineers never write a kernel. A few at specialized shops (vLLM core team, Modular, NVIDIA) write them full-time.

## The handoff between Python and the kernel

When you call PyTorch, the following happens per op:

```
Python call                            → Dispatcher (~1 μs)
Backend decides which kernel           → Dispatcher
Backend launches the kernel on CUDA     → Launch overhead (~5 μs)
Kernel runs on GPU                      → Work time (depends on op)
Result is available, next call starts   → Dispatcher
```

The 5 μs launch overhead × 20,000 kernels = 100 ms of pure overhead per forward pass if kernels are tiny. CUDA Graphs (M7.5) collapse this.

## Pitfalls

- **Writing custom kernels prematurely.** 99 % of infra teams don't need to. Use the ones vendors ship.
- **Confusing "kernel time" with "GPU time".** Kernel time is what Nsight attributes to each call. GPU total time includes launch overhead.
- **Assuming the best kernel is the one with the most flops/sec.** For memory-bound ops, bytes moved matters more.

<section class="faq">
<h2>FAQ</h2>
<h3>Do I need to know CUDA to serve LLMs?</h3>
<p>No. You need to know <em>what</em> kernels do, not <em>how</em> to write them. This module covers that.</p>
<h3>What's the difference between CUDA and Triton?</h3>
<p>CUDA is C++ with GPU extensions, low-level, max performance. Triton is a Python-ish DSL that compiles to PTX, faster to write, slightly less flexible. See Chapter 3.</p>
<h3>Can I replace a kernel in vLLM?</h3>
<p>Yes but rarely needed. vLLM's custom kernels are tuned by the project maintainers. Patching requires rebuilding.</p>
<h3>Does PyTorch's eager mode launch one kernel per op?</h3>
<p>Mostly yes — modulo autograd helpers. This is why <code>torch.compile</code> and CUDA Graphs matter: they collapse many small kernel launches into fewer, larger ones.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://developer.nvidia.com/cuda-zone" target="_blank" rel="noreferrer">NVIDIA CUDA developer zone</a></li>
<li><a href="https://github.com/NVIDIA/cutlass" target="_blank" rel="noreferrer">NVIDIA CUTLASS — GEMM building blocks</a></li>
<li><a href="https://github.com/openai/triton" target="_blank" rel="noreferrer">OpenAI Triton — GitHub</a></li>
</ul>
</section>

## What's next

[Chapter 2](?module=m7&chapter=compute-kernels) surveys the compute kernels an LLM actually runs. [Chapter 4](?module=m7&chapter=flashattention) is the deep dive on FlashAttention.
