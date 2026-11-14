<div class="chapter-meta">Module 2 · Chapter 2</div>

# GPU Architecture Fundamentals

<p class="lede">You do not need to write CUDA to serve LLMs well, but you do need a mental model of what a GPU is physically doing when you call <code>model.generate()</code>. This chapter is the one-page mental model: streaming multiprocessors, tensor cores, warps, and how a Python one-liner lands on silicon.</p>

## What a GPU physically is

A modern data-center GPU is a die carrying hundreds of nearly-identical compute units called **Streaming Multiprocessors** (SMs), surrounded by stacks of **HBM** (High-Bandwidth Memory) connected by a very wide bus. An H100 has 132 SMs and 80 GB HBM3. A B200 has 160 SMs and 180 GB HBM3e.

Each SM has:

- **CUDA cores** — fp32/fp64 arithmetic units, for general-purpose work.
- **Tensor Cores** — specialized matmul units that do $4 \times 4$ or $8 \times 8$ matrix multiply-accumulates in one instruction. Where LLM throughput actually lives.
- **Register file** — a few hundred KB of ultra-fast storage.
- **Shared memory / L1 cache** — ~256 KB per SM, programmer-controlled, nanosecond latency.
- **Warp schedulers** — dispatch groups of 32 threads at a time.

## Threads, warps, blocks, grids

GPU programming is hierarchical:

1. **Thread** — the atomic unit. Each thread executes one instruction stream.
2. **Warp** — 32 threads that execute the same instruction at the same time (SIMT). The warp is the real unit of execution; a single thread cannot diverge from its warp without a performance penalty.
3. **Block (thread block)** — a group of warps assigned to a single SM. Threads in a block can share shared-memory and synchronize with each other.
4. **Grid** — the set of all blocks launched for a single kernel invocation.

## Tensor Cores — the engine of LLM inference

A tensor core executes a fused matmul-accumulate: $D = A \times B + C$ on small matrices in one cycle. On H100 at bf16, each SM delivers ~500 TFLOPs; the whole GPU hits 1,979 TFLOPs sustained. At fp8 the same silicon reaches 3,958 TFLOPs; B200 doubles that again, and introduces fp4 at 9,000 TFLOPs.

Attention and feed-forward kernels are rewritten to be tensor-core friendly — specific tile shapes, specific precisions. This is why "just use fp16" sometimes runs slower than bf16 even on identical hardware: tensor-core pathways favor the native precision for that generation.

## How a Python one-liner lands on silicon

`torch.matmul(A, B)` in Python does roughly this:

1. PyTorch dispatches the op to a backend kernel (cuBLAS, Triton, FlashAttention, or similar).
2. The kernel decides a tiling plan — e.g., split the output into 128 × 128 tiles.
3. PyTorch launches a CUDA grid: one thread block per output tile.
4. Each block loads a slice of $A$ and $B$ from HBM into shared memory.
5. Threads within the block feed tensor cores to compute the output tile, accumulating in registers.
6. The block writes the result back to HBM.

The key infrastructure fact: step 4 (HBM to shared memory) is *slow* compared to step 5 (tensor-core compute). IO-aware kernels like FlashAttention (see M7.4) are entirely about amortizing step 4 across as much step-5 work as possible.

## Generations at a glance (2026)

| Arch | Flagship | SMs | Shader cores / SM | Tensor core precisions |
|---|---|---|---|---|
| Ampere (A100) | A100 | 108 | 64 | fp16, bf16, tf32, int8, int4 |
| Ada (L40S / 4090) | L40S | 142 | 128 | + fp8 (compute only, no sparsity) |
| Hopper (H100/H200) | H100 | 132 | 128 | + fp8 (first-class), Transformer Engine |
| Blackwell (B200) | B200 | 160 | 128 | + fp4, dual-die |

## Pitfalls

- **Treating "CUDA cores" as the figure of merit.** Tensor core throughput dominates LLM compute by 10 ×; CUDA-core count is nearly irrelevant.
- **Assuming all SMs work in parallel on the same problem.** A kernel launch may span one SM or all of them — grid size is up to the kernel.
- **Ignoring warp divergence.** Branchy code inside a warp (if/else on per-thread data) serializes execution. Fine in high-level code; important if you ever write kernels (Module 7).

<section class="faq">
<h2>FAQ</h2>
<h3>How many SMs does my GPU have?</h3>
<p>A100: 108. L40S: 142. H100 SXM: 132. H200: 132. B200: 160. SM count matters less than per-SM tensor-core throughput.</p>
<h3>What's a warp vs. a thread block?</h3>
<p>A warp is 32 threads executing in lockstep on one SM. A thread block is a group of warps that share shared memory and can synchronize. A kernel launches a grid of thread blocks.</p>
<h3>Do Tensor Cores support int4?</h3>
<p>Yes on Hopper and Blackwell. Blackwell also supports FP4 natively — the first-class FP4 pathway is the big Blackwell story for inference.</p>
<h3>Should I care about CUDA core count?</h3>
<p>Almost never for LLM work. Tensor-core throughput dominates by 10×; CUDA cores matter for the auxiliary ops (norms, activations, sampling) which are a small fraction of wall time.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet" target="_blank" rel="noreferrer">NVIDIA Hopper architecture white paper</a></li>
<li><a href="https://resources.nvidia.com/en-us-blackwell-architecture" target="_blank" rel="noreferrer">NVIDIA Blackwell architecture overview</a></li>
<li><a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html" target="_blank" rel="noreferrer">CUDA C++ Programming Guide</a></li>
</ul>
</section>

## What's next

The [next chapter](?module=m2&chapter=compute-memory-hierarchy) looks at the memory hierarchy (HBM → L2 → shared → registers) that feeds the SMs, then [Chapter 4](?module=m2&chapter=gpu-memory-for-inference) decomposes what actually lives in those bytes during inference.
