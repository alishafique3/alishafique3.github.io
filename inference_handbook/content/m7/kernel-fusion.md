<div class="chapter-meta">Module 7 · Chapter 5</div>

# Kernel Fusion and Low-Level Optimization

<p class="lede">A forward pass through a 70 B model launches 10,000+ small kernels — norms, residuals, activations. Each launch costs ~5 μs of overhead. 10k launches × 5 μs = 50 ms per forward pass of pure overhead. <strong>Kernel fusion</strong> and <strong>CUDA Graphs</strong> are the two techniques that turn that overhead into almost nothing.</p>

## Why fusion matters

Think of a simple residual + norm + linear sequence:

```
y = linear(norm(x + residual))
```

Naive execution launches 4 kernels: add, norm, linear, write. Each reads and writes HBM. **Fused execution** does the whole thing in one kernel: read x and residual once, do the math in registers, write y once.

Benefits:

- **Fewer launches** — less overhead.
- **Less HBM traffic** — intermediate results stay in registers or shared memory.
- **Better cache locality** — subsequent kernels start warm.

Typical win: 10–30 % on memory-bound regions (norms, activations, samplers), negligible on compute-bound (big matmul).

## Three mechanisms

### 1. Hand-written fused kernels

Specific fused ops shipped by engine authors. Examples:

- **RMSNorm + residual + matmul** — appears in every block.
- **Sampling (softmax + top-p + sample)** — one kernel instead of five.
- **Rotary embedding + attention** — RoPE fused into the attention launch.

vLLM, SGLang, TensorRT-LLM all ship these. You enable them by picking the engine.

### 2. torch.compile — automatic fusion

```python
model = torch.compile(model, mode="reduce-overhead")
```

Traces the model forward and compiles contiguous sequences of ops into fused kernels via Triton. In LLM inference, the biggest win is fusing the many small ops between layers.

Modes:

- `"default"` — balanced. Works for most training.
- `"reduce-overhead"` — uses CUDA Graphs. Best for inference.
- `"max-autotune"` — exhaustive search for best kernel. Slow compile.

### 3. CUDA Graphs — amortize launches

Record a sequence of kernel launches once, replay the whole batch. Launch overhead paid once instead of per-call. Works best when the shape and ordering are fixed (static batch size).

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = model(static_input)

# Replay many times — no per-launch overhead
for inputs in batches:
    static_input.copy_(inputs)
    g.replay()
    torch.cuda.synchronize()
    process(out)
```

Modern engines capture CUDA Graphs for each batch size they support, then replay the right one per iteration. Cuts decode TPOT by 20–40 % at low batch where launch overhead dominates.

## Where fusion wins the most

- **Decode at small batch.** Launch overhead is the largest fraction of time.
- **Sampling.** The 5-kernel sequence (softmax → top-p filter → top-k filter → multinomial sample → scatter) fuses to 1.
- **Position encoding.** RoPE applied per token benefits from being fused into QKV projection.

## Where fusion doesn't help

- **Large GEMM-dominated regions.** The matmul itself is 100 ms; shaving the 1 μs launch is noise.
- **Very long prefill.** Compute dominates launch overhead.
- **Already-fused code.** torch.compile won't re-fuse FlashAttention.

## A realistic example — decode TPOT breakdown

Llama 3 70B on 2 × H100, batch 1, bf16, decode:

| Technique | TPOT | Comment |
|---|---|---|
| Eager PyTorch, no fusion | 55 ms | Baseline |
| + torch.compile | 42 ms | Auto-fusion, no graph |
| + CUDA Graphs | 24 ms | Launch overhead amortized |
| + FP8 weights | 14 ms | Less HBM traffic |
| + Continuous batching @ bs=16 | 3 ms | Amortized across batch |

Fusion + graphs alone ~2× the TPOT. Combined with the other stacks, 15–20× the naive baseline.

## CUDA Graphs and dynamic shapes

CUDA Graphs record a specific shape. Dynamic batch sizes break them. Solutions:

- **Pad to static shapes.** vLLM captures graphs for a set of batch sizes (1, 2, 4, 8, 16, ...) and picks the smallest that fits.
- **Multiple graphs.** Capture one per shape. Memory cost is modest.
- **Fall back for odd shapes.** Hit the un-graphed path for rare shapes.

This is why vLLM shows "CUDA graph capture" at startup — it's building the shape-indexed graph set.

## Pitfalls

- **Enabling torch.compile without warmup.** First inference after compile is slow. Warm up before serving.
- **Using `max-autotune` mode in production.** Compile times are 10–30 minutes. Pin kernel choices after tuning.
- **Mixing CUDA Graphs with dynamic shapes silently.** Some ops break graph capture. Nsight will show the un-graphed launches.
- **Treating fusion as a replacement for right-sizing.** Fusion can't turn a memory-bound kernel into a compute-bound one.

<section class="faq">
<h2>FAQ</h2>
<h3>Does torch.compile help prefill?</h3>
<p>Modestly — maybe 5–10 %. Prefill is already compute-bound. Where it shines is decode.</p>
<h3>Can I use CUDA Graphs with TP?</h3>
<p>Yes, but the collectives must be included in the graph. NCCL supports this; modern engines handle it.</p>
<h3>What about Triton for fusion?</h3>
<p>Both torch.compile and vLLM/SGLang already generate Triton code for fused kernels under the hood.</p>
<h3>Is fusion compatible with FP8?</h3>
<p>Yes — fusion is orthogonal to precision. Combine for compounding wins.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://pytorch.org/docs/stable/torch.compiler.html" target="_blank" rel="noreferrer">PyTorch — torch.compile documentation</a></li>
<li><a href="https://developer.nvidia.com/blog/cuda-graphs/" target="_blank" rel="noreferrer">NVIDIA — CUDA Graphs explained</a></li>
<li><a href="https://pytorch.org/blog/cuda-graphs-pytorch/" target="_blank" rel="noreferrer">PyTorch Blog — Accelerating PyTorch with CUDA Graphs</a></li>
</ul>
</section>

## What's next

[Chapter 6 — Profiling](?module=m7&chapter=profiling) closes the module with the tools (Nsight, DCGM) that reveal which fusion or kernel change is actually moving the needle.
