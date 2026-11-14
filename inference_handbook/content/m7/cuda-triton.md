<div class="chapter-meta">Module 7 · Chapter 3</div>

# CUDA and Triton for LLM Inference

<p class="lede">When "just use PyTorch" doesn't give you the speed you need, you write a custom kernel. Two languages dominate 2026: <strong>CUDA</strong> (NVIDIA's C++ extension, max flexibility) and <strong>Triton</strong> (OpenAI's Python-like DSL, faster to iterate). This chapter covers what each is good for and when to reach for them.</p>

## PyTorch first — always

Before touching CUDA or Triton, make sure PyTorch's built-in op is actually the bottleneck. 90 % of apparent "need a custom kernel" cases are:

- Kernel launch overhead → use `torch.compile` or CUDA Graphs (M7.5).
- Inefficient memory layout → add a contiguous() call or reshape.
- Wrong precision → quantize to FP8 instead of writing a kernel.
- Un-fused elementwise ops → `torch.compile` fuses them for free.

Only after these fail do you actually need a kernel.

## When CUDA wins

Pick CUDA when:

- You need the absolute maximum performance on specific tensor-core shapes.
- You're targeting multiple GPU generations and need fine control over each.
- You're integrating with CUTLASS or cuBLAS APIs.
- The problem has very non-standard memory access (ragged tensors, custom KV formats).

CUDA kernels used in LLM inference are written by vendors (NVIDIA) and specialist teams (vLLM core, Dao AI Lab). Infra engineers rarely write new ones.

### A minimal CUDA kernel structure

```cpp
// elementwise_relu.cu
__global__ void relu_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] > 0 ? input[idx] : 0.0f;
    }
}

// Launch:
int threads = 256;
int blocks  = (n + threads - 1) / threads;
relu_kernel<<<blocks, threads>>>(d_input, d_output, n);
```

PyTorch binds via `torch.utils.cpp_extension.load_inline()` or a proper `setup.py`.

## When Triton wins

Pick Triton when:

- You need a custom kernel but can't justify CUDA's complexity.
- Your kernel is a standard shape (fused matmul + activation, custom attention variants).
- You're iterating quickly on tile-shape / precision trade-offs.
- You want the kernel to run on both NVIDIA and (via Triton's ROCm backend) AMD.

Triton reads like Python but compiles to PTX via LLVM. Performance within 10–15 % of hand-tuned CUDA for most LLM kernels, at a fraction of the development time.

### A minimal Triton kernel

```python
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offs, y, mask=mask)

def relu(x):
    out = torch.empty_like(x)
    n = x.numel()
    relu_kernel[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
    return out
```

Many vLLM and SGLang custom kernels (sampling, specialized attention variants) are Triton today.

## When to use installed kernels

For:

- Standard attention: FlashAttention 2 or 3 (`pip install flash-attn`).
- Standard matmul: cuBLAS via PyTorch.
- Elementwise fusion: `torch.compile`.
- MoE routing: Megablocks or custom vendor kernels.

Writing these yourself is rarely worth the effort.

## A realistic decision flow

```
Does PyTorch do it?
├── Yes → use PyTorch
└── No
    ├── Is it torch.compile-able?
    │   ├── Yes → torch.compile
    │   └── No
    │       ├── Is it a standard op with a known kernel?
    │       │   ├── Yes → use that kernel (FlashAttention, etc.)
    │       │   └── No
    │       │       ├── Are you building a general-purpose feature?
    │       │       │   ├── Yes → Triton
    │       │       │   └── No
    │       │       │       └── Maximum perf, specific hardware → CUDA
```

## Pitfalls

- **Writing custom CUDA before trying torch.compile.** torch.compile 2.4+ does kernel fusion well for most LLM use cases.
- **Using Triton for problems CUDA solves better.** Non-standard memory access patterns, multi-precision tensor cores — CUDA's flexibility still matters.
- **Shipping a custom kernel without vendor-supplied fallback.** New GPU architectures (B200) may not run your kernel efficiently. Always keep a working fallback.

<section class="faq">
<h2>FAQ</h2>
<h3>Can Triton target AMD GPUs?</h3>
<p>Yes — Triton has a ROCm backend that is maturing. Not at CUDA parity in late 2026, but close for many kernels.</p>
<h3>Which is faster, Triton or hand-tuned CUDA?</h3>
<p>CUDA, typically by 10–20 % for well-tuned kernels. Triton is "good enough" for 80 % of real cases and much easier to iterate on.</p>
<h3>Can I call a custom CUDA kernel from vLLM?</h3>
<p>Yes — rebuild with your kernel registered. vLLM accepts custom attention backends; several commercial engines are forks that swap kernels.</p>
<h3>Is CUTLASS a library or a language?</h3>
<p>A C++ template library of building blocks (tile shapes, epilogues, warp specializations) that you compose into a GEMM kernel. More flexible than cuBLAS, less abstract than Triton.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://openai.com/research/triton" target="_blank" rel="noreferrer">OpenAI — Triton release announcement</a></li>
<li><a href="https://triton-lang.org/main/getting-started/tutorials/index.html" target="_blank" rel="noreferrer">Triton documentation and tutorials</a></li>
<li><a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html" target="_blank" rel="noreferrer">CUDA C++ Programming Guide</a></li>
<li><a href="https://github.com/NVIDIA/cutlass" target="_blank" rel="noreferrer">NVIDIA CUTLASS</a></li>
</ul>
</section>

## What's next

[Chapter 4](?module=m7&chapter=flashattention) is the FlashAttention deep dive — the single most important kernel in modern LLM inference.
