<div class="chapter-meta">Module 4 · Chapter 2</div>

# LLM Quantization Fundamentals

<p class="lede">Quantization is the practice of representing weights (and often activations and KV cache) in fewer bits than the model was trained in. For LLM inference it is the single most impactful optimization: it halves or quarters HBM bandwidth per decode step at a small quality cost. This chapter covers the numerical foundations; the next chapter catalogs the methods.</p>

## What quantization actually does

Floating-point weights (fp32, fp16, bf16) carry ~3–4 significant decimal digits of information. For most LLM matmuls, that is more precision than the result actually needs. Quantization maps the floating-point distribution of a weight matrix to a small integer range, keeping only a coarse representation of magnitude. At decode time the engine either (a) dequantizes on the fly before the matmul or (b) runs a dedicated low-precision matmul kernel.

## The symmetric quant formula

For a weight tensor $W$ with absolute max $\alpha$, quantize to $b$-bit signed integers:

$$ s = \frac{\alpha}{2^{b-1} - 1}, \qquad q = \text{round}\!\left(\frac{W}{s}\right) $$

At runtime, dequantize via $\hat W = s \cdot q$. One scale $s$ per weight group (per row, per channel, per block of 128 values). Group granularity trades metadata overhead against quantization error.

## Asymmetric (zero-point) quantization

When the weight distribution is not centered on zero (common for activations), add a zero point $z$:

$$ s = \frac{\max - \min}{2^b - 1}, \quad z = \text{round}\!\left(\frac{-\min}{s}\right), \quad q = \text{round}\!\left(\frac{W}{s}\right) + z $$

Weights are almost always symmetric (centered). Activations are commonly asymmetric.

## What gets quantized — four regimes

| Regime | Quantizes | Benefit | Cost |
|---|---|---|---|
| Weight-only (W8 / W4) | Weights | Big HBM bandwidth win | Needs dequant per matmul (small) |
| Weight + activation (W8A8 / W4A8) | Weights + activations | Can use int / fp8 matmul kernels directly | Harder to preserve quality; needs calibration |
| KV cache quant (KV4 / KV8) | K and V tensors in cache | Shrinks the KV budget 2–4 × | Some long-context quality cost |
| Full fp8 end-to-end | Weights, activations, often KV | Native tensor-core fp8 path on H100/B200 | Needs a quant-aware recipe (SmoothQuant, LLM Compressor) |

## The bit widths that matter in 2026

- **INT8** — classic; well supported on Ampere+. Quality loss: typically < 1 %.
- **FP8** — first-class on Hopper and Blackwell. Two sub-formats: E4M3 (wider range) and E5M2 (wider precision). Quality loss: < 0.5 %.
- **INT4** — the de facto open-source "small" quantization. AWQ and GPTQ both target int4 weight-only. Quality loss: 0.5–1.5 % with good methods.
- **FP4** — Blackwell-native. 2 × throughput of FP8 on B200. Methods still maturing.

## Kernel support dominates bit width

A key infra-engineer insight: **a lower-bit-width quantization is only faster if a kernel actually supports it on your hardware**. FP8 is faster than bf16 on H100 because the H100 tensor cores have native FP8 pathways. INT4 can be slower than FP8 on H100 if the weights have to dequant to fp16 before matmul. Always benchmark on your target silicon.

## A minimal RTN quantization in PyTorch

```python
import torch

def quantize_int8_symmetric(w: torch.Tensor):
    """Round-to-nearest int8 symmetric per-row."""
    alpha = w.abs().amax(dim=-1, keepdim=True)   # per-row scale
    scale = alpha / 127
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale

def dequantize(q: torch.Tensor, scale: torch.Tensor):
    return q.to(torch.float16) * scale

W = torch.randn(4096, 4096, dtype=torch.float16)
q, s = quantize_int8_symmetric(W)
W_hat = dequantize(q, s)
print("max abs error:", (W - W_hat).abs().max().item())
```

Real production methods (GPTQ, AWQ, SmoothQuant) are smarter than RTN — they take activations into account and pick scales to minimize the matmul output error, not just the weight error. See the next chapter.

## Pitfalls

- **Confusing storage precision with compute precision.** Weight-only int4 still computes in fp16; the win is HBM traffic, not FLOPs.
- **Assuming fewer bits is always better.** If your kernel dequantizes before matmul, the "faster" int4 can be slower than well-optimized fp8.
- **Quantizing the LM head.** The last projection to vocabulary logits is sensitive; keep it in fp16/bf16 unless you have specific evidence it's safe.
- **Mixing a quant method with an unsupported engine.** AWQ checkpoints work in vLLM; GGUF only works in llama.cpp. Match the format to the engine.

<section class="faq">
<h2>FAQ</h2>
<h3>Is FP8 the same as INT8?</h3>
<p>No. FP8 keeps a small exponent (range), INT8 is a fixed-range integer. For LLM weights/activations with long-tailed distributions, FP8 usually preserves more information.</p>
<h3>Why group-wise quantization?</h3>
<p>One scale per matrix is too coarse — outliers in one row poison everything else. Group sizes of 64 or 128 balance error and metadata cost.</p>
<h3>Does KV quant hurt quality?</h3>
<p>At INT8: barely measurable. At INT4: visible on long-context retrieval. Test on your workload.</p>
<h3>Can I retrain into low precision?</h3>
<p>Yes — quantization-aware training (QAT). Rarely used in 2026 because post-training methods (AWQ, SmoothQuant) are fast and good enough.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://arxiv.org/abs/2103.13630" target="_blank" rel="noreferrer">Gholami et al. — A Survey of Quantization Methods for Efficient Neural Network Inference</a></li>
<li><a href="https://arxiv.org/abs/2209.05433" target="_blank" rel="noreferrer">NVIDIA — FP8 Formats for Deep Learning</a></li>
<li><a href="https://blog.vllm.ai/2024/07/23/fp8-quant.html" target="_blank" rel="noreferrer">vLLM Blog — FP8 quantization overview</a></li>
</ul>
</section>

## What's next

[Chapter 3](?module=m4&chapter=quantization-methods) surveys the production methods (GPTQ, AWQ, SmoothQuant, LLM Compressor) and the formats (safetensors, GGUF, compressed-tensors) you will actually encounter.
