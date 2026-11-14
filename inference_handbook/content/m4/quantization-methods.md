<div class="chapter-meta">Module 4 · Chapter 3</div>

# Quantization Methods and Frameworks

<p class="lede">Four methods dominate production LLM quantization in 2026: <strong>GPTQ</strong>, <strong>AWQ</strong>, <strong>SmoothQuant</strong>, and the newer <strong>LLM Compressor / compressed-tensors</strong> stack. Each uses a different strategy to minimize the quality impact of lower precision. This chapter maps the method → quality → engine support triangle so you can pick one without reading five papers.</p>

## The five production methods

| Method | What it does | Precision | Supported by |
|---|---|---|---|
| **GPTQ** | Layer-by-layer error-minimizing weight quantization | W4 (sometimes W3, W2) | vLLM, SGLang, TGI, llama.cpp (via GGUF conversion) |
| **AWQ** | Activation-aware channel reweighting → W4 | W4, W8 | vLLM, SGLang, TensorRT-LLM, TGI |
| **SmoothQuant** | Pushes outliers from activations into weights so W8A8 works | W8A8 | vLLM, TensorRT-LLM, TGI |
| **LLM Compressor** | Unified recipes (GPTQ, SmoothQuant, FP8) in one toolkit | FP8, W4A16, W8A8, KV | vLLM, SGLang (via compressed-tensors format) |
| **GGUF (llama.cpp)** | Format + many quant schemes (Q4_K_M, Q5_K_M, Q8_0…) | Q2–Q8 variants | llama.cpp, Ollama, LM Studio |

## GPTQ — error-minimizing weight quantization

Processes one layer at a time. For each weight, picks the quantized value that minimizes the *reconstruction error on calibration activations* — not just the weight itself. Compared to naive round-to-nearest (RTN), produces 1–3 pp better accuracy on MMLU at int4. Needs ~128 calibration samples.

## AWQ — activation-aware weight quantization

Observes that not all weights are equally important — a small fraction drives most of the model's output. Identifies "salient" channels via activation magnitudes and scales them up before quantizing so they land in the high-resolution portion of the integer range. Faster to compute than GPTQ (no Hessian inversion), often matches or beats it, especially at int4.

## SmoothQuant — making activations quantization-friendly

Activation distributions in LLMs have long tails (Transformer "outlier" channels). SmoothQuant mathematically migrates outlier magnitude from activations into weights by rescaling ($Y = X \cdot W = (X / s) \cdot (W \cdot s)$). After migration, both activations and weights are smooth enough to quantize to INT8 or FP8 without quality loss. Prerequisite for most W8A8 and FP8 recipes.

## LLM Compressor — the 2026 production stack

Neural Magic's open-source toolkit (now part of vLLM project). Unifies GPTQ, SmoothQuant, FP8, and KV quantization recipes into a declarative YAML/Python API. Produces **compressed-tensors**-format checkpoints that vLLM and SGLang serve natively. The default modern path for creating production FP8 and W4A16 checkpoints.

## GGUF — the llama.cpp ecosystem

Not a single method but a family of schemes (Q2_K, Q3_K, Q4_K_M, Q5_K_M, Q6_K, Q8_0…) that llama.cpp supports. Dominant for CPU and Apple Silicon serving. Not interchangeable with safetensors; conversion is a one-way GGUF process.

## A practical decision tree

1. Serving on vLLM/SGLang/TRT-LLM + Hopper or newer GPU → **FP8 via LLM Compressor**. Best quality/throughput in 2026.
2. Need int4 for memory reasons → **AWQ**. Widely supported, excellent quality.
3. Need int8 end-to-end for a legacy or Ampere stack → **SmoothQuant W8A8**.
4. CPU or llama.cpp serving → **GGUF Q4_K_M or Q5_K_M**.
5. Pre-quantized checkpoint on Hugging Face → use it; match the engine's supported format.

## Minimal LLM Compressor recipe

```python
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
]

oneshot(
    model="Qwen/Qwen2.5-7B-Instruct",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="./Qwen2.5-7B-Instruct-W4A16",
    num_calibration_samples=128,
)
```

The output directory contains a compressed-tensors checkpoint vLLM can serve directly with `--quantization compressed-tensors`.

## Typical quality / size numbers (Llama 3.1 8B)

| Precision | Size (GB) | MMLU (%) | Δ vs bf16 |
|---|---|---|---|
| bf16 reference | 16.1 | 68.5 | — |
| FP8 (LLM Compressor) | 8.1 | 68.3 | −0.2 |
| AWQ int4 | 5.4 | 67.6 | −0.9 |
| GPTQ int4 | 5.4 | 67.1 | −1.4 |
| GGUF Q4_K_M | 5.8 | 67.4 | −1.1 |
| GGUF Q2_K | 3.3 | 63.1 | −5.4 |

Below int4, quality drops get real. Int4 is the current sweet spot for memory-constrained serving.

## Pitfalls

- **Picking quant method before engine.** Your engine's supported formats narrow the choice. Check first.
- **Trusting the quant library without evaluation.** MMLU is a smoke test; your task might regress 5 % while MMLU shows < 1 %.
- **Over-aggressive calibration.** 128 samples of your target distribution beats 8k samples of mismatched data.
- **Quantizing the LM head or the embedding.** Both are sensitive. Most recipes skip them; make sure yours does.

<section class="faq">
<h2>FAQ</h2>
<h3>Should I always pick FP8 on Hopper?</h3>
<p>If quality permits (it usually does), yes. FP8 has native tensor-core support, biggest throughput gain, and smallest quality hit.</p>
<h3>AWQ vs. GPTQ — which is better?</h3>
<p>In 2026 they're comparable; AWQ is simpler to run and slightly ahead on average quality. Both are int4 weight-only.</p>
<h3>What is "compressed-tensors"?</h3>
<p>A unified checkpoint format from Neural Magic / vLLM project that encodes quantization metadata with the weights. Used by LLM Compressor outputs.</p>
<h3>Can I convert safetensors to GGUF?</h3>
<p>Yes — <code>llama.cpp</code> ships <code>convert_hf_to_gguf.py</code>. One-way: GGUF back to safetensors is not standard.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://arxiv.org/abs/2210.17323" target="_blank" rel="noreferrer">Frantar et al. — GPTQ</a></li>
<li><a href="https://arxiv.org/abs/2306.00978" target="_blank" rel="noreferrer">Lin et al. — AWQ: Activation-aware Weight Quantization</a></li>
<li><a href="https://arxiv.org/abs/2211.10438" target="_blank" rel="noreferrer">Xiao et al. — SmoothQuant</a></li>
<li><a href="https://github.com/vllm-project/llm-compressor" target="_blank" rel="noreferrer">LLM Compressor — GitHub</a></li>
<li><a href="https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md" target="_blank" rel="noreferrer">llama.cpp — GGUF quantization README</a></li>
</ul>
</section>

## What's next

Model distillation (Chapter 4) and fine-tuning (Chapter 5) are the other two model-level levers. Then Chapter 6 teaches how to evaluate the compressed result against your base model.
