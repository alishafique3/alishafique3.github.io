<div class="chapter-meta">Module 4 · Chapter 1</div>

# Introduction to Model Optimization

<p class="lede">There are two broad places to reduce the cost of inference: change the <strong>model</strong> (less of it, lower precision, specialized for the task) or change the <strong>runtime</strong> (better batching, better kernels, better scheduling). This module is about the first lever. Module 6 handles the second. Both matter — but the model-side wins compound through every other optimization downstream.</p>

## Three model-level optimizations

1. **Quantization** — reduce the precision of weights (and sometimes activations and KV cache) from fp16/bf16 to int8, fp8, int4, or fp4. Nearly free in quality, 2–4 × smaller memory footprint, 2–4 × higher effective bandwidth.
2. **Distillation** — train a smaller model to imitate a larger one. 5–10 × smaller and faster at ~90 % of quality on the target distribution.
3. **Fine-tuning** — specialize an open-weight model for your specific task. Often lets you use a smaller base model (because it's doing less work) at better quality than a bigger generic model.

Each technique is independent and composable. Fine-tuned 8 B model + int4 quantization is a routine production choice today.

## Why model-level wins compound

- Smaller/quantized weights → less HBM bandwidth per decode step → higher TPOT ceiling.
- Smaller/quantized weights → larger batches fit → better throughput amortization.
- Smaller/quantized weights → bigger KV budget → longer contexts / more concurrency.
- Smaller/quantized weights → lower energy per token → lower &#36;/token.

Every runtime optimization has diminishing returns once the model itself is well-sized and well-quantized.

## The quality trade-off is real but smaller than people expect

In 2026 the quality cost of production quantization is in the noise for most models and tasks:

- FP8 weight-only on H100: < 0.5 % on most lm-eval tasks.
- AWQ int4 weight-only: 0.5–1.5 %.
- GPTQ int4 weight + activation: 1–3 %.
- Int4 KV cache: often indistinguishable on chat; some long-context regressions.

Distillation and fine-tuning move quality in both directions — you can gain or lose a few percent depending on how aggressive you are and how well-matched the target distribution is to the eval.

## A realistic model-level pipeline

1. Pick the right base model (Module 3 Ch. 2).
2. Fine-tune if you have labeled data for your task.
3. Quantize to FP8 or int4 (Chapters 2–3).
4. Evaluate on your own benchmark (Chapter 6).
5. Deploy and monitor for quality drift.

## Pitfalls

- **Over-optimizing before measuring.** Quantize the wrong model and you optimize the wrong problem.
- **Relying on vendor benchmarks for quality.** They measure on their canonical eval, not on your task.
- **Mixing quantization methods without a plan.** AWQ weights + FP8 activations + INT4 KV cache may each work alone; the combination needs its own eval pass.
- **Treating distillation as free capability transfer.** Distilled models often lose some reasoning or agentic depth. Measure the regressions you care about.

<section class="faq">
<h2>FAQ</h2>
<h3>Should I quantize before or after fine-tuning?</h3>
<p>Fine-tune first on bf16, then quantize. Quantizing then fine-tuning (QLoRA style) is cheaper but gives slightly weaker results.</p>
<h3>Is runtime optimization sometimes a better first step?</h3>
<p>Yes — if your model is already correctly sized and your bottleneck is scheduling or kernels (e.g., no continuous batching yet), fix runtime first.</p>
<h3>Is this module about LLMs only?</h3>
<p>The techniques generalize to any transformer model. The specifics (which quantizer, which distillation recipe) are LLM-focused.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://arxiv.org/abs/2103.13630" target="_blank" rel="noreferrer">Gholami et al. — A Survey of Quantization Methods for Efficient Neural Network Inference</a></li>
<li><a href="https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/" target="_blank" rel="noreferrer">NVIDIA — Mastering LLM Techniques: Inference Optimization</a></li>
<li><a href="https://handbook.modular.com/model-preparation/" target="_blank" rel="noreferrer">Modular — Model Preparation chapter</a></li>
</ul>
</section>

## What's next

[Chapter 2 — Quantization Fundamentals](?module=m4&chapter=quantization-fundamentals) digs into the math of mapping fp16 to int8 and int4, and the follow-up chapters specialize into methods, distillation, fine-tuning, and evaluation.
