<div class="chapter-meta">Module 4 · Chapter 6</div>

# Evaluating Optimized Models

<p class="lede">Every model-level optimization — quantization, distillation, fine-tuning — raises the same question: <strong>did we break something?</strong> The honest answer requires an evaluation that covers quality, latency, throughput, and your own task-specific benchmark. This chapter is the framework you use to sign off on a shipping candidate.</p>

## Four axes to measure

1. **Quality** — accuracy, perplexity, task benchmarks, your own eval.
2. **Latency** — TTFT, TPOT, E2E.
3. **Throughput** — RPS and TPS under your SLO.
4. **Memory** — weights, KV cache, activation footprint.

A good optimization improves some axes without regressing the others past tolerance. A great one improves several.

## Perplexity — the cheap quality smoke test

Perplexity on a held-out corpus:

$$ \text{PPL} = \exp\!\left(-\frac{1}{N} \sum_{i=1}^{N} \log p(x_i \mid x_{\lt i})\right) $$

Lower is better. On WikiText-2, bf16 Llama 3.1 8B scores ~5.4; FP8 variants land within 0.05; int4 within 0.15. Fast to run (one pass over a few thousand tokens). Useful as the first signal — does nothing egregious; passes.

## Task benchmarks — lm-evaluation-harness

EleutherAI's harness runs standardized benchmarks across LLMs with consistent prompting. The common ones:

- **MMLU** — 57 subject multiple-choice. The "knowledge" sanity check.
- **ARC-Easy / ARC-Challenge** — Science QA.
- **GSM8K** — Grade-school math. Sensitive to quantization.
- **MATH** — Competition math. Hard; reveals reasoning regressions.
- **HumanEval / MBPP** — Code generation. Sensitive to tokenizer and sampling.
- **HellaSwag / Winogrande** — Common sense reasoning.

Typical regression tolerance: < 1 % absolute on MMLU, < 2 % on GSM8K, < 3 % on MATH for a production ship. Anything more deserves investigation.

## Running lm-eval-harness

```bash
pip install lm-eval

lm_eval --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,gpu_memory_utilization=0.85 \
    --tasks mmlu,gsm8k,arc_challenge,hellaswag \
    --batch_size 32 \
    --output_path ./eval_results
```

Takes 1–8 hours depending on tasks and hardware. Run it on your base and candidate in parallel; diff.

## Your own task eval

Leaderboards are a smoke test, not a ship signal. Build a private evaluation set of 100–500 real examples from your production distribution. For each, store a target behavior (correct output, acceptable vs. unacceptable patterns, format requirements). Score the candidate against that. This beats any public benchmark for your specific decision.

## Latency and throughput deltas

Use a load generator (Module 8 Ch. 5) to compare the base model and the optimized candidate under the same workload shape and SLO:

- TTFT P50, P99
- TPOT P50, P99
- Goodput at SLO
- Max sustainable RPS

Report deltas, not absolutes. The win from a 15% TPOT reduction may be worth a 0.5 % MMLU drop; a 0 % TPOT change is not worth any quality cost.

## A sign-off scorecard

| Axis | Base | Candidate | Delta | Tolerance | Pass? |
|---|---|---|---|---|---|
| MMLU | 68.5 | 68.3 | −0.2 | < 1.0 | ✓ |
| GSM8K | 76.2 | 75.8 | −0.4 | < 2.0 | ✓ |
| Private eval | 0.84 | 0.83 | −0.01 | < 0.03 | ✓ |
| TPOT P99 | 48 ms | 22 ms | −54 % | improve | ✓ |
| Max RPS @ SLO | 8 | 16 | +100 % | improve | ✓ |
| Weights VRAM | 16 GB | 8 GB | −50 % | improve | ✓ |

All green, ship. Any red, investigate.

## Pitfalls

- **Only measuring MMLU.** MMLU rarely regresses much. Tasks that stress reasoning or formatting regress more.
- **Comparing across engines.** Different engines have different sampling implementations; use the same engine for base and candidate.
- **Measuring at the wrong batch.** Quantization wins are batch-dependent. Measure at your production batch.
- **Skipping the private eval.** Public benchmarks don't cover your prompts, format, or domain. Your private 100 examples are worth more than every leaderboard.

<section class="faq">
<h2>FAQ</h2>
<h3>How big should my private eval be?</h3>
<p>100 examples catches gross regressions; 300–500 gives confidence. For life-critical applications, aim for 1,000+ with a holdout you don't tune against.</p>
<h3>Can I use GPT-4 as a judge?</h3>
<p>Yes — LLM-as-judge is now standard for open-ended tasks where exact match doesn't work. Cross-validate with human review occasionally to catch judge bias.</p>
<h3>What if MMLU improves on the quantized model?</h3>
<p>It happens, within noise (~0.5 %). Treat as "no regression" rather than "improvement" unless it's robust across seeds and benchmarks.</p>
<h3>Should I evaluate on the quantized or dequantized model?</h3>
<p>On the quantized one, running through the inference engine you'll serve with. That's what users see.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://github.com/EleutherAI/lm-evaluation-harness" target="_blank" rel="noreferrer">EleutherAI — lm-evaluation-harness</a></li>
<li><a href="https://huggingface.co/open-llm-leaderboard" target="_blank" rel="noreferrer">Hugging Face Open LLM Leaderboard</a></li>
<li><a href="https://arxiv.org/abs/2306.05685" target="_blank" rel="noreferrer">Zheng et al. — Judging LLM-as-a-Judge</a></li>
<li><a href="https://lmarena.ai/" target="_blank" rel="noreferrer">LMArena — human preference evaluation</a></li>
</ul>
</section>

## What's next

Module 4 is done — you can select, optimize, and evaluate a model. [Module 5 — Model Interaction Choices](?module=m5&chapter=inference-apis) covers the application-side half: APIs, streaming, sampling, tool use, and agents.
