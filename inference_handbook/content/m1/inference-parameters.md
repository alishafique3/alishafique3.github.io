<div class="chapter-meta">Module 1 · Chapter 6</div>

# Understanding LLM Inference Parameters

<p class="lede">After the model produces logits for the next token, a <strong>sampler</strong> turns those logits into a single token to emit. The sampler is controlled by a small set of parameters — temperature, top-k, top-p, min-p, repetition penalty — exposed in nearly every inference API. Each parameter reshapes the distribution the sampler draws from. Picking them well is the cheapest quality lever in production.</p>

## From logits to probabilities

The model output at each position is a vector of logits $z \in \mathbb{R}^V$. The softmax turns this into a probability distribution:

$$ p_i = \frac{e^{z_i}}{\sum_j e^{z_j}} $$

Greedy decoding picks $\arg\max_i p_i$ every step — deterministic but repetitive. Sampling draws from the distribution — more varied but can produce low-quality tokens if the tail is not pruned.

## Temperature

Temperature $T$ scales logits before softmax:

$$ p_i(T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}} $$

- $T \to 0$: distribution collapses onto the top token — effectively greedy.
- $T = 1$: raw model distribution.
- $T > 1$: distribution flattens; more risk, more creativity.

Production chat defaults are 0.6–0.8. Analytical / structured tasks often use $T = 0$ or $T = 0.1$. Note: many serving engines treat $T = 0$ as a special-case greedy path to avoid division by zero.

## Top-k and top-p (nucleus)

**Top-k** keeps only the $k$ highest-probability tokens, renormalizes, and samples. Simple, but a fixed $k$ is wrong when the distribution is sharply peaked (should narrow) or flat (should widen).

**Top-p (nucleus)** keeps the smallest set of tokens whose cumulative probability exceeds $p$, then samples. Adaptive: narrow when the model is confident, wider when it is not. The dominant setting in production: $p = 0.9$ or $0.95$.

Top-p and top-k are typically applied *after* temperature.

## Min-p

A newer alternative: keep only tokens with probability $\ge p_{\min} \cdot \max_i p_i$ — i.e., within a fixed factor of the top token. Avoids a known pathology of top-p where a strong top token still allows a long tail of low-probability candidates. Setting `min_p = 0.05` often matches or beats top-p on quality benchmarks with simpler behavior.

## Repetition and presence penalties

**Repetition penalty** divides the logit of previously-seen tokens by a constant > 1 (OpenAI convention) or subtracts a constant (Anthropic / HF presence_penalty / frequency_penalty). Discourages literal loops. Typical: 1.05–1.2 for repetition_penalty; 0.3–0.7 for presence/frequency penalty.

These are blunt. Overzealous penalties make the model refuse to repeat required terminology (JSON field names, function names).

## Length controls

- **max_tokens** — hard cap on output length. Do not leave it unbounded in production.
- **stop sequences** — strings that halt generation. Useful for template boundaries (e.g., stop at `\n\n` for single-paragraph answers).
- **min_tokens** — rare but exposed by some engines; useful for forcing at least a short answer.

## A typical production config

```python
client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=messages,
    temperature=0.7,          # moderate creativity for chat
    top_p=0.95,               # nucleus sampling
    max_tokens=1024,          # hard cap
    frequency_penalty=0.3,    # mild repetition suppression
    stop=["\n\nUser:"],       # avoid fake multi-turn
)
```

## Pitfalls

- **Temperature 0 as a quality panacea.** Deterministic does not mean better. Greedy decoding exhibits more hallucination on many tasks than $T = 0.7$ with top-p.
- **Combining aggressive top-p with aggressive temperature.** Two filters both narrowing the distribution can over-prune and cause degeneracy.
- **Forgetting max_tokens in production.** A single runaway generation can burn a GPU for minutes.
- **Trusting penalty parameter semantics across providers.** OpenAI's `frequency_penalty`, HuggingFace's `repetition_penalty`, and vLLM's `repetition_penalty` are all different formulas. Read the docs.

<section class="faq">
<h2>FAQ</h2>
<h3>What's the "right" temperature?</h3>
<p>Chat: 0.6–0.8. Analytical/structured: 0 or 0.1. Creative writing: 0.9–1.1. Combine with <code>top_p = 0.95</code> or <code>min_p = 0.05</code> for the modern production default.</p>
<h3>Does setting <code>top_p = 1</code> disable nucleus sampling?</h3>
<p>Effectively yes — <code>p = 1</code> keeps the whole distribution. Setting <code>top_p = 1</code> and <code>top_k = 0</code> gives pure temperature sampling.</p>
<h3>Why do I get different outputs with the same seed?</h3>
<p>Sampling is seeded, but batch composition changes numerics at every step unless you also fix determinism flags on the engine (vLLM has <code>--enforce-eager</code> and seed-per-request APIs; most engines don't guarantee bit-exact repro at batch > 1).</p>
<h3>Are OpenAI's and vLLM's penalty parameters the same?</h3>
<p>No — different formulas. OpenAI's <code>frequency_penalty</code> and <code>presence_penalty</code> subtract from logits; HF/vLLM's <code>repetition_penalty</code> divides. Check the docs before porting values.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://arxiv.org/abs/1904.09751" target="_blank" rel="noreferrer">Holtzman et al. — The Curious Case of Neural Text Degeneration (nucleus sampling)</a></li>
<li><a href="https://arxiv.org/abs/2407.01082" target="_blank" rel="noreferrer">Nguyen et al. — Min-P Sampling</a></li>
<li><a href="https://docs.vllm.ai/en/stable/api/sampling_params.html" target="_blank" rel="noreferrer">vLLM — SamplingParams documentation</a></li>
</ul>
</section>

## What's next

Samplers are a client-facing concept. On the serving side, every sampling parameter adds a small amount of per-token work — see [M5 Ch. 2](?module=m5&chapter=openai-anthropic-apis) for how APIs surface them, and [M6 Ch. 2](?module=m6&chapter=batching-scheduling) for how they interact with continuous batching.
