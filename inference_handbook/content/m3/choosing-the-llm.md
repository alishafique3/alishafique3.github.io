<div class="chapter-meta">Module 3 · Chapter 2</div>

# Choosing the Right LLM — Selection and Configuration

<p class="lede">The 2026 open-weight menu is overwhelming: Llama, Qwen, Mistral, Gemma, DeepSeek, Phi, GLM, Kimi, Falcon, MiniMax — each in 3–5 size tiers and 2–3 variants. Your choice sets your GPU budget, your context-length limit, your API shape, and your licensing exposure. This chapter is a practical guide to narrowing the menu to one or two candidates before you benchmark.</p>

## Size tiers that actually ship

| Tier | Examples | Fits on | Use case |
|---|---|---|---|
| < 3 B | Qwen 2.5 1.5B, Phi-3 Mini, Gemma 2 2B | Mobile, consumer GPU, free Colab | Edge, autocomplete, classification |
| 7–9 B | Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B, Gemma 2 9B | Single A10 / L40S | Chat, light RAG, drafter for spec decoding |
| 14–34 B | Qwen 2.5 14B/32B, Phi-3 Medium 14B, Mixtral-Lite | Single H100 | Quality chat, code, multi-turn |
| 70 B | Llama 3.1 / 3.3 70B, Qwen 2.5 72B | 2 × H100 or 1 × H200/B200 | Flagship open |
| MoE 100 B+ | DeepSeek V3, Mixtral 8×22B, Qwen3-MoE | 4–8 × H100/H200 | Frontier quality, high token economics |

## Architectural variants that matter

- **Base vs. instruct.** Base models predict the next token on raw text; instruct models follow user turns. For chat and API use, pick the instruct variant. Mixing them accidentally produces nonsense.
- **Attention type.** Modern 70 B+ models use GQA (8 KV heads) or MLA (DeepSeek). This dominates KV cache size — see [M2 Ch. 4](?module=m2&chapter=gpu-memory-for-inference).
- **MoE vs. dense.** MoE models (DeepSeek, Mixtral) hold many more parameters but activate a fraction per token. Memory bill scales with total; compute with active. Great throughput; harder to shard efficiently.
- **Native context length.** Advertised context in the model card. Realistic long-context quality is usually 25–50 % of the advertised number for non-RoPE-extended models — measure with needle-in-haystack (M4.6).
- **Tool-use fine-tune.** Function-calling-tuned variants exist for most families. Use them if your app relies on structured tool outputs.

## The config.json decoder ring

Every Hugging Face model ships a `config.json`. The fields that matter for infra planning:

```text
num_hidden_layers        → L (depth)
hidden_size              → d (width of residual stream)
intermediate_size        → i (FFN width, usually ~4d)
num_attention_heads      → a (query heads)
num_key_value_heads      → kv (KV heads — matters for GQA!)
vocab_size               → V (embedding + LM head size)
max_position_embeddings  → advertised context length
rope_theta               → RoPE base (10,000 default, 500,000 for long-context tunes)
tie_word_embeddings      → true → one embedding matrix for both input and LM head
torch_dtype              → the native precision of the saved weights
```

These are the inputs to every memory calculator. The [Model Explorer](https://alishafique3.github.io/model-explorer/) renders them live with worked parameter counts.

## Weight formats you will see

- **safetensors** — Hugging Face standard. Memory-mapped loading. Default for GPU inference engines.
- **GGUF** — llama.cpp's self-contained format. Includes tokenizer and metadata. The default for CPU and consumer-GPU inference.
- **AWQ / GPTQ checkpoints** — pre-quantized weight bundles. See M4.3.

## Licensing — the quiet gotcha

- **Llama 3.1 / 3.3** — commercial use allowed; attribution required; > 700 M monthly active users must request a license from Meta.
- **Qwen 2.5** — Apache 2.0 for 0.5 B – 32 B; 72 B has its own Qwen license (commercial allowed, > 100 M monthly active users must contact Alibaba).
- **Mistral 7B, Mixtral** — Apache 2.0.
- **Gemma** — Gemma license, custom.
- **Phi-3** — MIT.
- **DeepSeek V3** — DeepSeek license; commercial allowed with notice.

Have legal sign off *before* integration if you are at scale. License terms around redistribution of fine-tunes are a particular minefield.

## Benchmarks worth trusting

Model leaderboards are a swamp. The ones worth consulting:

- **LMArena** — human preference on blind pairwise comparison. The closest thing to "which model do users actually like?"
- **lm-eval-harness** (EleutherAI) — reproducible task benchmarks (ARC, MMLU, GSM8K, MATH). The standard for defensible internal evaluation.
- **Hugging Face Open LLM Leaderboard** — standardized runs of lm-eval. Good for sanity-checking.
- **Your own private eval** — the only benchmark that matters in production. Even 50 well-chosen prompts beats a broad leaderboard for your workload.

<section class="faq">
<h2>FAQ</h2>
<h3>Instruct or chat variant — which?</h3>
<p>They are usually the same model with a different name. Use whichever one the chat template in the tokenizer says to use.</p>
<h3>Should I always pick the newest release?</h3>
<p>No. Newer often means less community tooling. Llama 3.1 has more mature quantized checkpoints and drafter pairs than fresh releases.</p>
<h3>How long before a model "goes stale"?</h3>
<p>Behaviorally: 12–24 months. Economically: when a new release matches your current quality at half the parameters.</p>
<h3>Can I mix tokenizers across models in one app?</h3>
<p>Yes, but token-count billing and context-limit checks need to use the correct tokenizer per model.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://lmarena.ai/" target="_blank" rel="noreferrer">LMArena — human preference leaderboard</a></li>
<li><a href="https://github.com/EleutherAI/lm-evaluation-harness" target="_blank" rel="noreferrer">EleutherAI — lm-evaluation-harness</a></li>
<li><a href="https://huggingface.co/open-llm-leaderboard" target="_blank" rel="noreferrer">Hugging Face Open LLM Leaderboard</a></li>
<li><a href="https://alishafique3.github.io/model-explorer/" target="_blank" rel="noreferrer">Model Explorer — interactive config.json walkthrough</a></li>
</ul>
</section>

## What's next

With the model chosen, size the GPU in [Chapter 3](?module=m3&chapter=gpu-selection-memory), plan capacity in [Chapter 4](?module=m3&chapter=capacity-planning), then pick the framework in [Chapter 5](?module=m3&chapter=inference-framework).
