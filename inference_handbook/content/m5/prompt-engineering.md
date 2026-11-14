<div class="chapter-meta">Module 5 · Chapter 4</div>

# Prompt Engineering and Context Management

<p class="lede">The prompt is the model's whole world. Every token you include costs latency and money, every token you omit might have been the one that mattered. This chapter covers the engineering view of prompts: chat templates, system prompts, few-shot patterns, context-window hygiene, and how to construct prompts that play nicely with prefix caching.</p>

## The three roles

Every chat API has three message roles:

- **`system`** — sets persistent behavior, personality, output format. Visible to the model, not to the user.
- **`user`** — the input the model should respond to.
- **`assistant`** — model-generated turns, replayed on multi-turn.

A fourth role, **`tool`**, carries function-call results in tool-using agents (M5.6).

## Chat templates — the invisible wrapper

Every instruction-tuned model expects messages wrapped in a specific template. Llama 3's template uses `<|begin_of_text|><|start_header_id|>user<|end_header_id|>...`; Qwen uses `<|im_start|>user\n...<|im_end|>`. The tokenizer applies it:

```python
msgs = [{"role": "user", "content": "Hello"}]
tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
```

Serving engines apply the template automatically if you use the chat endpoint. Hand-rolling the template is the #1 source of "the model used to be smart and suddenly isn't" bugs.

## System prompt patterns

Good system prompts share three traits:

1. **Specific role.** "You are a senior Python engineer reviewing this code" beats "You are helpful".
2. **Output format.** "Return JSON matching the schema below" beats "Be concise".
3. **Guardrails.** "Refuse to generate harmful content. Decline politely with a brief reason." beats implicit expectation.

Keep system prompts short but intentional. Every token is on the prefill critical path of every request.

## Few-shot examples

Inline examples teach the pattern you want:

```
System: You are a JSON entity extractor.

User: "Apple stock closed at $190 on Friday."
Assistant: {"companies": ["Apple"], "prices": [190]}

User: "Microsoft hit $420 yesterday."
Assistant: {"companies": ["Microsoft"], "prices": [420]}

User: "{new input here}"
```

Two or three examples usually suffice. More than five rarely helps and increases prefill cost. Fine-tuning eventually beats few-shot for scale, but few-shot is the cheapest way to prototype.

## Context-window hygiene

Just because a model advertises 128k context doesn't mean you should fill it. Three honest rules:

- **Long-context quality degrades.** Most non-RoPE-extended models lose coherence past 25–50% of advertised window. Measure with needle-in-haystack.
- **Prefill cost is $O(s^2)$.** A 128k prefill is not 2× a 64k prefill. It's 4×.
- **Decode TPOT rises with context.** Each token reads a larger KV cache.

Rule of thumb: retrieve the *top-k relevant snippets*, not *everything that might be relevant*. RAG quality drops when the model is drowning in off-topic context.

## Multi-turn conversation state

The client, not the server, maintains conversation history by default. Each turn sends the full history:

```python
history = []
def chat(user_msg):
    history.append({"role": "user", "content": user_msg})
    r = client.chat.completions.create(model="m", messages=history)
    assistant_msg = r.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_msg})
    return assistant_msg
```

At some point the history exceeds your context budget. Choices:

- **Truncate** — drop the oldest turns. Simple; loses history.
- **Summarize** — periodically compress the oldest turns into a summary message. Preserves intent, loses detail.
- **Server-side state** — some engines (SGLang) support session caching, which keeps KV state across turns server-side. Dramatically cuts prefill cost on multi-turn.

## Prefix caching awareness

Prefix caching (M6.3) stores the KV state of common prefixes. You want prompts to share prefixes across requests. Patterns that maximize cache reuse:

- Put the system prompt first, before anything variable.
- Keep few-shot examples in the system prompt or just after, before user content.
- Put user-specific content last.
- Avoid embedding timestamps or request IDs anywhere above the user turn.

Result: the long system+examples prefix tokenizes once and gets reused across thousands of requests, dropping prefill cost by 50–90%.

## A prefix-caching-friendly prompt structure

```
[System prompt — fixed, same across all requests]
[Few-shot examples — fixed]
[Retrieved context chunks — varies per request, placed before user input]
[User message]
```

The upper portion (system + examples) hits cache; the lower portion is the only thing the model actually has to prefill.

## Pitfalls

- **Hand-rolling chat templates.** Use `apply_chat_template()`. Hand-rolled templates miss edge cases (special tokens, assistant prefill) that matter.
- **Appending timestamps to system prompts.** "You are helpful. Current time: 2026-11-13 02:14:33" — every request is unique and prefix caching never hits.
- **Trusting advertised context length as "the whole thing is good".** Measure quality on your long-context tasks before shipping.
- **Letting conversation history grow without bound.** Eventually you hit the context limit and the engine errors. Truncate or summarize proactively.

<section class="faq">
<h2>FAQ</h2>
<h3>Where does the system prompt live in memory?</h3>
<p>In the KV cache. If multiple requests share it and the engine supports prefix caching, only one copy exists.</p>
<h3>Is "zero-shot" always worse than few-shot?</h3>
<p>No. Well-tuned chat models often handle zero-shot excellently. Few-shot shines for specific output formats or unusual tasks.</p>
<h3>How do I measure prompt length?</h3>
<p>Call <code>tokenizer.apply_chat_template(msgs)</code> and len() the result. Character counts lie.</p>
<h3>Can I cache per-user prefixes?</h3>
<p>Yes with prefix caching enabled — any shared prefix gets cached, not just system prompts. Per-user system prompts work fine.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://huggingface.co/docs/transformers/chat_templating" target="_blank" rel="noreferrer">Hugging Face — Chat templating documentation</a></li>
<li><a href="https://arxiv.org/abs/2307.03172" target="_blank" rel="noreferrer">Liu et al. — Lost in the Middle (long-context attention degradation)</a></li>
<li><a href="https://docs.vllm.ai/en/stable/features/prefix_caching.html" target="_blank" rel="noreferrer">vLLM — Prefix caching documentation</a></li>
</ul>
</section>

## What's next

Structured outputs ([Chapter 5](?module=m5&chapter=structured-outputs)) and function calling ([Chapter 6](?module=m5&chapter=function-calling-mcp)) build on well-formed prompts to produce reliable machine-readable output.
