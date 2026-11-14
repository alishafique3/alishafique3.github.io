<div class="chapter-meta">Module 1 · Chapter 3</div>

# Understanding Tokens and Tokenization

<p class="lede">An LLM does not see text. It sees sequences of <strong>token IDs</strong> — integers. Tokenization is the first and last step of every inference request, converting text into those IDs and back again. For an infra engineer, two facts matter: tokenization decides the length of every prompt and completion in the units your model counts, and <em>different tokenizers compress text at different rates</em> — which directly moves your cost-per-request even on identical hardware.</p>

## What a token actually is

A token is a chunk of bytes mapped to a vocabulary index. The vocabulary size $V$ is typically 32k–200k for modern LLMs — 32,000 for Llama 2, 128,256 for Llama 3, 151,936 for Qwen 2.5, ~200k for GPT-4o. An English token averages roughly 4 characters or 0.75 words, but the ratio varies enormously by content type and language. Code is denser; non-Latin scripts are far less dense (a Mandarin character often takes 2–3 tokens; a Japanese character can take 1–4).

## Byte-pair encoding and SentencePiece

**BPE** (Byte-Pair Encoding) starts from raw bytes and iteratively merges the most frequent adjacent pair into a new vocabulary entry, until the target vocab size is reached. GPT-2, GPT-4, Llama 3, and Qwen all use BPE-family tokenizers, though with implementation differences (tiktoken, HuggingFace tokenizers, Qwen's own).

**SentencePiece** is the dominant alternative — used by Llama 2, Gemma, T5, Mistral. It operates on Unicode characters (not bytes) and ships Unigram and BPE variants. It handles whitespace explicitly as a character, which is why SentencePiece tokens often include a leading space (`▁word`).

The practical takeaway: a prompt tokenized with Llama 3's tiktoken-style BPE is typically 10–20% shorter than the same prompt tokenized with Llama 2's SentencePiece. That is a direct speedup on prefill and a direct cost reduction on priced APIs.

## Parameter cost of the vocabulary

The embedding matrix and (if untied) the LM head are both of shape $(V, d)$:

$$ \text{embedding params} = V \cdot d, \qquad \text{LM head params} = V \cdot d \; (\text{untied}) $$

For Llama 3 8B ($V = 128{,}256$, $d = 4096$), the embedding alone is 525 M parameters — 6.5% of the whole model. Most modern small models *tie* the embedding and LM head (one matrix shared) to save this cost; large models often don't, because the LM head quality matters more at scale.

## A tokenizer demo

```python
from transformers import AutoTokenizer

for name in ["meta-llama/Llama-3.1-8B-Instruct",
             "Qwen/Qwen2.5-7B-Instruct",
             "mistralai/Mistral-7B-Instruct-v0.3"]:
    tok = AutoTokenizer.from_pretrained(name)
    text = "Large language models have become remarkably capable."
    ids = tok(text).input_ids
    print(f"{name:55s} vocab={tok.vocab_size:>7} tokens={len(ids):>3}")
```

Expected rough output: Llama 3 ≈ 11 tokens, Qwen ≈ 10 tokens, Mistral SentencePiece ≈ 13 tokens. Same text, different bill.

## Chat templates and special tokens

Instruction-tuned models expect their prompts wrapped in a chat template — a specific sequence of special tokens like `<|im_start|>`, `<|im_end|>`, or Llama's `<|begin_of_text|><|start_header_id|>user<|end_header_id|>`. If you serve a chat-tuned model with the wrong template or no template at all, output quality collapses. Every tokenizer ships the correct template; use `tok.apply_chat_template(messages)` and do not hand-roll.

## Why tokenizers are the hidden tax

- **Price-per-token APIs** charge by the provider's tokenizer, not by characters. Switching models can change your bill by 10–30% even at the same list price.
- **Context limits** are in tokens. A 128k-token Llama context fits more text than a 128k-token Mistral context, because Llama's tokenizer compresses better on English.
- **Prefill time** is proportional to token count, not character count. A worse tokenizer makes long-prompt TTFT visibly slower.
- **Non-English users pay more** on API services because their text tokenizes less efficiently. This is a quiet source of bias in model economics.

## Pitfalls

- **Mismatched tokenizers between client and server.** If your client estimates token counts with a different tokenizer than the server, usage-based billing and context-limit checks break.
- **Forgetting special-token overhead.** Chat templates typically add 10–30 tokens per message turn. In a 10-turn conversation that is 100+ tokens before any user content.
- **Trusting word counts.** "This prompt is 50 words" means nothing. Count tokens.

<section class="faq">
<h2>FAQ</h2>
<h3>Why not use characters or words directly?</h3>
<p>Characters produce sequences too long to model efficiently; words produce vocabularies too large and brittle. BPE/SentencePiece give a middle ground that handles out-of-vocabulary text gracefully by falling back to bytes.</p>
<h3>Can I estimate tokens from character count?</h3>
<p>Roughly: English ≈ 4 chars/token, code ≈ 3 chars/token, Mandarin ≈ 1–2 tokens/char, emoji and uncommon scripts can be worse. For billing, always count with the real tokenizer.</p>
<h3>Do Llama and Qwen count tokens the same way?</h3>
<p>No. Same text can tokenize to different counts. Switching models changes your bill even at the same price per million tokens.</p>
<h3>What is <code>apply_chat_template()</code>?</h3>
<p>The tokenizer's canonical formatter that wraps a message list in the model's expected special tokens and roles. Do not hand-roll it.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://huggingface.co/docs/tokenizers/" target="_blank" rel="noreferrer">Hugging Face Tokenizers library</a></li>
<li><a href="https://github.com/openai/tiktoken" target="_blank" rel="noreferrer">OpenAI tiktoken — BPE reference</a></li>
<li><a href="https://www.youtube.com/watch?v=zduSFxRajkE" target="_blank" rel="noreferrer">Karpathy — Let's build the GPT Tokenizer</a></li>
</ul>
</section>

## What's next

With tokens defined, [Chapter 5 — Prefill and Decode](?module=m1&chapter=prefill-and-decode) explains what happens to those IDs once the model sees them, and [Chapter 7](?module=m1&chapter=performance-metrics) defines the latency metrics measured in tokens.
