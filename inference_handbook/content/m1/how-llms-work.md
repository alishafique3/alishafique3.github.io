<div class="chapter-meta">Module 1 · Chapter 2</div>

# How Large Language Models Work

<p class="lede">Every LLM in production today is a Transformer stack generating text one token at a time. To tune inference, you don't need to derive attention from scratch — but you do need a mental model of the five things the model does on every forward pass: embed, attend, transform, project, and sample.</p>

## The five-step mental model

1. **Embed** — Each input token ID maps to a dense vector (the *embedding*). A sequence of $s$ tokens becomes a matrix of shape $(s, d)$ where $d$ is the hidden size.
2. **Attend** — For every token, self-attention computes how much every other token in the context should influence it. This is the expensive step.
3. **Transform** — A feed-forward network (two matrix multiplies and a nonlinearity) acts on each position independently.
4. **Repeat** — Steps 2 and 3 are stacked $L$ times (24 for Qwen-0.5B, 80 for Llama-3 70B). Residual connections and layer norms keep the signal stable.
5. **Project & sample** — The final hidden state is projected to a vector of logits over the entire vocabulary ($V \approx$ 128k for modern models), softmax turns logits into probabilities, and a sampler picks the next token.

## The math that matters

Attention is defined as

$$ \text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V $$

where $Q$, $K$, and $V$ are projections of the input through learned weight matrices. For a single token being generated, only its query $Q$ is new; the keys and values of every previous token are read from the **KV cache**. This is the single most important fact for inference performance — and the subject of Module 6.

The feed-forward network is a sandwich:

$$ \text{FFN}(x) = W_2 \, \sigma(W_1 x) $$

where $\sigma$ is typically SiLU or GELU, and the intermediate size is roughly $4d$. For a 70B model with $d = 8{,}192$ and $i = 28{,}672$, each layer's FFN alone is ~1.1 B parameters.

## Autoregressive generation

LLMs don't predict a sentence in one shot. They predict *the next token*, append it to the input, and predict again. Given input $x_{1..n}$, the model computes

$$ P(x_{n+1} \mid x_{1..n}) $$

a sampler picks $x_{n+1}$, and the loop continues until an end-of-sequence token or a max-length limit is hit. This loop is the **decode phase** and it is memory-bandwidth bound — each new token requires reading the entire set of model weights and KV cache from HBM.

## A minimal generation loop

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                             torch_dtype=torch.bfloat16,
                                             device_map="cuda")

ids = tok("The three axes of inference engineering are",
          return_tensors="pt").input_ids.cuda()

past = None
for _ in range(30):
    with torch.inference_mode():
        out = model(ids[:, -1:] if past else ids, past_key_values=past, use_cache=True)
    past = out.past_key_values                    # KV cache grows each step
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
    ids = torch.cat([ids, next_id], dim=-1)

print(tok.decode(ids[0]))
```

The first forward pass is *prefill* — it processes the whole prompt at once. Every pass after that is *decode* — one new token, reading the growing KV cache. Chapter 5 treats this split in depth.

## Why this matters for engineers

- Model weights dominate memory: for 70B at BF16, weights alone are ~140 GB.
- KV cache grows linearly with context length — for Llama-3 70B at 128k context it is roughly 40 GB per sequence (see Module 3).
- FLOPs per generated token are approximately $2P$ (where $P$ is parameter count); bytes moved per decode token are also approximately $2P$ — so decode is **memory-bound** for anything but enormous batches.

<section class="faq">
<h2>FAQ</h2>
<h3>Do I need to understand attention math to serve LLMs?</h3>
<p>No. But knowing where attention memory and compute go makes KV cache, FlashAttention, and parallelism choices much easier to reason about.</p>
<h3>Why is the KV cache necessary?</h3>
<p>Without it, each decode step would recompute keys and values for all previous positions — making decode $O(n^2)$ in compute per step instead of $O(n)$.</p>
<h3>What's the "2P" rule of thumb for decode?</h3>
<p>For one new decode token in a model with $P$ parameters, FLOPs ≈ $2P$ and bytes moved ≈ $2P$. The ratio is near 1, so decode is always memory-bound at small batch.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">Vaswani et al. — Attention Is All You Need</a></li>
<li><a href="https://blog.eleuther.ai/transformer-math/" target="_blank" rel="noreferrer">EleutherAI — Transformer Math 101</a></li>
<li><a href="https://huggingface.co/learn/nlp-course/chapter1" target="_blank" rel="noreferrer">Hugging Face NLP Course — Transformer basics</a></li>
</ul>
</section>

## What's next

With the mental model in hand, [Chapter 3](?module=m1&chapter=tokens) zooms into what "token" really means, [Chapter 5](?module=m1&chapter=prefill-and-decode) formalizes prefill vs. decode, and [Chapter 7](?module=m1&chapter=performance-metrics) defines the latency and throughput metrics you will tune against.
