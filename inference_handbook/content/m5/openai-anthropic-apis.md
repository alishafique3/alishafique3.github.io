<div class="chapter-meta">Module 5 · Chapter 2</div>

# OpenAI- and Anthropic-Compatible APIs

<p class="lede">Two API shapes dominate 2026 inference: OpenAI's <code>/v1/chat/completions</code> and Anthropic's <code>/v1/messages</code>. Every major open-source engine implements both. They are mostly the same, but their small differences matter when porting code or writing clients.</p>

## Side-by-side — the request shape

OpenAI:

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user",   "content": "Explain KV cache."}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": true
}
```

Anthropic:

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "system": "You are helpful.",
  "messages": [
    {"role": "user", "content": "Explain KV cache."}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": true
}
```

## Three practical differences

1. **System prompt location.** OpenAI puts it in `messages` with role `"system"`. Anthropic hoists it to a top-level `system` field. When porting code, remember to split it out.
2. **Content shape.** OpenAI accepts `content` as either a string or a list of multimodal parts. Anthropic's `content` is always a list of typed parts (`{"type":"text", ...}`, `{"type":"image", ...}`). Clients that pass strings to Anthropic must wrap them.
3. **Tool use.** OpenAI returns `tool_calls` inside the assistant message. Anthropic returns `content` blocks of type `tool_use`. The payload shape of the tool call differs — see M5.6.

## Streaming differences

Both use Server-Sent Events but with different event formats.

OpenAI (one example event):

```
data: {"choices":[{"delta":{"content":"KV cache"}}]}

data: [DONE]
```

Anthropic (one example event):

```
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"KV cache"}}

event: message_stop
data: {"type":"message_stop"}
```

Anthropic is more verbose and more structured. OpenAI is terser and used by more client SDKs.

## Which do self-hosted engines expose?

| Engine | OpenAI API | Anthropic API |
|---|---|---|
| vLLM | ✓ (default) | Partial via proxy adapters |
| SGLang | ✓ | Partial |
| TensorRT-LLM / Triton | ✓ | Via middleware |
| TGI | ✓ | Partial |
| llama.cpp (server) | ✓ | — |

For 95% of use cases, the OpenAI shape is the lingua franca; Anthropic compatibility exists mainly for teams that already code against Claude.

## Launching a vLLM OpenAI-compatible server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192
```

Then any OpenAI SDK — Python, TypeScript, Go, Ruby — connects with `base_url="http://localhost:8000/v1"`. No code changes vs. hosted OpenAI.

## Launching Anthropic-compatible on top of vLLM

There is no first-class Anthropic endpoint in vLLM today. The common pattern is to put a small proxy (e.g., [LiteLLM](https://github.com/BerriAI/litellm)) in front that translates Anthropic-shaped requests into OpenAI-shaped and back.

## A dual-SDK client that works against either

```python
import os

if os.getenv("USE_ANTHROPIC"):
    from anthropic import Anthropic
    client = Anthropic(base_url=os.environ["BASE"])
    r = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        system="You are helpful.",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=256,
    )
    print(r.content[0].text)
else:
    from openai import OpenAI
    client = OpenAI(base_url=os.environ["BASE"], api_key="x")
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user",   "content": "Hello"},
        ],
        max_tokens=256,
    )
    print(r.choices[0].message.content)
```

## Pitfalls

- **Assuming "both are OpenAI under the hood" when it comes to tool-use payloads.** They differ enough that your tool-handling code needs branches.
- **Mixing streaming formats in one UI.** Clients that render SSE must handle both event shapes if you support both endpoints.
- **Trusting error schemas to match.** OpenAI returns `{"error":{"message":...,"type":...}}`. Anthropic returns a different shape. Normalize at the gateway if you expose both.
- **Shipping with the wrong SDK version.** The OpenAI Python SDK had a breaking change in 1.0 (2024); older code that uses `openai.ChatCompletion.create(...)` will not work.

<section class="faq">
<h2>FAQ</h2>
<h3>Can I use the Anthropic SDK against a vLLM server?</h3>
<p>Only if you put an Anthropic-shape proxy in front. The direct vLLM server exposes OpenAI shape only.</p>
<h3>Which schema has better tool support?</h3>
<p>Comparable. Anthropic's tool schema is slightly more explicit about typing; OpenAI's is terser. See M5.6.</p>
<h3>Why would I use the Anthropic endpoint?</h3>
<p>If you're migrating code written against the Anthropic SDK and want to keep one client. Otherwise OpenAI is the more universal target.</p>
<h3>Are the model names standardized?</h3>
<p>No. Each engine decides how to advertise models. Many servers accept any string and map to the loaded weights.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://platform.openai.com/docs/api-reference/chat" target="_blank" rel="noreferrer">OpenAI — Chat Completions reference</a></li>
<li><a href="https://docs.anthropic.com/en/api/messages" target="_blank" rel="noreferrer">Anthropic — Messages API reference</a></li>
<li><a href="https://github.com/BerriAI/litellm" target="_blank" rel="noreferrer">LiteLLM — Multi-provider proxy</a></li>
</ul>
</section>

## What's next

Streaming is the next deep dive — [Chapter 3](?module=m5&chapter=streaming-vs-buffered) — followed by prompt engineering, structured outputs, and tool use.
