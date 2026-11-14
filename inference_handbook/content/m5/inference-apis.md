<div class="chapter-meta">Module 5 · Chapter 1</div>

# Understanding LLM Inference APIs

<p class="lede">An inference API is the contract between an LLM server and an application. In 2026 the ecosystem has converged on two shapes — OpenAI's chat/completions and Anthropic's messages — and every major serving engine (vLLM, SGLang, TGI, TensorRT-LLM) implements both. This chapter covers the request-response patterns, error semantics, and design constraints that every API-side decision maps back to.</p>

## The request-response pattern

A chat request carries a list of messages (role + content), a model name, sampling parameters, and length bounds. The server returns one of:

- A completed assistant message (buffered).
- A stream of partial message chunks (streaming).
- A structured error with a code and retry hint.

Nearly every production app uses streaming (M5.3), but both shapes are first-class in every API.

## Core request fields

| Field | Meaning | Typical |
|---|---|---|
| `model` | Model identifier. Serving-engine-specific. | `"llama-3.1-70b-instruct"` |
| `messages` | List of `{role, content}`. Roles: `system`, `user`, `assistant`, `tool`. | See below |
| `temperature`, `top_p` | Sampling (M1.6). | `0.7, 0.95` |
| `max_tokens` | Hard output cap. Never leave blank in prod. | `1024` |
| `stop` | Strings that halt generation. | `["\n\nUser:"]` |
| `stream` | Server-sent events if true. | `true` |
| `tools` | Function schemas for tool-calling. | See M5.6 |
| `response_format` | JSON mode / grammar. | `{"type":"json_object"}` |

## Core response fields

| Field | Meaning |
|---|---|
| `id` | Server-assigned request ID — log this for debugging. |
| `choices[0].message` | The assistant's response. |
| `choices[0].finish_reason` | `"stop"`, `"length"`, `"tool_calls"`, `"content_filter"`. |
| `usage` | `prompt_tokens`, `completion_tokens`, `total_tokens`. |

Streaming replaces `choices[0].message` with repeated `choices[0].delta` chunks. The final chunk carries `finish_reason` and (optionally) `usage`.

## Idempotency, retries, and errors

LLM requests are not idempotent — the same prompt + non-zero temperature yields a different answer. If you need "exactly once", generate a client-side idempotency key and have the server cache results by key (most engines don't do this by default).

Retries belong in the client. Retry on:

- `429 Too Many Requests` — with exponential backoff.
- `503 Service Unavailable` — same.
- `500 Internal Server Error` — usually, but inspect first.

Do not retry on `400 Bad Request` (usually a schema error) or `401/403` (auth).

## Minimal client — OpenAI Python SDK

```python
from openai import OpenAI, RateLimitError
import time, random

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

def ask(prompt, retries=3):
    for i in range(retries):
        try:
            return client.chat.completions.create(
                model="llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,
            )
        except RateLimitError:
            time.sleep(2 ** i + random.random())
    raise RuntimeError("giving up after retries")
```

The same client works against OpenAI's hosted API or any vLLM/SGLang/TGI deployment. That interoperability is the point.

## Rate limiting and admission control

Production APIs limit:

- **RPM / RPS** — requests per unit time.
- **TPM** — tokens per minute (input + output combined).
- **Concurrent requests** — in-flight count per key.

On the server side, the inference engine's scheduler (M6.1) decides which incoming request runs now vs. queues. Admission control at the gateway happens before anything reaches the scheduler.

## API design constraints that bite later

- **Timeouts.** A 5-minute client timeout vs. a 60-second server timeout will mislead both sides. Align them explicitly.
- **Streaming and load balancers.** HTTP/1.1 SSE needs long-lived connections; some LBs silently kill them after 30 s. Test.
- **Logging.** Log request ID + prompt tokens + completion tokens + finish_reason, not full prompts. Prompts contain user data.
- **Tokenization at the client.** Clients that pre-tokenize must use the same tokenizer as the server (M1.3). Otherwise usage billing diverges.

## Pitfalls

- **Treating completions and chat as interchangeable.** Chat applies the model's chat template; raw completions do not. A chat model fed via the completions endpoint usually produces lower quality.
- **Missing `max_tokens` in production.** One runaway generation can burn a GPU for minutes and bill a lot of tokens.
- **Returning streaming errors as partial output.** If a stream fails mid-flight, close it with an error frame. Silent truncation causes silent bugs in downstream parsers.
- **Hand-rolling the chat template client-side.** Use `tokenizer.apply_chat_template()` when you must; otherwise send `messages` and let the engine format.

<section class="faq">
<h2>FAQ</h2>
<h3>Chat vs. completions — which should I use?</h3>
<p>Chat. Completions is legacy — kept for compatibility. Instruction-tuned models require the chat template, which the chat endpoint applies automatically.</p>
<h3>What is the "system" role?</h3>
<p>A message placed at the top of the conversation that sets behavior ("You are a helpful assistant specializing in X."). Treated specially by most models and token-cached by most engines.</p>
<h3>How do I pass metadata per request?</h3>
<p>OpenAI accepts a <code>user</code> field for user IDs. For custom metadata, either embed in a system message or pass out-of-band headers.</p>
<h3>Is HTTPS required for self-hosted?</h3>
<p>Not required by the engines, but required by any sensible production gateway. Put nginx/Envoy/Traefik in front with TLS.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://platform.openai.com/docs/api-reference/chat" target="_blank" rel="noreferrer">OpenAI — Chat Completions API reference</a></li>
<li><a href="https://docs.anthropic.com/en/api/messages" target="_blank" rel="noreferrer">Anthropic — Messages API reference</a></li>
<li><a href="https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html" target="_blank" rel="noreferrer">vLLM — OpenAI-compatible server documentation</a></li>
</ul>
</section>

## What's next

Chapter 2 compares the OpenAI and Anthropic schemas side-by-side. Chapter 3 goes deeper on streaming, and Chapter 6 covers function/tool calling.
