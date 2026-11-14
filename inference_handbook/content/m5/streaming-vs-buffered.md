<div class="chapter-meta">Module 5 · Chapter 3</div>

# Streaming vs. Buffered Inference

<p class="lede">Streaming delivers tokens to the user as the model produces them — the typing effect you see in ChatGPT and Claude. It cuts perceived latency by a factor of 10 for long answers. The engineering cost is small but real: SSE plumbing, backpressure handling, and a reshaped error model. This chapter covers both patterns and when each wins.</p>

## Buffered — the simple default

The client sends a request, the server generates the whole response, and the server returns it in a single response body. Pseudocode:

```python
r = client.chat.completions.create(model="m", messages=msgs, stream=False)
print(r.choices[0].message.content)
```

Simple, cache-friendly, easy to log. Perceived latency equals full E2E time — fine for short answers, painful for long ones.

## Streaming — token-by-token delivery

Set `stream=True`. The server returns `text/event-stream` and writes partial deltas as tokens are produced. The user sees output starting after TTFT (~100–500 ms) rather than E2E (~5–30 s on a long answer).

```python
stream = client.chat.completions.create(model="m", messages=msgs, stream=True)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

## SSE protocol in one page

Server-Sent Events is an HTTP response with `Content-Type: text/event-stream`. The server writes lines of the form:

```
data: {"choices":[{"delta":{"content":"first chunk"}}]}

data: {"choices":[{"delta":{"content":" more"}}]}

data: [DONE]
```

Each event ends with a blank line. Clients read line-by-line and emit tokens incrementally. No bi-directional channel — one-way push from server to client.

## Backpressure

The server generates tokens at a fixed TPOT (say, 30 ms/token). If the client (or an intermediate proxy) can't consume that fast, the network buffer fills. Most engines tolerate this up to a limit, then stall or drop. In practice:

- Web clients consume plenty fast; rarely an issue.
- Proxies with small write buffers (some nginx configs) truncate silently.
- Mobile with poor connectivity queues locally; also fine.

Test your stream path end-to-end on a realistic slow network before shipping.

## Load-balancer gotchas

HTTP/1.1 SSE needs long-lived connections. Common breakages:

- **Idle timeout.** Many LBs kill connections at 30–60 s. If a stream outputs slowly and goes quiet, the LB cuts it. Fix: ping events or raise the timeout.
- **Buffered proxies.** Nginx default buffers responses — kills streaming. Add `proxy_buffering off` on the location block.
- **HTTP/2 vs. HTTP/1.1.** Both work for SSE but buffering defaults differ. Test on your stack.

## When buffered is still right

- **Non-interactive callers.** A nightly job that enriches 100M records doesn't care about TTFT.
- **Caching.** HTTP caches prefer buffered responses. Streaming responses are not cacheable by proxies.
- **Deterministic logging.** A buffered response is a single row in your logs. A streaming response is many rows to reassemble.
- **Structured output post-processing.** If the client must validate complete JSON before showing it, streaming is wasted.

## When streaming is right

- **Chat.** Perceived latency dominates UX.
- **Code generation.** Developers watch output appear and can cancel early.
- **Long answers.** Summaries, analyses — anything >100 tokens.
- **Interactive agents.** Each step of a plan appears as it forms.

## A tiny Node.js streaming client

```javascript
async function stream(prompt) {
  const r = await fetch("http://localhost:8000/v1/chat/completions", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      model: "m", stream: true,
      messages: [{role: "user", content: prompt}]
    })
  });
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += decoder.decode(value, {stream: true});
    const lines = buf.split("\n");
    buf = lines.pop();            // partial last line
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6);
      if (payload === "[DONE]") return;
      const j = JSON.parse(payload);
      const delta = j.choices[0].delta.content;
      if (delta) process.stdout.write(delta);
    }
  }
}
```

The real plumbing (reconnect, retries, abort on user cancel) is more work, but this is the mental model.

## Error handling during a stream

If the server fails mid-stream, you can't return a status code — the headers are already sent. Options:

- Emit a final event with `{"error":"..."}` so the client sees failure explicitly.
- Just close the connection; clients treat an unexpected close as a transient error and retry.

Neither is great. In practice both are used; be explicit about which you chose.

## Pitfalls

- **Treating streaming as "just add `stream=True`".** The server-side work is small; the client-side resilience work (reconnects, partial parse) is real.
- **Logging only the final assembled answer.** A streaming response can be truncated; log the final chunk's `finish_reason` and token count too.
- **Rendering raw tokens in the UI.** Many models emit special tokens (`<|eot|>`) that shouldn't be displayed. The server strips them; your front-end should trust that.
- **Mixing streaming with JSON-mode.** If you need valid JSON, don't display partial tokens — display a progress indicator until the stream closes, then render the full JSON.

<section class="faq">
<h2>FAQ</h2>
<h3>Does streaming increase server load?</h3>
<p>Slightly — more kernel writes, more connection overhead. Negligible in practice.</p>
<h3>Can I stream through a WebSocket instead of SSE?</h3>
<p>Some engines support it; most use SSE. SSE is simpler and has broader HTTP-middleware support.</p>
<h3>Why does my Nginx proxy buffer the stream?</h3>
<p>Default <code>proxy_buffering on</code>. Set it off for your LLM location block.</p>
<h3>Can I pause a stream mid-flight?</h3>
<p>No standard protocol for pause. For cancel: close the client connection. The engine will stop generating.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events" target="_blank" rel="noreferrer">MDN — Server-Sent Events</a></li>
<li><a href="https://platform.openai.com/docs/api-reference/streaming" target="_blank" rel="noreferrer">OpenAI — Streaming documentation</a></li>
<li><a href="https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#streaming-responses" target="_blank" rel="noreferrer">vLLM — Streaming support</a></li>
</ul>
</section>

## What's next

Chapter 4 moves to prompt engineering and context-window management; Chapter 5 covers structured outputs, which have their own streaming nuances.
