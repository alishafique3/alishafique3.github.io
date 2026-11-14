<div class="chapter-meta">Module 3 · Chapter 6</div>

# Serving Your First LLM

<p class="lede">A hands-on, ten-minute walkthrough from "I have a GPU" to "my app can talk to a local LLM via the OpenAI SDK." Everything on one machine, nothing hidden. This is the baseline every production stack extends from.</p>

## Prerequisites

- A Linux (or WSL) machine with an NVIDIA GPU and 24 GB+ of VRAM (an L4, L40S, A10, or RTX 4090 all work for Qwen 2.5 0.5B–7B).
- Python 3.11, `pip`, and a working CUDA driver.
- Hugging Face account + access token if the model requires it.

## Step 1 — Install vLLM

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install vllm openai
```

vLLM ships pre-built wheels for CUDA 12. First install usually takes a minute or two.

## Step 2 — Launch the OpenAI-compatible server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --port 8000
```

First launch downloads the weights to `~/.cache/huggingface` (~15 GB for Qwen 2.5 7B). After "Starting Uvicorn… http://0.0.0.0:8000", vLLM is ready.

## Step 3 — Test with curl

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "messages": [{"role": "user", "content": "Explain KV cache in one sentence."}]
    }'
```

You should get a JSON response. If so, the server works.

## Step 4 — Enable streaming in Python

```python
from openai import OpenAI
import time

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

t0 = time.perf_counter()
ttft = None
chunks = 0
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Write a haiku about inference."}],
    max_tokens=64,
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        if ttft is None: ttft = time.perf_counter() - t0
        print(chunk.choices[0].delta.content, end="", flush=True)
        chunks += 1

total = time.perf_counter() - t0
print(f"\n\nTTFT = {ttft*1000:.0f} ms, TPOT = {(total-ttft)/max(1,chunks-1)*1000:.1f} ms")
```

You just measured the two metrics from M1.7 on your first serving stack.

## Step 5 — Swap in a quantized checkpoint

Stop the server (Ctrl-C). Relaunch with an FP8 or AWQ variant:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --max-model-len 8192 \
    --port 8000
```

Run Step 4 again. Expect lower VRAM footprint, ~1.5–2× TPOT improvement, and < 1 % quality drop on most benchmarks. The quantization story is the whole of Module 4.

## What you just built

- A standalone OpenAI-compatible endpoint on `localhost:8000`.
- Streaming token delivery via SSE.
- Continuous batching (default in vLLM).
- PagedAttention (default).
- Prefix caching (enabled with the flag).

That's a production-shaped serving stack in one process. The rest of this handbook is about scaling it.

## Pitfalls

- **Forgetting `--dtype bfloat16`.** fp16 on some older GPUs is fine; on Hopper, bf16 is faster and more numerically stable.
- **Setting `--max-model-len` too high.** It pre-reserves KV budget. 8k is a safe start; raise only when needed.
- **Running the vLLM server and heavy client load on the same machine.** Your client steals CPU from the vLLM scheduler.
- **Assuming CUDA OOM = wrong hardware.** Often a mis-sized `--gpu-memory-utilization`. Try 0.85 first.

<section class="faq">
<h2>FAQ</h2>
<h3>Does this work without Docker?</h3>
<p>Yes — vLLM runs natively as a Python process. Docker is the recommendation for prod (M8 Ch. 2) but not required to learn.</p>
<h3>Can I use an Apple Silicon Mac?</h3>
<p>Not with vLLM — use llama.cpp / Ollama / MLX instead. The API shape is similar; the engine is different.</p>
<h3>How do I expose the port to the network?</h3>
<p><code>--host 0.0.0.0</code>. Only do this on a trusted network — vLLM has no auth; put it behind a gateway in production.</p>
<h3>What if I don't have a GPU?</h3>
<p>Use Google Colab (free T4 GPU) and run the same commands in a notebook, or use llama.cpp on CPU with a smaller model.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://docs.vllm.ai/en/stable/getting_started/quickstart.html" target="_blank" rel="noreferrer">vLLM — Quickstart documentation</a></li>
<li><a href="https://huggingface.co/Qwen" target="_blank" rel="noreferrer">Qwen model family on Hugging Face</a></li>
<li><a href="https://github.com/openai/openai-python" target="_blank" rel="noreferrer">OpenAI Python SDK — reference client</a></li>
</ul>
</section>

## What's next

With a server running, the next decisions are *where* it runs — [Chapter 7 — Cloud, Hybrid, On-Prem](?module=m3&chapter=cloud-hybrid-onprem). From there Module 4 looks at compressing the model, and Module 6 at scaling the serving.
