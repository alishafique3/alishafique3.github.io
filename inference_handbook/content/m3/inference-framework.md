<div class="chapter-meta">Module 3 · Chapter 5</div>

# Choosing the Right Inference Framework

<p class="lede">Five frameworks dominate 2026 LLM serving: <strong>vLLM</strong>, <strong>SGLang</strong>, <strong>TensorRT-LLM</strong>, <strong>Hugging Face TGI</strong>, and <strong>llama.cpp</strong>. Each is excellent, and each is optimized for a different corner of the workload space. Picking well before you invest in a stack saves weeks of rework.</p>

## The five candidates

| Framework | Strength | Pick when |
|---|---|---|
| vLLM | Best-in-class open-source throughput, huge community, broad model support | Default for most self-hosted serving |
| SGLang | Shared-prefix / RAG / chat workloads, structured outputs | Chatbots, RAG, agent loops |
| TensorRT-LLM | Lowest TTFT on NVIDIA when engines pre-compiled, FP8/FP4-first | High-end NVIDIA fleets where every ms counts |
| TGI | Deep Hugging Face integration, Prometheus metrics out of the box | Teams already on the HF ecosystem |
| llama.cpp | CPU + consumer GPU + Apple Silicon, GGUF format | Edge, dev, offline, small models |

## What they share in 2026

The feature set has converged. All five server-side engines now implement:

- Continuous batching
- PagedAttention (or equivalent block-based KV)
- FP8 on Hopper+ hardware
- FlashAttention / FlashAttention-3
- Speculative decoding (EAGLE / Medusa / n-gram)
- OpenAI-compatible API endpoints
- Prefix caching

So the framework decision is rarely about "does it have feature X" and almost always about *which workload is the engine's default optimization target*.

## The honest trade-offs

- **vLLM** ships new models fastest and has the largest ecosystem. Day-one Llama 3.3 and Qwen 2.5 support. Throughput tuned for mixed workloads. Starts in ~60 s.
- **SGLang** has RadixAttention — extended prefix caching that caches at a token level with a radix tree. For workloads with heavy shared context (chat with long system prompts, RAG, agents), SGLang can beat vLLM by 20–30% on throughput. Weaker on fresh / unique-prompt workloads.
- **TensorRT-LLM** can lead on throughput and TTFT on Hopper when the engine is pre-compiled. Downsides: compile step takes 10–30 minutes per model + hardware combo, binary not portable, iteration is slow. Pick it when you have one model, one hardware SKU, and a release train measured in weeks.
- **TGI** — the most "operator-friendly" option. Great Prometheus/Grafana integration, Hugging Face endpoints, Inference Endpoints hosted service. Throughput slightly behind vLLM/SGLang in late-2026 benchmarks (~4,400 tok/s vs. 6,200 on an H100 at Llama 3.3 70B FP8).
- **llama.cpp** — CPU/GGUF/metal/consumer-GPU champion. Powers Ollama and LM Studio under the hood. Lacks continuous batching and multi-user serving story; perfect for single-user or edge.

## A decision tree

1. Serving on CPU, Apple Silicon, or consumer GPU → **llama.cpp**.
2. Already on Hugging Face Inference Endpoints or deep HF stack → **TGI**.
3. Workload is heavy shared-context (RAG, chat with long system prompts, agents) → **SGLang**.
4. NVIDIA-only, one or two model SKUs, latency is paramount, team has compile discipline → **TensorRT-LLM**.
5. Everything else (which is most of the time) → **vLLM**.

## Launching the leading server

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --port 8000
```

The same launch on SGLang is a `python -m sglang.launch_server …`. For TensorRT-LLM you compile first with `trtllm-build`, then serve with Triton. Full walkthrough in [Chapter 6](?module=m3&chapter=serving-first-llm).

## Pitfalls

- **Picking by last week's benchmark.** All five ship releases monthly. The ranking changes.
- **TensorRT-LLM in a fast-moving stack.** The compile-per-change cost burns your iteration speed. Only lock in TRT-LLM once you've frozen the model and hardware.
- **Assuming vLLM will beat SGLang on RAG.** It will not; RadixAttention is the right tool there.
- **Mixing engines across environments.** Dev on llama.cpp, prod on vLLM — sampling quirks and tokenization edge-cases differ, and your QA signals stop transferring.

<section class="faq">
<h2>FAQ</h2>
<h3>Is vLLM really always a safe default?</h3>
<p>For most self-hosted open-weight serving in 2026, yes. If your workload has an obvious shape (big shared prefix, tiny model on CPU), the specialized option wins.</p>
<h3>What about Ollama, LM Studio, Oobabooga?</h3>
<p>Great for dev and desktop. Not production serving — they wrap llama.cpp without multi-user scheduling.</p>
<h3>Does framework choice affect model quality?</h3>
<p>Should not, but can. Sampling implementation differences and FP8 kernel differences can shift outputs by ~1 % on quality benchmarks. Lock the framework before QA.</p>
<h3>What is LMDeploy?</h3>
<p>Another strong option from InternLM. Benchmarks competitive with vLLM. Smaller community in the West.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://github.com/vllm-project/vllm" target="_blank" rel="noreferrer">vLLM — GitHub</a></li>
<li><a href="https://github.com/sgl-project/sglang" target="_blank" rel="noreferrer">SGLang — GitHub</a></li>
<li><a href="https://github.com/NVIDIA/TensorRT-LLM" target="_blank" rel="noreferrer">NVIDIA TensorRT-LLM — GitHub</a></li>
<li><a href="https://github.com/huggingface/text-generation-inference" target="_blank" rel="noreferrer">Hugging Face TGI — GitHub</a></li>
<li><a href="https://github.com/ggerganov/llama.cpp" target="_blank" rel="noreferrer">llama.cpp — GitHub</a></li>
<li><a href="https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/" target="_blank" rel="noreferrer">Spheron — vLLM vs TensorRT-LLM vs SGLang Benchmarks 2026</a></li>
</ul>
</section>

## What's next

The next chapter spins one up. Then the capacity math from Chapter 4 becomes real numbers in [M8 Ch. 5](?module=m8&chapter=load-testing-slo).
