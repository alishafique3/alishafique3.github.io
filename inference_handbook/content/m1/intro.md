<div class="chapter-meta">Module 1 · Chapter 1</div>

# Introduction to LLM Inference Engineering

<p class="lede">LLM inference engineering is the discipline of making a trained large language model answer real users under a budget — a latency budget, a dollar budget, and an energy budget. The research question is "<em>can</em> the model do this task?" The engineering question is "<em>at what cost, how fast, and for how many users?</em>" This handbook is written for that second question.</p>

## What this handbook covers

Nine modules walk from first principles to production operations:

1. **Fundamentals** — how a request flows from prompt to tokens.
2. **GPU fundamentals** — the hardware your model lives on.
3. **Planning your deployment** — choosing the model, GPU, framework, and footprint.
4. **Model optimization** — quantization, distillation, fine-tuning for serving.
5. **Model interaction** — APIs, streaming, function calling, agents.
6. **Inference infrastructure optimization** — batching, KV cache, parallelism, disaggregation.
7. **GPU kernel optimization** — the matmul and attention primitives underneath.
8. **Production operations** — Kubernetes, observability, load testing, SLOs.
9. **Cost and energy** — &#36;/token, J/token, and how to tune both without violating latency.

## Who it is for

Infrastructure engineers, ML platform engineers, and senior software engineers who are asked to serve an LLM reliably at some scale. Familiarity with Python, basic deep learning vocabulary (model, weights, tokens), and Linux/Docker/Kubernetes helps. No PhD required.

## The three axes you will always be balancing

Every decision in production inference lives on three axes that fight each other:

- **Latency** — Time to first token (TTFT) and time per output token (TPOT).
- **Throughput** — Requests per second (RPS) and tokens per second (TPS) the fleet can sustain.
- **Cost** — Dollars per million tokens, and increasingly joules per token.

Nearly every optimization — bigger batch size, FP8 quantization, speculative decoding, disaggregated prefill/decode — moves you along one axis at the cost of another. A good engineer reads the trade-off before applying the trick.

## How to read this handbook

Each chapter follows the same shape: a plain-English opening, a core mental model, the math that matters, a short code snippet, a worked numeric example with real numbers for a real model, the pitfalls engineers hit, and forward pointers to related chapters. Chapters are short (a few minutes each) and self-contained — you can read linearly or jump to the one that matches the problem on your desk today.

## What this is not

- Not a training handbook — we assume someone else trained the model.
- Not an LLM applications book — we cover the serving layer, not the product.
- Not married to one framework — we discuss vLLM, SGLang, TensorRT-LLM, and llama.cpp as peers.

<section class="faq">
<h2>FAQ</h2>
<h3>Who should read this handbook?</h3>
<p>Infrastructure and ML platform engineers who are responsible for serving an LLM at some scale — from one GPU in a dev stack to a multi-region fleet.</p>
<h3>Do I need to write CUDA kernels?</h3>
<p>No. Module 7 covers kernel optimization but as a diagnostic skill, not as a job. The rest of the handbook assumes Python, Linux, Docker, and basic Kubernetes.</p>
<h3>Is this material tied to one framework?</h3>
<p>No. vLLM, SGLang, TensorRT-LLM, TGI, and llama.cpp are all treated as peers. We say when one is objectively better for a workload.</p>
<h3>How current is the data?</h3>
<p>Benchmark numbers, GPU prices, and framework feature mentions are 2026-era. The fundamentals (prefill/decode, KV cache, roofline) are stable and apply regardless of year.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://handbook.modular.com/" target="_blank" rel="noreferrer">Modular — LLM Inference Handbook</a> — structural reference for topic ordering</li>
<li><a href="https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/" target="_blank" rel="noreferrer">NVIDIA developer blog — Mastering LLM Techniques: Inference Optimization</a></li>
<li><a href="https://vllm.ai/blog/" target="_blank" rel="noreferrer">vLLM Blog</a> — serving-engine architecture and benchmarks</li>
</ul>
</section>

## Let's begin

Head to [Chapter 2 — How Large Language Models Work](?module=m1&chapter=how-llms-work) if you want to build the mental model from the model side up, or jump to [Chapter 5 — Prefill and Decode](?module=m1&chapter=prefill-and-decode) if you want the inference-engine view first.
