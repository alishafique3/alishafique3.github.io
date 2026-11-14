<div class="chapter-meta">Module 2 · Chapter 1</div>

# Where Does LLM Inference Run?

<p class="lede">An LLM is just math — any processor that can do fused multiply-adds on tensors can, in principle, run inference. In practice, 95 %+ of production LLM inference in 2026 runs on NVIDIA GPUs. This chapter lays out the hardware landscape and the honest reasons the rest of the ecosystem still exists.</p>

## The hardware landscape

| Class | Who | When it wins |
|---|---|---|
| Data-center GPU | NVIDIA (H100, H200, B200, L40S), AMD (MI300X, MI325X) | Everything serious |
| Consumer GPU | NVIDIA RTX 4090 / 5090, prosumer Macs | Dev, small models, local |
| Custom AI accelerator | Google TPU, AWS Trainium/Inferentia, Groq, Cerebras | Specific workloads, hyperscaler lock-in |
| CPU | Intel, AMD, Apple Silicon | Very small models, edge, offline |
| Mobile NPU | Apple ANE, Qualcomm Hexagon, Google Tensor | On-device, < 3 B params |

## Why GPUs dominate

Three reasons, none of them nostalgia:

1. **Memory bandwidth.** Decode is bandwidth-bound (see M1.5). H100's 3.35 TB/s HBM, H200's 4.8 TB/s, and B200's 8 TB/s are an order of magnitude ahead of DDR5 server memory (~0.3 TB/s). No matter how clever your CPU kernel, you can't feed weights fast enough.
2. **Tensor cores.** Dedicated matmul units deliver 10–30 × the FLOPs of general-purpose vector units on the same die. Prefill compute lives here.
3. **The software stack.** CUDA, cuBLAS, cuDNN, FlashAttention, vLLM, TensorRT-LLM — twenty years of accumulated ecosystem. Every new model ships with a PyTorch reference first, and PyTorch is CUDA-first.

## When CPU makes sense

- **Very small models** (< 3 B at int4) on an M-series Mac or a modern server CPU, for dev workflows or when you have no GPU.
- **Offline batch jobs** that can run overnight, where cost matters more than latency.
- **Edge devices** where a GPU is impossible for size, power, or bill of materials reasons.

[llama.cpp](https://github.com/ggerganov/llama.cpp) is the reference CPU/consumer inference engine. GGUF is its native format. At int4, a 7 B model happily runs on a MacBook Air.

## When custom AI accelerators make sense

- **TPU** — Google Cloud's inference stack (JAX / MaxText / Pathways). Strong perf/W on large models, excellent for Gemini. Software still trails CUDA in portability.
- **AWS Inferentia / Trainium** — Price/performance on AWS specifically. Neuron SDK required. Common for cost-sensitive workloads already inside AWS.
- **Groq, SambaNova, Cerebras** — Specialty silicon with extreme single-stream latency (Groq is particularly impressive here). Narrower model and feature support.
- **AMD MI300X / MI325X** — Genuine credible alternative to H100/H200, especially where HBM capacity matters (192–256 GB per GPU). ROCm has caught up but still lags CUDA in some kernels.

## The realistic decision tree

1. Need lowest latency at scale and you have budget → **NVIDIA H100/H200/B200**.
2. Already in GCP, serving Gemini-class models → **TPU**.
3. Already in AWS, cost-sensitive, open-weight models → **Inferentia / Trainium**.
4. Need the memory of H200 at a lower list price → **MI300X**.
5. Running small models locally or at the edge → **Apple Silicon / Qualcomm NPU / llama.cpp on CPU**.
6. Dev / experimentation → **Consumer GPU or Colab**.

## Pitfalls

- **Buying a non-NVIDIA stack without the software team.** AMD/TPU/Inferentia all work, but kernel coverage, framework features, and third-party support lag. Plan for extra engineering if you leave CUDA.
- **Over-weighting raw TFLOPs.** Decode is memory-bound; bandwidth beats compute. H200 and MI300X both win over H100 on decode *because of memory*, not FLOPs.
- **Picking hardware before picking a model.** Model choice constrains hardware more than the other way around.

<section class="faq">
<h2>FAQ</h2>
<h3>Can I serve a 70B model on CPU?</h3>
<p>Yes, with llama.cpp at int4 and 64+ GB RAM. TPOT will be 300–1000 ms/token — fine for offline, painful for chat.</p>
<h3>Is TPU cheaper than GPU for LLM inference?</h3>
<p>On GCP serving Gemini-family models, often yes. For open-weight Llama/Qwen inference, GPUs usually win on both cost and tooling.</p>
<h3>Why don't more people use AMD MI300X?</h3>
<p>Great hardware — 192 GB HBM at a lower list price than H200. Software (ROCm, kernels, third-party support) is close but not yet at CUDA parity. Pick it if your team can invest in the stack.</p>
<h3>What about on-device LLMs?</h3>
<p>Apple Silicon + MLX, Qualcomm Hexagon, and Google Tensor can serve 1–3B models on-device at usable latency. Larger than that still needs cloud or desktop GPU.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://github.com/ggerganov/llama.cpp" target="_blank" rel="noreferrer">llama.cpp — reference CPU / consumer inference engine</a></li>
<li><a href="https://cloud.google.com/tpu" target="_blank" rel="noreferrer">Google Cloud TPU overview</a></li>
<li><a href="https://www.amd.com/en/products/accelerators/instinct/mi300.html" target="_blank" rel="noreferrer">AMD Instinct MI300 series</a></li>
<li><a href="https://groq.com/" target="_blank" rel="noreferrer">Groq LPU — specialty latency silicon</a></li>
</ul>
</section>

## What's next

The rest of Module 2 looks inside a GPU — [Chapter 2](?module=m2&chapter=gpu-architecture) on the architecture, [Chapter 3](?module=m2&chapter=compute-memory-hierarchy) on memory hierarchy, and [Chapter 7](?module=m2&chapter=performance-bottlenecks) on the roofline analysis you will use every day.
