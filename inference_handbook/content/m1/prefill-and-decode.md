<div class="chapter-meta">Module 1 · Chapter 5</div>

# Anatomy of LLM Inference: Prefill and Decode

<p class="lede">A single LLM request has two phases with opposite performance profiles. <strong>Prefill</strong> processes the entire prompt in one parallel pass — it's compute-bound. <strong>Decode</strong> generates output tokens one at a time, each pass reading the whole model and the growing KV cache — it's memory-bound. Almost every serving optimization is a response to this split.</p>

## What happens in each phase

**Prefill** consumes a prompt of length $s$ and produces one output: the hidden state (and KV cache) for every prompt position, plus the first output token. The attention kernel sees an $s \times s$ score matrix, and compute per layer scales as $O(s^2 d)$. Because all tokens are known in advance, the GEMM is tall and GPU-friendly — on an H100 at reasonable prompt length, prefill saturates tensor cores.

**Decode** consumes one query token and the KV cache of length $n$ (the running output position), and produces one new token. The attention kernel is a vector-matrix product. Compute per layer is $O(n d)$ — tiny. But the GPU must still read all of the model's weights (~140 GB for 70B at BF16) plus the growing KV cache from HBM. The GPU is starved for data.

## The latency metrics you measure

- **TTFT (time to first token)** — dominated by prefill. If your prompt is 4k tokens, prefill time is where TTFT lives.
- **TPOT (time per output token)** — dominated by decode. If you output 500 tokens, TPOT × 500 is most of the response time.
- **E2E latency** = TTFT + TPOT × (output_tokens − 1).

## Why prefill is compute-bound and decode is memory-bound

The ratio of compute to bytes moved — **arithmetic intensity** — tells you where you sit on the GPU roofline. For prefill at length $s$ with weights of size $W$:

$$ \text{FLOPs} \approx 2 W s, \qquad \text{bytes} \approx W, \qquad \text{AI} \approx s $$

For decode at batch size $B$ (one new token per sequence):

$$ \text{FLOPs} \approx 2 W B, \qquad \text{bytes} \approx W, \qquad \text{AI} \approx B $$

An H100 has roughly 1,979 BF16 TFLOPs and 3.35 TB/s HBM3 bandwidth — a *ridge point* near 590 FLOPs/byte. Prefill at $s \gtrsim 600$ clears the ridge and the GPU runs flat-out doing math. Decode at bs = 1 sits far below and the GPU is idle waiting for HBM. Only large batches push decode toward the ridge, which is exactly why continuous batching exists (see [M6 Ch. 2](?module=m6&chapter=batching-scheduling)).

## A worked example — Llama-3 70B on H100

Prompt = 1,024 tokens, output = 256 tokens, dtype = BF16.

- Weights: 70 B × 2 B = 140 GB. Does not fit on one H100 (80 GB) — you'll need TP=2 at minimum.
- Prefill compute: 2 × 70B × 1024 ≈ 143 TFLOPs. At 50% of peak on 2×H100 (BF16), that's ≈ 72 ms — a realistic TTFT floor.
- Decode per step (bs = 1): 2 × 70B × 1 = 140 GFLOPs, but 140 GB of weight traffic. At 3.35 TB/s per H100 and TP=2 reading in parallel, that's 140/6.7 ≈ 21 ms per token — a realistic TPOT floor at low batch.

Doubling the batch to 4 drops cost per token nearly linearly (same weight read, 4× tokens) — the single biggest lever for decode throughput.

## Pitfalls

- **Measuring E2E instead of splitting.** If TTFT and TPOT aren't split, you can't tell whether your bottleneck is prefill (prompts too long, batch-prefill contention) or decode (batch too small, HBM-bound).
- **Optimizing prefill when decode dominates.** For a chatbot answering with 500 tokens, decode is ~90% of wall time. FlashAttention for prefill will not save you.
- **Assuming one phase's fix helps the other.** Quantization reduces decode bytes (big win) but often doesn't speed up prefill (it's already compute-bound; the kernel may even be slower at low precision on current silicon).

<section class="faq">
<h2>FAQ</h2>
<h3>Can prefill and decode run on different GPUs?</h3>
<p>Yes — this is "disaggregated prefill/decode" (DistServe, Splitwise, Mooncake). Covered in <a href="?module=m6&chapter=advanced-serving">M6 Ch. 7</a>.</p>
<h3>Why does TPOT get worse as context grows?</h3>
<p>Each decode step reads a longer KV cache. For a fixed model, TPOT scales approximately linearly with current output position.</p>
<h3>Is prefill always a minority of wall time?</h3>
<p>No. For prompt-heavy workloads (RAG, long summarization, code review), prefill can dominate end-to-end latency.</p>
<h3>Does batching speed up prefill?</h3>
<p>Not much — prefill is already compute-bound. Batching mostly helps decode. Chunked prefill exists to prevent large prefills from blocking decode steps.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin" target="_blank" rel="noreferrer">Zhong et al. — DistServe: Disaggregating Prefill and Decoding (OSDI 2024)</a></li>
<li><a href="https://arxiv.org/abs/2311.18677" target="_blank" rel="noreferrer">Patel et al. — Splitwise (Microsoft, 2024)</a></li>
<li><a href="https://vllm.ai/blog/2025-09-05-anatomy-of-vllm" target="_blank" rel="noreferrer">vLLM Blog — Inside vLLM: Anatomy of a High-Throughput Inference System</a></li>
</ul>
</section>

## What's next

Head to [Chapter 7](?module=m1&chapter=performance-metrics) for precise definitions of the latency and throughput metrics, then [Module 6 Chapter 2](?module=m6&chapter=batching-scheduling) for how continuous batching exploits this split.
