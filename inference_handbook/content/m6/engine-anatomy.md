<div class="chapter-meta">Module 6 · Chapter 1</div>

# Anatomy of an LLM Inference Engine

<p class="lede">Behind the single-line OpenAI endpoint is a small system — queues, schedulers, workers, memory managers, and sampling pipelines. Understanding that system makes every subsequent chapter make sense. This chapter walks through the shape every serious engine (vLLM, SGLang, TensorRT-LLM) shares.</p>

## The six components

Every modern LLM inference engine is built from the same pieces:

1. **API server** — speaks the OpenAI/Anthropic HTTP protocol.
2. **Request queue** — incoming requests wait here.
3. **Scheduler** — decides which requests to run at each iteration.
4. **Model executor** — runs the forward pass on the GPU(s).
5. **KV cache manager** — allocates and tracks KV blocks (PagedAttention).
6. **Sampler** — turns logits into tokens using the per-request sampling parameters.

Different engines differ in how they implement each piece, not in the overall shape.

## One iteration — the scheduling loop

Every ~20–50 ms (one decode step for one layer at batch), the engine runs:

```
1. Read incoming requests from the queue.
2. Scheduler decides the batch for this iteration:
   - Which running sequences get a decode step.
   - Which new requests get prefilled (possibly chunked).
   - Preempt sequences if KV cache is full.
3. KV manager allocates blocks for new tokens.
4. Model executor runs the forward pass on the batch.
5. Sampler produces one new token per sequence.
6. Update per-request output buffers; detect EOS/max_tokens.
7. Finished sequences return results to the API server.
8. Loop.
```

This is **iteration-level scheduling** — the invention that made continuous batching possible (M6.2).

## The scheduler decides latency

Scheduling policy choices:

- **FCFS (First-Come-First-Served)** — simplest. A long prompt in front of short ones blocks everyone.
- **Chunked prefill** — slice long prefills into pieces interleaved with decode (M6.2).
- **Priority** — tier based (free vs. paid users). Rarely implemented in open engines.
- **KV-aware** — prefer sequences whose KV is already warm.

vLLM uses FCFS + chunked prefill by default. SGLang adds radix-tree-aware scheduling for shared prefixes.

## The KV cache manager

This is the biggest memory on the GPU after weights. It allocates **blocks** (typically 16 tokens of KV each) and gives each sequence a **block table** (pointer list). When a sequence finishes, its blocks return to the free pool. When memory is tight, the scheduler can **preempt** a sequence — return its blocks and re-prefill on the next iteration.

PagedAttention (M6.3) is the killer feature here. It is the reason vLLM throughput is 2–4 × classic serving.

## The model executor

Thin layer over the model forward pass. Responsibilities:

- Gather KV from the right blocks (requires the custom attention kernel).
- Run attention + FFN + norms.
- Produce logits for the next-token position of each sequence in the batch.
- Return to the sampler.

On multi-GPU, this is where tensor parallelism lives — the executor orchestrates AllReduces across GPUs (M6.6). FlashAttention is called from inside this component.

## The sampler

For each sequence in the batch, applies the per-request sampling parameters (temperature, top-p, min-p, repetition penalty) to the logits and picks a token. Modern engines fuse the sampler into GPU kernels so the overhead is small (&lt; 5 % of decode time).

## An end-to-end request trace

```
t = 0 ms     Client POST /v1/chat/completions arrives.
t = 2 ms     API server validates, enqueues.
t = 5 ms     Scheduler admits for prefill (chunk 1).
t = 70 ms   Prefill chunk 1 done, chunk 2 queued; KV blocks allocated.
t = 140 ms  Prefill complete. First token produced by sampler.
t = 141 ms  Stream opens; first token flushed to client.    ← TTFT
t = 170 ms  Second token produced (decode step).            ← TPOT
...
t = 10,340 ms  EOS token. Stream closes with usage.         ← E2E
```

Every component on the critical path ran many times. The engine's job is to keep them all busy.

## Pitfalls

- **Treating the engine as a black box.** Capacity-planning and SLO issues trace back to specific components. Knowing which one helps you fix them.
- **Over-tuning a single component in isolation.** FlashAttention saves prefill time; if your decode is the bottleneck, you won't see it.
- **Changing scheduler policy in prod without a measurement.** Non-FCFS can hurt tail latency unexpectedly. Benchmark before switching.

<section class="faq">
<h2>FAQ</h2>
<h3>Is the scheduler part of the model executor?</h3>
<p>No — scheduler decides what to run, executor runs it. Separating them is what makes continuous batching possible.</p>
<h3>What is "preemption"?</h3>
<p>Freeing a running sequence's KV blocks under memory pressure, with intent to re-prefill it later. Costly but preserves throughput on bursty workloads.</p>
<h3>Does the engine own the HTTP server?</h3>
<p>Usually yes in dev stacks (vLLM ships Uvicorn). Production commonly puts a dedicated gateway (Envoy, Traefik) in front for auth, rate limits, and TLS.</p>
<h3>How many iterations per second?</h3>
<p>Depends on batch and model. Llama 3 70B on 2×H100 at bs=16: ~25 iterations/s. Each iteration produces 16 tokens.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://vllm.ai/blog/2025-09-05-anatomy-of-vllm" target="_blank" rel="noreferrer">vLLM Blog — Inside vLLM: Anatomy of a High-Throughput Inference System</a></li>
<li><a href="https://dl.acm.org/doi/10.1145/3600006.3613165" target="_blank" rel="noreferrer">Kwon et al. — PagedAttention (SOSP 2023)</a></li>
<li><a href="https://www.usenix.org/conference/osdi22/presentation/yu" target="_blank" rel="noreferrer">Yu et al. — Orca: Iteration-level scheduling (OSDI 2022)</a></li>
</ul>
</section>

## What's next

Each component has its own chapter. [Chapter 2](?module=m6&chapter=batching-scheduling) deep-dives the scheduler and batching; [Chapter 3](?module=m6&chapter=kv-cache-paged) deep-dives the KV manager and PagedAttention.
