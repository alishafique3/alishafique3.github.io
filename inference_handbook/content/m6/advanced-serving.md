<div class="chapter-meta">Module 6 · Chapter 7</div>

# Advanced Serving — Disaggregation, Routing, Offline Batch

<p class="lede">Three production patterns show up when single-instance serving stops scaling: <strong>disaggregated prefill/decode</strong> for latency isolation, <strong>cache-aware routing</strong> for prefix cache hit rates, and <strong>offline batch</strong> for throughput-only workloads. Each is a well-defined architecture, not just a tuning flag.</p>

## Disaggregated prefill and decode

In standard "colocated" serving, the same GPU runs both phases. Problem: a long prefill blocks running decodes. Chunked prefill (M6.2) mitigates but doesn't eliminate the contention.

**Disaggregation**: dedicate separate GPU pools to prefill and decode.

```
Request → Router → [Prefill pool] → KV transfer → [Decode pool] → Stream to client
                    (compute-bound)                 (memory-bound)
```

Prefill GPUs run flat-out on tensor cores. Decode GPUs are tuned for bandwidth (H200 wins here). KV cache transfers over NVLink or InfiniBand RDMA between the two pools.

Production implementations:

- **DistServe** (OSDI 2024) — research baseline.
- **Splitwise** (Microsoft) — powers Azure OpenAI.
- **Mooncake** (Moonshot/Kimi) — KV-centric disaggregation with a distributed KV store.
- **llm-d** — open-source Kubernetes-native implementation.
- **vLLM 1.0+** — ships NixlConnector for disaggregated setups.

### When disaggregation pays off

- **Long prompts, short outputs.** RAG, code review, summarization. Prefill dominates; dedicating it isolates it.
- **High concurrency.** When colocated serving suffers prefill-decode contention.
- **Mixed hardware.** Use compute-heavy H100s for prefill, bandwidth-heavy H200s for decode.

Reported gains: 2–7× throughput at the same SLO vs. colocated serving on prompt-heavy workloads.

### When it doesn't

- Short prompts (chat): colocated is simpler and usually faster.
- Small deployments (&lt; 8 GPUs): disaggregation's complexity doesn't amortize.

## Inference routing

Once you have multiple replicas, which one should take a request? Four policies:

- **Round-robin** — simplest. Ignores state.
- **Least-connections** — route to the replica with fewest in-flight requests. Works for homogeneous workloads.
- **KV-aware** — route a continuation request to the replica that already has its KV warm.
- **Prefix-aware** — route a request with a shared system prompt to the replica whose prefix cache has it.

### KV-aware routing

Chat sessions span multiple turns. If turn 2 goes to a different replica than turn 1, that replica must re-prefill the whole history. Routing turn 2 back to the same replica keeps the prefix cache hit; TTFT stays low.

Implemented in SGLang, llm-d, and a few commercial gateways (Portkey, LiteLLM).

### Prefix-aware routing

Multiple users with the same system prompt → route them to the same replica's prefix cache. The gateway hashes the system prompt and picks a replica deterministically.

Both patterns need a stateful gateway. Round-robin LBs defeat them.

## Offline batch inference

For non-interactive workloads (nightly enrichment, dataset generation, log processing), you don't need streaming or low TTFT. You want maximum TPS at minimum cost.

Pattern:

- Pack prompts into very large batches (256–1024).
- Disable prefix caching (cold distinct prompts).
- Enable FP8 and INT4 to the max.
- No SLO — let the engine run flat-out.
- Use spot GPUs (M3.7).

Reported throughput: 5–10× higher TPS than SLO-constrained serving on the same hardware.

Libraries built for this:

- **vLLM batch API** — offline-mode entrypoint.
- **LitGPT batch** — thin wrapper.
- **Anyscale Ray Serve** — orchestration layer.

## A production fleet that uses all three

- Interactive tier: 20 colocated 2 × H100 replicas, round-robin LB. Chat.
- Agent tier: 8 disaggregated prefill (H100 × 4) + 4 decode (H200 × 2), with KV-aware routing. Long contexts, multi-step.
- Batch tier: spot H100s running offline mode on a schedule. Nightly dataset generation.

Each tier is sized to its workload shape. Mixing one tier's traffic into another is where scale breaks.

## Pitfalls

- **Enabling disaggregation at small scale.** The orchestration complexity (two pools, KV transfer, split metrics) is not worth it below ~16 GPUs.
- **Running round-robin with prefix caching enabled.** Cache hit rate drops to near zero.
- **Treating "offline" as "no latency budget".** You still have a wall clock on the batch finishing. Monitor TPS, not TTFT.
- **Mixing batch and interactive on the same engine.** The batch starves the interactive traffic. Dedicate replicas per tier.

<section class="faq">
<h2>FAQ</h2>
<h3>Is disaggregation the default in vLLM?</h3>
<p>No — vLLM 1.0+ includes the connector but you must configure it. Default is colocated.</p>
<h3>What about KV cache sharing across disaggregated instances?</h3>
<p>Yes, that's the Mooncake pattern — a cluster-wide KV store. Complex to operate; big wins when it fits.</p>
<h3>Is KV-aware routing worth it for stateless requests?</h3>
<p>No. Most benefit for multi-turn chat and agents.</p>
<h3>Can I use offline batch patterns in real-time?</h3>
<p>Yes with careful latency budgeting; the techniques overlap. Just don't co-mingle production chat traffic.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin" target="_blank" rel="noreferrer">Zhong et al. — DistServe (OSDI 2024)</a></li>
<li><a href="https://arxiv.org/abs/2311.18677" target="_blank" rel="noreferrer">Patel et al. — Splitwise (Microsoft)</a></li>
<li><a href="https://arxiv.org/abs/2407.00079" target="_blank" rel="noreferrer">Qin et al. — Mooncake KV Store</a></li>
<li><a href="https://llm-d.ai/" target="_blank" rel="noreferrer">llm-d — Open-source disaggregated serving on Kubernetes</a></li>
</ul>
</section>

## What's next

Module 6 is done. [Module 7 — GPU Kernel Optimization](?module=m7&chapter=intro-kernel) looks inside the model executor at the kernels that do the actual math.
