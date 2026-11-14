<div class="chapter-meta">Module 9 · Chapter 7</div>

# Designing a Cost- and Energy-Efficient Inference Stack

<p class="lede">The capstone. Every module in this handbook has been a piece. This chapter is how the pieces fit into one defensible production design — the kind you'd present to a review board. We'll walk through a complete reference design for a mid-scale LLM product, with every decision justified by a chapter.</p>

## The scenario

A SaaS company launching an AI assistant for enterprise customers:

- **Target scale**: 100k monthly active users, 50M requests/month at steady state, 2× peak-to-average.
- **Workload shape**: 60 % short chat, 30 % medium QA, 10 % long summarization.
- **SLO**: TTFT P99 ≤ 600 ms, TPOT P99 ≤ 50 ms.
- **Budget**: &#36;25k/month for inference.
- **Compliance**: data stays in-region; SOC 2.

## The design, bottom to top

### Layer 0 — Hardware (Modules 2, 3)

- **GPU**: H200 (chosen over H100 for memory headroom; 140 GB fits 70B unquantized with KV room for high concurrency).
- **Node**: 8 × H200 per node, NVLink Switch.
- **Fabric**: InfiniBand NDR between nodes.
- **Hosting**: Specialist cloud (Lambda / Nebius) at ~&#36;3.50/GPU-hr with 1-year reservation.

**Why not**: hyperscaler (2–3 × more expensive, same hardware). On-prem (4-month procurement, no team).

### Layer 1 — Model (Modules 3, 4)

- **Base model**: Llama 3.1 70B Instruct (strong quality, mature ecosystem, Meta license).
- **Quantization**: FP8 via LLM Compressor (compressed-tensors format).
- **Fine-tuning**: LoRA per enterprise tenant via hot-swap.
- **Draft model for speculative decoding**: Llama 3.1 8B FP8.

**Why not**: Qwen 2.5 72B (equally good; Llama's drafter story is slightly more mature). INT4 (tighter margins; FP8 is the sweet spot on Hopper).

### Layer 2 — Engine (Modules 5, 6, 7)

- **Framework**: vLLM.
- **Config flags**:
  - `--tensor-parallel-size=2` (per replica; 2 × H200 per replica).
  - `--enable-prefix-caching` (chat workload has big system prompt reuse).
  - `--speculative-model=llama-3.1-8b --num-speculative-tokens=5`.
  - `--kv-cache-dtype=fp8`.
  - `--max-model-len=16384` (cover long summarization P99).
  - `--enable-lora --max-lora-rank=32` (enterprise tenant adapters).

**Why not**: SGLang (strong RAG option but team already knows vLLM). TensorRT-LLM (compile-chain cost too high for monthly model iteration).

### Layer 3 — Serving platform (Module 8)

- **Orchestration**: Kubernetes on specialist-cloud managed cluster.
- **Gateway**: LiteLLM — handles auth, rate limiting, OpenAI translation, model routing.
- **Routing**: KV-aware (session affinity) via LiteLLM.
- **Autoscaling**: KEDA on `vllm:num_requests_waiting`, min 4 replicas, max 20.
- **Observability**: Prometheus + Grafana + DCGM Exporter + OpenTelemetry → Tempo.
- **CI/CD**: ArgoCD + GitHub Actions.

### Layer 4 — Workload tiers (Module 8.6, 9.6)

- **Tier 1 — Quick chat**: Llama 3.1 8B FP8, 2 replicas. Handles short-reply cases via cascade from Tier 2.
- **Tier 2 — Main chat**: Llama 3.1 70B FP8, 4–20 autoscaled replicas. Default tier.
- **Tier 3 — Enterprise**: Llama 3.1 70B + per-tenant LoRA, 2 replicas. Isolated by namespace.
- **Tier 4 — Batch** (nightly enrichment): spot H100s, run during low-carbon grid hours.

Cascade routes most traffic to Tier 1; escalates on low-confidence or long-output signal.

## The capacity math (Module 3.4)

Peak = 2 × avg = 100M / (30 × 24 × 3600) × 2 ≈ 77 req/sec.

Per-replica goodput (measured in staging): 10 RPS on 2 × H200 at the standard SLO.

Replicas needed: ceil(77 × 1.4 safety / 10) = 11.

Add tiers:

- Tier 1: 2 replicas (quick cascade handling).
- Tier 2: 11 autoscaled.
- Tier 3: 2 reserved.
- Tier 4: spot, ephemeral.

Total steady: 15 × 2 × H200 = 30 GPUs. At &#36;3.50/hr reserved = &#36;75,600/month.

That's over the &#36;25k budget. Four moves to close the gap:

1. **Cascade more aggressively**: push 70 % of traffic to Tier 1. Tier 2 replicas drop to 5.
2. **Specular + FP8**: measured 1.6 × effective throughput. Tier 2 replicas drop to 3.
3. **Off-peak scale-down**: autoscaler aggressive scale-down after midnight. Average active replicas 60 % of peak.
4. **Prefix caching**: 60 % cache hit rate on chat workload cuts effective prefill cost by ~50 %.

Revised total: ~12 GPU-replicas average (2 × H200 each) × &#36;3.50 × 730 hr = &#36;61k/mo for compute. Still 2.4 × the budget.

Final move: **drop to 2 × H100 FP8 for Tier 2 instead of 2 × H200**. &#36;2.80/GPU-hr reserved. Same throughput for this workload. Budget: ~&#36;49k/month.

Still above &#36;25k. At this point you either (a) raise the budget, (b) use a cheaper model, or (c) accept lower quality. Negotiate with product.

**Lesson**: cost math is an honesty check. Discovering a mismatch early is valuable.

## The energy accounting (Module 9.4, 9.5)

At the revised 12-replica average, each 2 × H100 drawing ~1,100 W:

- Energy: 12 × 1,100 W × 730 hr = 9,636 kWh/month.
- At &#36;0.10/kWh (included in GPU-hr price): matches math.
- CO₂: at US average 400 gCO₂/kWh = 3.85 tonnes/month. Per-request: ~0.077 g.

Tokens-per-joule target (M9.4): 2–4 tokens/J for Llama 3.1 70B FP8 at production batch. Our fleet is in that range.

## What makes this design defensible

- **Every decision maps to a chapter.** Reviewers can trace "why 2 × H200 instead of 1 × B200" to M2.4 and M3.3.
- **It's measured, not guessed.** Capacity, goodput, J/token — all from staged load tests and pynvml.
- **It has fallbacks.** Cascade degradation, failover to Tier 1, spot fallback to on-demand.
- **It's monitored end-to-end.** Prometheus + DCGM + OTel.
- **It's honest.** When the budget didn't close, we iterated the design, not the math.

## What's missing from this handbook

- **Multimodal** — vision and audio add their own kernels and bandwidth story.
- **Training alongside serving** — many teams co-deploy. Outside scope.
- **Reinforcement learning from serving traffic** — RLHF pipelines are a separate book.
- **Agent-specific infrastructure** — long-trajectory workloads are a research frontier.

Future revisions will cover these.

## Pitfalls (final list)

- **Design that ignores capacity math.** You'll ship, underprovision, and firefight in week one.
- **Engineering that outpaces observability.** You can't optimize what you don't measure. Build dashboards first.
- **Cost framing without SLO.** Every cost cut has a quality or latency tax. Price them together.
- **Perfectionism before launch.** Ship a 60 % design and iterate. A perfect design that doesn't ship is zero.

## Closing

Every module in this handbook has been about one piece of the puzzle: tokens, GPUs, deployment, model optimization, APIs, engines, kernels, operations, economics, energy. The point was never any single piece. The point is the whole stack.

Every engineering decision in LLM serving — every quantization, every batch-size knob, every scheduling policy — ultimately shows up as a point on the three-axis plot of latency, dollars, and joules. The engineer who can see those three axes at once and defend the chosen point is what this handbook was written to produce.

Welcome to the field. Build well.

<section class="faq">
<h2>FAQ</h2>
<h3>Is this a real company?</h3>
<p>No — a composite. But every number is realistic for late-2026 and every decision maps to patterns real teams use.</p>
<h3>What if my scale is 10 × bigger?</h3>
<p>Multi-region becomes mandatory. Disaggregated prefill/decode starts to pay off. BYOC or on-prem gets real cost advantages.</p>
<h3>What if my scale is 10 × smaller?</h3>
<p>Skip Kubernetes. Run vLLM behind a Cloudflare Worker. Use the API tier for anything frontier-class.</p>
<h3>Where should I go next?</h3>
<p>Read the <a href="https://vllm.ai/" target="_blank" rel="noreferrer">vLLM blog</a> (practitioner updates), the <a href="https://blog.pytorch.org/" target="_blank" rel="noreferrer">PyTorch blog</a> (frameworks), and <a href="https://arxiv.org/list/cs.DC/recent" target="_blank" rel="noreferrer">arXiv cs.DC</a> (research). Build a stack for a small internal tool. Measure everything. Share what you learn.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li>Every reference in Modules 1–9 of this handbook.</li>
<li><a href="https://github.com/alishafique3/LLM-infra-from-scratch" target="_blank" rel="noreferrer">LLM-infra-from-scratch — the complete repo backing this handbook</a></li>
<li><a href="https://alishafique3.github.io/model-explorer/" target="_blank" rel="noreferrer">Model Explorer</a>, <a href="https://alishafique3.github.io/memory-calculator/" target="_blank" rel="noreferrer">Memory Calculator</a>, <a href="https://alishafique3.github.io/kv-cache-calculator/" target="_blank" rel="noreferrer">KV Cache Calculator</a> — the handbook's companion tools</li>
</ul>
</section>

## That's the handbook

You've reached the end. Nine modules, sixty-one chapters, one coherent mental model of what it takes to serve LLMs at production scale. Thanks for reading.

— *Ali Shafique and Hamza Farooq*
