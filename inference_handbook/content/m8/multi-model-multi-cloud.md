<div class="chapter-meta">Module 8 · Chapter 6</div>

# Multi-Model, Multi-Cloud, and Cross-Region Inference

<p class="lede">Most production stacks eventually serve more than one model, in more than one region, often across more than one cloud. This chapter is about the orchestration patterns — <strong>model routing</strong>, <strong>cascades</strong>, <strong>regional placement</strong>, and <strong>failover</strong> — that make that complexity operable.</p>

## Why multiple models

- **Different workloads want different models.** A short-answer chat endpoint doesn't need Llama 3 70B; a complex reasoning endpoint does.
- **Cost-quality trade.** Cheap small model handles 80 % of requests; big model takes the hard ones.
- **Fine-tunes per tenant.** Enterprise customers each get their own LoRA (M4.5).
- **A/B testing.** Shipping a new model safely requires running old + new side by side.

## Model routing — the dispatcher

A router inside your gateway decides which model gets each request. Signals to route on:

- **Workload class** — detected from prompt length, system prompt, or explicit `model` field.
- **Tenant** — user or customer-ID based.
- **Quality tier** — paid vs. free, premium vs. standard.
- **Current load** — cheap model if the expensive one is saturated.

Open implementations: LiteLLM, Portkey, OpenRouter. vLLM's router (`vllm serve --router`) handles the simple case.

## Model cascades — ask cheap first

A cascade pattern: route first to a cheap model; if its confidence is low or an uncertainty signal fires, escalate to a bigger model.

```python
def cascade(prompt):
    small = call_model("qwen-2.5-7b", prompt)
    if is_confident(small):             # a cheap heuristic
        return small
    big = call_model("llama-3-70b", prompt)
    return big
```

Reported real-world wins: 3–5× cost reduction while hitting the same quality bar on most workloads.

The hard part is the confidence signal. Common patterns:

- **Logprob threshold** — low-confidence outputs are often the ones the small model gets wrong. Needs log-probs exposed (vLLM supports).
- **LLM-as-judge** — a small judge model scores the answer; low scores escalate.
- **Tool-use trigger** — if the small model wants to invoke tools, escalate.
- **Length-based** — short answers stay on small; long/complex escalate.

## Regional placement

For global users, latency matters. Round-trip from Europe to a US-only service is 100+ ms before any inference.

Two patterns:

- **Replicate per region.** Full serving stack in every region. Expensive but simple.
- **Route to nearest region.** Latency-aware DNS or anycast puts users on the closest region.

Model storage: use a global object store (S3 with cross-region replication, GCS multi-region) so pods in any region can pull weights.

## Failover

Single region can fail — AZ outage, network partition, bad deploy. Design for it:

- **Health checks per region.** Gateway stops routing to unhealthy regions.
- **Warm standby in secondary region.** At least one replica running; traffic shifts automatically.
- **Capacity planning assumes N-1 regions.** You should still meet SLO with one region down.

For LLM services, cross-region failover doesn't preserve conversation state (KV cache is local). Short-term quality drop during failover is unavoidable but acceptable.

## Multi-cloud

Three reasons to go multi-cloud:

1. **Negotiating leverage.** Vendor lock-in is expensive.
2. **GPU availability.** A single provider can't always give you the capacity you need.
3. **Compliance.** Some workloads want "not in any one cloud."

Multi-cloud overhead:

- Each cloud's networking, IAM, LB is different.
- Costs can be hard to compare apples-to-apples.
- Expertise doubles.

Practical pattern: one primary cloud, one burst cloud for peak capacity. Not *fully* multi-cloud, but gets 80 % of the benefit at 20 % of the ops cost.

## Multi-tenant isolation

Serving multiple customers (or multiple fine-tunes per customer) on shared infrastructure needs explicit isolation:

- **Logical isolation** — different API keys route to different model deployments. Cheapest.
- **Namespace isolation** — one K8s namespace per tenant. More overhead, cleaner separation.
- **Physical isolation** — dedicated GPU pools per tenant. Expensive; required for strictly regulated workloads.

LoRA hot-swap (M4.5) is the economical pattern for per-tenant fine-tunes: same base model, many adapters, routed per request.

## A realistic multi-model production fleet

A SaaS company serving enterprise customers:

- **Small / cheap tier**: Qwen 2.5 7B FP8, one pool, round-robin, 4 replicas. Handles 70 % of traffic.
- **Large tier**: Llama 3.1 70B FP8, one pool, KV-aware routing, 6 replicas. Hard reasoning + long context.
- **Enterprise fine-tune tier**: Llama 3.1 70B base + 100 LoRA adapters, hot-swap serving, 4 replicas. Per-customer specialization.
- **Cascade router** at the gateway: route to small first, escalate on low-logprob or long-answer need.
- **Regions**: US-East, EU-West, APAC-SE. Each has its own small+large pool. Enterprise fine-tune tier only in US-East (data residency).
- **Failover**: inter-region DNS failover for small+large tiers. No failover for enterprise (customer requirement to stay in region).

Nine tiers, three regions, two clouds. A real production topology.

## Pitfalls

- **Routing by model name without smart fallback.** If the expected model is down, route somewhere, even if degraded. Hard failure is worse than graceful degradation.
- **Cross-region without latency awareness.** Serving US users from EU replicas adds 100+ ms per request. Measure.
- **Multi-cloud without automation.** Manual cloud switches at 3 AM don't work. Automate from day one.
- **No cost tagging.** Multi-tenant + multi-region without per-tier cost visibility becomes a budget nightmare.

<section class="faq">
<h2>FAQ</h2>
<h3>Do I need a router if I have one model?</h3>
<p>No — but you probably will. Even single-model services end up with a draft + target pair for speculative decoding, or dev vs. prod versions.</p>
<h3>How much does a cascade save in practice?</h3>
<p>3–5× cost reduction on real workloads is common. Depends heavily on the cheap-tier model and the confidence signal.</p>
<h3>Can I fail over mid-request?</h3>
<p>Streaming responses — no, the connection is open. For a new request in a failed region — yes.</p>
<h3>Does multi-tenant routing hurt performance?</h3>
<p>Routing itself: negligible. Loss of prefix caching from mixing tenants on one replica: significant. Prefer tenant-affine routing.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://github.com/BerriAI/litellm" target="_blank" rel="noreferrer">LiteLLM — Router for 100+ LLM providers</a></li>
<li><a href="https://portkey.ai/" target="_blank" rel="noreferrer">Portkey — LLM gateway with cascades</a></li>
<li><a href="https://openrouter.ai/" target="_blank" rel="noreferrer">OpenRouter — Public multi-model gateway</a></li>
</ul>
</section>

## What's next

The last chapter — [InferenceOps](?module=m8&chapter=inference-ops) — covers the lifecycle: rollout, rollback, canary, incident response.
