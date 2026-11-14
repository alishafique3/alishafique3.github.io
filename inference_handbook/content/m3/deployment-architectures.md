<div class="chapter-meta">Module 3 · Chapter 1</div>

# Understanding LLM Deployment Architectures

<p class="lede">Before you pick a model, a GPU, or a framework, pick an <strong>architecture</strong> — the shape of how your LLM reaches users. Four basic shapes dominate the 2026 landscape: managed API, serverless, self-hosted cloud (BYOC), and on-prem/edge. Each has a different cost curve, latency floor, and operational burden. Getting this call right up front saves you from re-architecting at 10 × scale.</p>

## The four shapes

| Architecture | You own | Provider owns | Pay for |
|---|---|---|---|
| Managed API | The app | Model, GPU, serving | Per token |
| Serverless inference | Model + app | GPU, autoscaling, cold starts | Per second of GPU use |
| Self-hosted (cloud / BYOC) | Everything in the cluster | Rents the hardware | GPU-hour (24 × 7) |
| On-prem / edge | Everything, including the hardware | Nothing | Capex + power + staff |

## The cost-control-latency triangle

Each shape sits at a different point of a three-way trade:

- **Latency floor.** On-prem can hit the lowest wall-clock latency because there is no network to the cloud. Managed APIs add 30–100 ms of inter-region latency.
- **Operational cost.** Managed APIs are the cheapest at low volume (you pay only for what you use). Self-hosted is cheapest at high volume (idle time is free).
- **Control.** Managed = least control (closed weights, provider-dictated features). On-prem = total control (your model, your silicon, your compliance story).

## When each wins

- **Managed API** — prototyping, intermittent workloads, you want frontier-closed-model quality (GPT-4-class, Claude-class). Break-even against self-hosted is around 10–50 M tokens/day.
- **Serverless inference** — bursty traffic, you want to pay per request without managing scale. Trade-off is cold starts (5–30 s for a 70 B model).
- **Self-hosted cloud** — steady-state traffic above the break-even, you need open weights, custom fine-tunes, or control over sampling / tokenizer / guardrails.
- **On-prem** — data-residency rules, regulated industries, large steady fleets (> 100 GPUs amortizing capex).
- **Edge** — mobile/on-device for small models (< 3 B), privacy-first apps, no-network environments.

## The break-even math

Compare managed-API cost against self-hosted GPU-hour cost:

$$ \text{break-even tokens/month} = \frac{\text{GPU-hour} \times 730}{\text{API price per token}} $$

Example: a 2 × H100 node at &#36;6/hr (specialist cloud) runs ~730 × &#36;6 = &#36;4,380/month. At OpenAI's GPT-4o-mini output rate (~&#36;0.60/M tokens), break-even is ~7.3 B output tokens/month. Below that → API. Above that → self-host.

For reference-grade closed models (GPT-4-class at &#36;15–30/M tokens), break-even is ~150–300 M tokens/month — much lower, so the self-host question comes up fast.

## A decision tree

1. Need a frontier closed model and < 100 M tokens/month → managed API.
2. Need an open-weight model + custom fine-tune → self-hosted.
3. Steady traffic above break-even and have a platform team → self-hosted.
4. Bursty traffic, no platform team → serverless inference.
5. Data cannot leave your network → on-prem.
6. Model ≤ 3 B, runs on consumer device → edge.

## Pitfalls

- **Picking "serverless" to avoid ops — then burning hours on cold-start tuning.** A 70 B model's cold start is minutes. Warm pools cost real money.
- **Choosing "on-prem" to save money, forgetting the staff cost.** On-prem labor (CUDA, Kubernetes, networking) typically costs more than the GPUs over three years.
- **Mixing managed API for development and self-hosted for prod without matching samplers.** Outputs diverge; QA signals from prototyping stop transferring.
- **Assuming "cloud" means a specific latency.** Cross-region latency can be 100+ ms. Pick regions close to users.

<section class="faq">
<h2>FAQ</h2>
<h3>Is managed API always slower than self-hosted?</h3>
<p>Not necessarily. Providers have more capacity and better-optimized stacks than most self-hosts. The tax is network round-trip and queueing under load — typically 50–150 ms overhead vs. a co-located self-host.</p>
<h3>What is BYOC?</h3>
<p>Bring Your Own Cloud — self-hosting infrastructure inside your cloud account (AWS/GCP/Azure), sometimes via a managed control plane (e.g., Anyscale, Together AI BYOC). You get self-hosted economics with managed ops.</p>
<h3>How do serverless cold starts work for LLMs?</h3>
<p>The model binary (tens of GB) must load from object storage to GPU HBM. Cold-start time scales with model size and HBM bandwidth. Warm pools mitigate it at a cost.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://handbook.modular.com/getting-started/serverless-vs-self-hosted-inference/" target="_blank" rel="noreferrer">Modular — Serverless vs. Self-Hosted LLM Inference</a></li>
<li><a href="https://openai.com/api/pricing/" target="_blank" rel="noreferrer">OpenAI API Pricing</a></li>
<li><a href="https://www.anthropic.com/pricing" target="_blank" rel="noreferrer">Anthropic API Pricing</a></li>
</ul>
</section>

## What's next

Having picked the architecture, the next decision is [which LLM to serve](?module=m3&chapter=choosing-the-llm). The rest of Module 3 progresses through GPU selection, framework choice, and the hands-on serving walkthrough.
