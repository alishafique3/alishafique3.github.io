<div class="chapter-meta">Module 9 · Chapter 2</div>

# Calculating Inference Cost

<p class="lede">Every architectural decision in inference eventually lands on one number: dollars per million tokens. This chapter gives you the formula, a current-market table of GPU-hour prices, a worked example, and the failure modes that make real-world costs 2–5× worse than the back-of-envelope.</p>

## The core formula

$$ \text{cost per 1M tokens} \;=\; \frac{\text{GPU-hourly}}{\text{sustained tokens/s}} \times \frac{10^6}{3600} $$

That's it. Every lever — model size, GPU choice, quantization, batch size, speculative decoding — moves either the *numerator* (GPU-hourly on fewer or smaller GPUs) or the *denominator* (sustained tokens/s per GPU).

## GPU-hour prices (late 2026 snapshot)

| Provider | GPU | &#36;/GPU-hour (on-demand) |
|---|---|---|
| Hyperscaler — Azure ND H100 v5 | H100 | &#36;12.29–&#36;15.12 |
| Hyperscaler — GCP A3 | H100 | &#36;10.98 |
| Hyperscaler — AWS p5 | H100 | &#36;6.88 |
| Specialist — Lambda Labs | H100 | &#36;2.49–&#36;3.29 |
| Specialist — RunPod Secure Cloud | H100 | &#36;2.79 |
| Community spot — RunPod | H100 | &#36;1.19–&#36;1.99 |
| Specialist — H200 | H200 | &#36;3.49–&#36;4.49 |
| Specialist — B200 | B200 | &#36;3.99–&#36;6.99 |

Hyperscalers charge 2–5× specialist-cloud rates. For steady workloads this gap is real money; for bursty workloads, hyperscalers' availability and tooling matter.

## Worked example — Llama-3 70B serving

Config: 2 × H100 TP=2, continuous batching, mean output 500 tokens.

- vLLM measured throughput at ~2,500 output tokens/s sustained across the pair.
- Lambda Labs pricing: 2 × &#36;2.99 = &#36;5.98/hr for both GPUs.
- Cost per 1M output tokens: $(5.98 / 2500) \times (10^6 / 3600) = \$0.66$ per 1M output tokens.
- Same workload on AWS p5: 2 × &#36;6.88 = &#36;13.76/hr → &#36;1.53 per 1M tokens.
- Same workload on Azure: 2 × &#36;12.29 = &#36;24.58/hr → &#36;2.73 per 1M tokens.

Compare to the OpenAI GPT-4o-mini list price at roughly &#36;0.60 per 1M output tokens — self-hosted Llama-3 70B on specialist cloud is competitive with the mini tier; on hyperscalers, it is not.

## What the back-of-envelope misses

- **Utilization.** If your GPU runs at 50 % of the measured throughput (traffic troughs, bad scheduling, cold starts), your effective cost doubles. Real production utilization is typically 40–70 %.
- **Prompt vs. output pricing.** Prefill compute is roughly $s/B$ times decode cost per token — long-prompt workloads shift cost toward prefill. Most price lists charge input tokens at a lower rate than output; mirror that in your own cost model.
- **Ingress/egress + storage.** Pulling a 140 GB model into a fresh pod costs minutes and a few dollars; across hundreds of cold starts per day, it adds up.
- **Platform overhead.** Kubernetes nodes, load balancers, logging — typically 5–15 % on top of raw GPU rental.
- **Reserved vs. on-demand.** 1-year reservations typically cut hyperscaler GPU-hourly by 30–50 %.

## Pitfalls

- **Reporting peak throughput.** Cost calculations must use *sustained* throughput under your actual SLO. Benchmark numbers are peak.
- **Assuming on-demand prices apply at scale.** Serious workloads buy commitment discounts.
- **Forgetting data egress.** Serving 1 TB/day of output text across regions can double the bill.

<section class="faq">
<h2>FAQ</h2>
<h3>Why is Azure H100 so much more expensive than AWS?</h3>
<p>Premium pricing for availability, enterprise support SLAs, and bundled services. For cost-sensitive workloads, specialist clouds (Lambda, RunPod, Nebius) are 2–5× cheaper for the same H100.</p>
<h3>Do reserved GPUs pay off?</h3>
<p>Yes — hyperscaler 1-year reservations typically cut GPU-hour cost by 30–50 %. Worth it whenever the workload is steadier than 60 % utilization.</p>
<h3>How do input vs. output tokens price differently?</h3>
<p>Input tokens get amortized by batching efficiently (compute-bound prefill). Output tokens pay the memory-bound decode tax. Most APIs price output 3–4× higher than input — that's where the money goes.</p>
<h3>What's missing from my back-of-envelope cost?</h3>
<p>Utilization (your 60 % not 100 %), platform overhead (K8s + LB + logging), egress (bandwidth out of the cloud), and model storage (TB-scale model weights at cold-start). Budget 1.2–1.5× the raw GPU-hourly math.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://www.cloudzero.com/blog/h100-gpu-cost/" target="_blank" rel="noreferrer">CloudZero — H100 cost: buy, rent, and cloud pricing compared</a></li>
<li><a href="https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison" target="_blank" rel="noreferrer">IntuitionLabs — H100 rental prices across 15+ providers</a></li>
<li><a href="https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/" target="_blank" rel="noreferrer">Spheron — GPU Cloud Pricing Comparison 2026</a></li>
</ul>
</section>

## What's next

Cost in dollars tracks closely with cost in joules — see [Chapter 4 — Energy and Tokens-per-Joule](?module=m9&chapter=energy-tokens-per-joule) for the companion metric, and the [Memory Calculator](https://alishafique3.github.io/memory-calculator/) to iterate configurations.
