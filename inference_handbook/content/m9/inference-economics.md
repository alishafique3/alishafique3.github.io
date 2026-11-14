<div class="chapter-meta">Module 9 · Chapter 1</div>

# Understanding LLM Inference Economics

<p class="lede">An LLM inference service has three cost drivers: the GPU hour, the utilization of that GPU hour, and the tokens you produce per hour. Every architectural choice in modules 3–8 ultimately maps to one of these three. This chapter is the economic mental model you'll apply in the rest of this module.</p>

## The three cost knobs

$$ \$/\text{token} = \frac{\$/\text{GPU-hr}}{\text{utilization}} \times \frac{1}{\text{tokens/sec per GPU}} \times \frac{1}{3600} $$

Three knobs to turn:

1. **Cheaper GPU-hours** — specialist cloud vs. hyperscaler; on-prem; reserved instances.
2. **Higher utilization** — better load balancing; fewer idle hours; smarter scaling.
3. **More tokens per GPU-sec** — better engine, better model, better precision, better batch.

Most engineering time goes into the third. The first two have bigger impact for most teams.

## Why utilization matters so much

A GPU at 100 % utilization serving steady traffic runs &#36;4,380 / month (2 × H100 at &#36;6/hr on specialist cloud). A GPU at 40 % utilization doing the same work still costs &#36;4,380 — you burn 60 % of every dollar on idle time.

Real production utilization is typically 40–70 %. The gap between measured throughput and sustained delivered throughput *is the biggest hidden cost* in most LLM serving budgets.

## Three operating points

For a Llama 3.1 70B FP8 fleet on 2 × H100 at &#36;6/hr/GPU:

| Operating point | Utilization | Effective &#36;/GPU-hr | Effective &#36;/M tokens |
|---|---|---|---|
| Peak benchmark | 100 % | &#36;6 | &#36;0.55 |
| Sustained production | 60 % | &#36;10 | &#36;0.90 |
| Bursty with headroom | 40 % | &#36;15 | &#36;1.35 |

Benchmark numbers are a lie for production economics. Multiply by 1.5–2× to get your real &#36;/M tokens.

## Where the money actually goes

Breakdown of a steady-state LLM production bill:

- **GPU compute** — 70–85 %.
- **Networking / egress** — 2–10 %. Grows with multi-region, streaming at scale.
- **Storage** — &lt; 2 %. Model weights at rest are cheap.
- **Platform overhead** — 5–15 %. K8s nodes, LB, monitoring, logging.
- **Cold-start waste** — 2–10 %. Cost of warm pools or slow-scale-down.

The lesson: focus on compute and utilization. Storage and networking are usually noise.

## When to focus on each knob

**Reduce GPU-hourly:**

- Move from hyperscaler to specialist cloud (&#36;12/hr → &#36;3/hr).
- Commit to a 1-year reservation (30–50 % discount).
- Fine-tune a smaller model (M4.5) to run on cheaper GPUs.

**Increase utilization:**

- Smarter autoscaling (M8.3).
- Shared infrastructure across multiple workloads.
- Spot GPUs for offline batch tiers.
- Reduce cold-start overhead (M8.3).

**Increase tokens/sec:**

- Quantization (M4.2).
- Better engine / newer version.
- Speculative decoding (M6.5).
- Higher batch (M6.2).
- Prefix caching (M6.3).

Most teams over-invest in the third knob and under-invest in the first two.

## The amortization trap

Easy mistake: compare specialist-cloud on-demand (&#36;3/hr) vs. hyperscaler reserved (&#36;5/hr) and conclude specialist wins. Then your traffic drops, specialist is still &#36;3/hr running on-demand, and hyperscaler reserved is still &#36;5/hr even when idle.

Reservations = discount in exchange for commitment. The commitment IS the cost — accept it only if utilization is predictably high.

## Two break-even curves

**API vs. self-hosted** — at low volume, API wins. At high volume, self-hosted wins. Break-even for Llama-3-70B-class quality vs. Claude Haiku or GPT-4o-mini is 5–20 B output tokens/month.

**On-demand vs. reserved** — reservations save ~40 %. Break-even is ~60 % utilization sustained for the reservation period.

Draw both curves for your traffic forecast before deciding.

## Pitfalls

- **Comparing list prices.** Reserved rates, volume discounts, and specialist clouds make list prices nearly meaningless. Compare effective rates.
- **Ignoring utilization.** Every hour of idle GPU is paid for. Autoscaling policy is a cost lever.
- **Over-quantizing to "save money."** INT4 saves bandwidth but if the quality loss triggers escalation to a bigger model via a cascade, your bill goes up.
- **Treating cold starts as free.** They cost both latency (user-facing) and dollars (burned for 2–5 minutes of model loading).

<section class="faq">
<h2>FAQ</h2>
<h3>What utilization should I target?</h3>
<p>60–70 % sustained is a good target. Below 40 % you're paying for idle. Above 85 % you have no headroom for spikes.</p>
<h3>Is API or self-hosted cheaper?</h3>
<p>API for low volume (break-even is typically 100M–1B tokens/mo depending on model). Self-hosted above that.</p>
<h3>Does fine-tuning save money?</h3>
<p>Yes, if it lets you use a smaller base model. The smaller model is cheaper to serve. Fine-tune cost is amortized once.</p>
<h3>How much does energy cost matter?</h3>
<p>For cloud-rented GPUs, it's baked into the hourly price. For on-prem, it's a direct line item — see Chapter 4.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://www.cloudzero.com/blog/cloud-gpu-pricing-comparison/" target="_blank" rel="noreferrer">CloudZero — Cloud GPU pricing comparison</a></li>
<li><a href="https://alishafique3.github.io/memory-calculator/" target="_blank" rel="noreferrer">Memory Calculator — interactive sizing tool</a></li>
<li><a href="https://openai.com/api/pricing/" target="_blank" rel="noreferrer">OpenAI API pricing</a></li>
</ul>
</section>

## What's next

[Chapter 2](?module=m9&chapter=calculating-cost) does the detailed &#36;/M tokens math. [Chapter 4](?module=m9&chapter=energy-tokens-per-joule) is the energy companion.
