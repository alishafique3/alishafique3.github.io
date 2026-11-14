<div class="chapter-meta">Module 9 · Chapter 3</div>

# Cloud vs. Self-Hosted Inference Economics

<p class="lede">Four honest paths exist for LLM inference: <strong>managed API</strong> (OpenAI/Anthropic), <strong>serverless inference</strong> (Modal/Replicate/Together), <strong>self-hosted cloud</strong> (your GPUs on Lambda/AWS/GCP), and <strong>on-prem</strong>. Each is cheapest in a different volume regime. This chapter maps the break-even curves.</p>

## The four options side by side

| Option | What you pay | What you own |
|---|---|---|
| Managed API | &#36;/token | App only |
| Serverless | &#36;/GPU-sec | App + model |
| Self-hosted cloud | &#36;/GPU-hr (24×7) | Everything in the cluster |
| On-prem | Capex + power + staff | Everything, including hardware |

## Break-even — managed API vs. self-hosted

Compare managed API rates against your self-hosted &#36;/M tokens:

**Example.** You serve Llama 3.1 70B at ~&#36;0.90 per 1M output tokens self-hosted (M9.2). A comparable API (Anthropic Claude Haiku) costs ~&#36;1.25/1M. Break-even is at the volume where your self-hosted fixed cost (2 × H100 running 24×7 = ~&#36;8,760/month) stops being wasted.

$$ \text{break-even tokens/month} = \frac{\text{monthly fixed self-host cost}}{\text{API price} - \text{self-host variable cost}} $$

At &#36;8,760 fixed, &#36;1.25/M API, &#36;0.90/M variable → ~25 B tokens/month. Below that, API wins. Above that, self-host wins.

For a frontier model (GPT-4-class at &#36;15–30/M), break-even drops to ~0.5–2 B tokens/month. The gap closes fast at higher API prices.

## Break-even — serverless vs. self-hosted

Serverless is cheap for low traffic — you pay only for GPU-seconds used. It's expensive for sustained traffic because the per-second rate is 2–4× hourly.

Example. Modal's H100 is roughly &#36;0.00175/GPU-sec = &#36;6.30/GPU-hr if run continuously. On-demand specialist cloud is &#36;3/hr. Serverless breaks even at ~50 % utilization.

Serverless wins when:

- Traffic is bursty (hours of silence, minutes of activity).
- You can tolerate 2–10 min cold starts, or pay for warm pools.
- You don't want to manage ops.

Serverless loses when:

- Traffic is steady, &gt; 40 % utilization.
- Cold starts are unacceptable.

## Break-even — self-hosted cloud vs. on-prem

On-prem requires capex, power, cooling, staff. Amortized over 3 years at reasonable utilization: ~&#36;1.50–&#36;2.50/GPU-hr. Below specialist clouds, well below hyperscalers.

**Threshold for on-prem to make sense:**

- Utilization &gt; 60 % sustained.
- Workload horizon &gt; 3 years (amortization).
- Platform team capable of running physical infrastructure.
- OR regulatory requirement that forces it.

Most teams don't meet all four. Even at hyperscale, many stay on cloud for flexibility.

## Cost-per-user decomposition

For a user-facing LLM product, slice the economics per active user:

$$ \$/\text{user/month} = \text{API tier} \times \text{avg tokens per user} $$

A chatbot where each active user sends 200 messages/month × 400 tokens each = 80k tokens/user/month.

At a &#36;0.90/1M self-hosted → &#36;0.07/user/month. At a &#36;15/1M frontier API → &#36;1.20/user/month.

If you're charging &#36;20/month per user, both work. If you're charging &#36;5/month, you can't afford the frontier API.

## Reserved vs. on-demand

Hyperscaler reservations:

- 1-year: ~30 % discount.
- 3-year: ~50 % discount.

Break-even is the utilization at which reservation cost = on-demand cost. Usually 60 % for 1-year, 55 % for 3-year.

Specialist cloud reservations are smaller discounts (10–20 %) but still worth it for steady workloads.

## Spot GPUs — cheap but risky

Spot instances are 60–90 % off on-demand. Reclaimed at short notice. Workable for:

- Offline batch (M6.7).
- Non-critical workloads with warm standbys.
- Development.

Unworkable for:

- Real-time chat without graceful fallback.
- Session-affine workloads where state can't rehydrate.

## The hidden cost: your time

The self-hosted path saves money on GPU-hours. It costs engineering time to:

- Run Kubernetes clusters.
- Configure autoscaling that doesn't flap.
- Debug CUDA OOMs at 3 AM.
- Keep up with new models and engine releases.

Rule of thumb: a self-hosted stack adds 0.5–2 FTE of ops work vs. API usage. At &#36;200k/FTE loaded cost, that's &#36;100k–&#36;400k/year. The savings have to clear that threshold.

## A real-world decision matrix

| Monthly tokens | Latency need | Compliance | Pick |
|---|---|---|---|
| &lt; 100M | Normal | None | Managed API |
| 100M–1B | Normal | None | Managed API or serverless |
| 1B–10B | Normal | None | Self-hosted cloud |
| 10B–100B | Normal | None | Self-hosted cloud, reserved |
| &gt; 100B | Tight | None | Self-hosted cloud or on-prem |
| Any | Any | Regulated | On-prem or BYOC |
| Any | Any | Real-time | Self-hosted (API latency adds) |

## Pitfalls

- **Comparing list prices.** The effective rate after reservations, discounts, and your utilization is what matters.
- **Ignoring ops cost.** Self-host saves GPU dollars at the cost of FTE dollars. Count both.
- **Over-optimistic utilization forecast.** Plan for 50 %, not 80 %. You'll feel smart when it's 65 %.
- **Switching platforms mid-launch.** Plumbing differences hide. Pick the platform, commit for a year.

<section class="faq">
<h2>FAQ</h2>
<h3>Is API always more expensive than self-hosted?</h3>
<p>No. API wins on low volume because fixed self-host costs don't amortize. Draw the break-even.</p>
<h3>Can I mix API and self-host?</h3>
<p>Yes — common pattern. Self-host the bulk tier, API for the frontier tier.</p>
<h3>Are BYOC providers worth the margin?</h3>
<p>If they save you an ops hire, yes. The 5–20 % margin is often less than an FTE.</p>
<h3>How do I estimate ops burden?</h3>
<p>Count the on-call pages per week × 15 min each. For a single-model, mature stack, &lt; 1 hour/week. For multi-model, multi-region, 10+ hours/week.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://openai.com/api/pricing/" target="_blank" rel="noreferrer">OpenAI API pricing</a></li>
<li><a href="https://www.anthropic.com/pricing" target="_blank" rel="noreferrer">Anthropic API pricing</a></li>
<li><a href="https://modal.com/pricing" target="_blank" rel="noreferrer">Modal — serverless GPU pricing</a></li>
<li><a href="https://lambdalabs.com/service/gpu-cloud" target="_blank" rel="noreferrer">Lambda Labs GPU Cloud</a></li>
</ul>
</section>

## What's next

The energy chapters — [Chapter 4](?module=m9&chapter=energy-tokens-per-joule) and [Chapter 5](?module=m9&chapter=power-energy-optimization) — show the second half of the economic picture.
