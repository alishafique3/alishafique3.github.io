<div class="chapter-meta">Module 3 · Chapter 7</div>

# Cloud, Hybrid, and On-Premises Deployments

<p class="lede">Where your GPUs physically live shapes everything: cost, latency, compliance, the shape of your ops team. There are now four honest places to deploy: hyperscalers (AWS/GCP/Azure), specialist GPU clouds (Lambda, RunPod, CoreWeave, Nebius), BYOC (your account on a managed stack), and on-prem. This chapter is a straight-talk comparison.</p>

## The four venues

| Venue | H100 &#36;/hr (2026) | Setup time | Compliance |
|---|---|---|---|
| AWS / GCP / Azure | &#36;6.88 – &#36;15.12 | Hours | SOC2, HIPAA, FedRAMP available |
| Specialist cloud (Lambda, RunPod, Nebius, CoreWeave) | &#36;2.49 – &#36;4.49 | Hours | SOC2 common; others vary |
| Community spot (RunPod, Vast.ai) | &#36;1.19 – &#36;1.99 | Minutes | Not for regulated data |
| On-prem | Capex equivalent &#36;1.50 – &#36;2.50/hr | Months | Full control |

## Hyperscalers — the premium default

AWS, GCP, Azure all offer H100, H200, and (patchy) B200. You pay 2–5× specialist-cloud rates for: SLA-backed availability, IAM/VPC integration, compliance certifications, and the same bill as everything else you're already running. Reserved instances cut 30–50% for 1-year commitments.

Pick a hyperscaler when: you need enterprise-grade SLAs, your data already lives there, or compliance dictates it.

## Specialist GPU clouds

Lambda Labs, RunPod, Nebius, CoreWeave, Together, and Scaleway target AI workloads directly. H100 at &#36;2.49–&#36;4.49/hr. Trade-offs: fewer regions, variable SLAs, less IAM machinery. For most "I just need GPUs" cases they are the economic win.

Pick a specialist when: cost matters, workload is primarily GPU compute, your data governance is portable.

## BYOC — Bring Your Own Cloud

You run your own AWS/GCP account; a third party (Anyscale, Together AI, Fireworks, Modal) runs a managed control plane that provisions GPUs into your account. You get hyperscaler pricing with managed ops. Good for teams that need compliance and don't want to build the platform.

## On-prem — the long-horizon math

An HGX H100 chassis (8 × H100) costs ~&#36;250–320 k new, add &#36;50 k for networking and racks. Power is 10 kW; at &#36;0.10/kWh that's ~&#36;8,800/year. Over 36 months of 24 × 7 utilization:

$$ \$/\text{GPU-hr (on-prem)} \approx \frac{\$300{,}000}{8 \times 24 \times 365 \times 3} + \frac{\$8{,}800}{8 \times 24 \times 365} \approx \$1.56 $$

Plus: cooling, floor space, staff, replacement parts. A reasonable all-in is &#36;2.00–&#36;2.50/GPU-hr. Below specialist clouds, well below hyperscalers, but *only if you keep the hardware busy*. At 50% utilization, the effective number doubles.

Pick on-prem when: data must stay physical for regulatory reasons, your utilization is predictably > 60%, and you have a platform team that can run physical infrastructure.

## Hybrid patterns that work

- **Baseline + burst.** On-prem for steady-state, cloud for traffic spikes. Needs a smart gateway.
- **Prod on hyperscaler, dev on specialist.** Prototyping cheap, prod on the compliant stack.
- **Batch on spot, serve on reserved.** Offline enrichment on cheap spot GPUs, real-time serving on reserved capacity.

## A practical decision table

| Your situation | Pick |
|---|---|
| Regulated, large steady workload | On-prem or BYOC on hyperscaler |
| Regulated, small workload | Hyperscaler |
| Unregulated, cost-sensitive, steady | Specialist cloud with reservation |
| Unregulated, cost-sensitive, batch | Spot GPUs on specialist/RunPod/Vast |
| Dev, small team, varied needs | Specialist cloud, on-demand |

## Pitfalls

- **Choosing on-prem for cost, then utilizing at 30%.** Idle GPUs are more expensive than hyperscaler on-demand.
- **Spot GPU surprises.** Spot instances get reclaimed. Any inference stack running on spot needs instant-restart machinery.
- **Mixing regions without latency planning.** Serving US users from an EU GPU adds ~90 ms per request. Place GPUs near users.
- **Forgetting egress.** Serving 1 TB/day of output tokens across regions can double the monthly bill.

<section class="faq">
<h2>FAQ</h2>
<h3>Is BYOC really hyperscaler pricing?</h3>
<p>Yes, you pay AWS/GCP rates for the GPUs. The BYOC provider charges a small management fee or margin on top.</p>
<h3>What is "spot" and when is it safe?</h3>
<p>Unused capacity sold at a discount, reclaimable at short notice. Safe for offline batch and preemption-tolerant workloads; not safe for real-time serving unless you have warm standbys.</p>
<h3>Can I move from specialist cloud to hyperscaler later?</h3>
<p>Easy at the compute level (containers and models are portable). Hard if you've built around a specific control plane (serverless, managed scheduler).</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://lambdalabs.com/service/gpu-cloud" target="_blank" rel="noreferrer">Lambda Labs GPU Cloud</a></li>
<li><a href="https://runpod.io" target="_blank" rel="noreferrer">RunPod GPU Cloud</a></li>
<li><a href="https://nebius.com/" target="_blank" rel="noreferrer">Nebius AI Cloud</a></li>
<li><a href="https://www.coreweave.com/" target="_blank" rel="noreferrer">CoreWeave</a></li>
<li><a href="https://www.cloudzero.com/blog/cloud-gpu-pricing-comparison/" target="_blank" rel="noreferrer">CloudZero — Cloud GPU pricing comparison</a></li>
</ul>
</section>

## What's next

Module 3 is done. Head to [Module 4 — Model Optimization](?module=m4&chapter=intro-model-optimization) for the levers that reduce the model's cost before it ever hits the GPU.
